"""Minimal, versioned schema migrations.

Why not Alembic?
----------------
Alembic is not a declared dependency of this project and is not installed, so
adopting it would add an undeclared requirement. Instead this module implements
the same discipline with no extra dependency:

* an ordered list of :class:`Migration` steps, each with an ``id`` and ``upgrade``
  callable;
* a ``schema_migrations`` table recording which ids have been applied;
* upgrades run inside a transaction per step, in order, exactly once;
* ``downgrade`` reverses to a target revision where the step declares it.

It is intentionally not a general-purpose migration framework. When Alembic is
added later, this module can be replaced by autogenerate with the same effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from sqlalchemy import Connection, MetaData, Table, Column, String, text
from sqlalchemy.engine import Engine

from app.db.base import Base
from app.logging_conf import get_logger

logger = get_logger("db.migrations")

__all__ = [
    "MIGRATIONS",
    "Migration",
    "applied_revisions",
    "current_revision",
    "downgrade",
    "init_db",
    "upgrade",
]

VERSION_TABLE = "schema_migrations"


@dataclass(frozen=True, slots=True)
class Migration:
    """One reversible schema step."""

    id: str
    upgrade: Callable[[Connection], None]
    downgrade: Callable[[Connection], None] | None = None
    description: str = ""


def _version_table(metadata: MetaData) -> Table:
    return Table(
        VERSION_TABLE,
        metadata,
        Column("id", String(64), primary_key=True),
        Column("applied_at", String(32), nullable=False),
        Column("description", String(256), nullable=True),
        extend_existing=True,
    )


def _ensure_version_table(connection: Connection) -> Table:
    table = _version_table(MetaData())
    table.create(bind=connection, checkfirst=True)
    return table


def _create_all(connection: Connection) -> None:
    """Initial migration: create every table declared on :class:`Base`."""
    # Importing the models registers their tables on Base.metadata.
    from app.db import models  # noqa: F401
    from app.db.base import Base

    Base.metadata.create_all(bind=connection)


def _add_dataset_provenance(connection: Connection) -> None:
    """Phase 5: add ``dataset_provenance`` (additive, backward compatible).

    Creates only the new table. It adds no column to an existing table, so a
    database on the previous revision keeps working and rolling back is just a
    table drop.
    """
    from app.db import models  # noqa: F401  (registers the table on Base)
    from app.db.base import Base

    table = Base.metadata.tables["dataset_provenance"]
    table.create(bind=connection, checkfirst=True)


def _add_provenance_integrity(connection: Connection) -> None:
    """Phase 8.3: add cryptographic and metadata columns to ``dataset_provenance``.

    Additive and backward compatible. Every new column is nullable, so a row
    written by an earlier revision keeps working and reads back with ``None``,
    which every verifier treats as *unknown* - a failure for observational use,
    never a pass. No existing column is altered or dropped, so downgrade is a
    no-op rather than a destructive table drop.
    """
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(connection)
    if "dataset_provenance" not in inspector.get_table_names():
        # The table is created by migration 0002; nothing to extend.
        return
    existing = {c["name"] for c in inspector.get_columns("dataset_provenance")}
    additions = (
        # SHA-256 of the original source bytes; NULL means never recorded.
        ("source_sha256", "VARCHAR(64)"),
        # True only when those bytes still exist and can be re-hashed.
        ("source_bytes_available", "BOOLEAN"),
        # Organisation that provided the data.
        ("provider", "VARCHAR(64)"),
        # channel -> unit mapping, stored as JSON.
        ("channel_units", "JSON"),
        # Explicit spatial coverage declaration.
        ("coverage", "VARCHAR(128)"),
        # Acquisition time, distinct from ingest time.
        ("acquired_at_src", "TIMESTAMP"),
    )
    for name, sql_type in additions:
        if name in existing:
            continue
        connection.execute(
            text(
                f"ALTER TABLE dataset_provenance ADD COLUMN {name} {sql_type}"
            )
        )


def _drop_provenance_integrity(connection: Connection) -> None:
    """Reverse 0003. Additive columns are dropped; no data is lost elsewhere."""
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(connection)
    if "dataset_provenance" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("dataset_provenance")}
    for name in (
        "source_sha256", "source_bytes_available", "provider",
        "channel_units", "coverage", "acquired_at_src",
    ):
        if name in existing:
            connection.execute(
                text(f"ALTER TABLE dataset_provenance DROP COLUMN {name}")
            )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        id="0001_initial_schema",
        upgrade=_create_all,
        description="forecast_runs, risk_cells, audit_records",
    ),
    Migration(
        id="0002_dataset_provenance",
        upgrade=_add_dataset_provenance,
        description="dataset_provenance: reproducible dataset records with dedupe key",
    ),
    Migration(
        id="0003_provenance_integrity",
        upgrade=_add_provenance_integrity,
        downgrade=_drop_provenance_integrity,
        description=(
            "dataset_provenance: source_sha256, source_bytes_available, provider, "
            "channel_units, coverage, acquired_at_src (additive, nullable)"
        ),
    ),
)


def applied_revisions(engine: Engine) -> list[str]:
    """Ids already applied, in order."""
    with engine.connect() as connection:
        table = _ensure_version_table(connection)
        rows = connection.execute(
            text(f"SELECT id FROM {VERSION_TABLE} ORDER BY applied_at, id")
        ).fetchall()
        return [row[0] for row in rows]


def current_revision(engine: Engine) -> str | None:
    """The newest applied revision id, or ``None`` for an empty database."""
    revisions = applied_revisions(engine)
    return revisions[-1] if revisions else None


def upgrade(engine: Engine, migrations: Sequence[Migration] = MIGRATIONS) -> list[str]:
    """Apply every pending migration in order; returns the ids applied.

    Each step commits on its own, so a failure leaves the database at the last
    good revision rather than half-migrated.
    """
    from datetime import datetime, timezone

    done = set(applied_revisions(engine))
    applied: list[str] = []
    for migration in migrations:
        if migration.id in done:
            continue
        logger.info("applying migration", extra={"revision": migration.id})
        with engine.begin() as connection:
            migration.upgrade(connection)
            table = _ensure_version_table(connection)
            connection.execute(
                table.insert().values(
                    id=migration.id,
                    applied_at=datetime.now(tz=timezone.utc).isoformat(),
                    description=migration.description,
                )
            )
        applied.append(migration.id)
    if applied:
        logger.info("migrations complete", extra={"applied": applied})
    return applied


def downgrade(engine: Engine, to_revision: str | None = None) -> list[str]:
    """Revert applied migrations back to ``to_revision``.

    Steps without a ``downgrade`` callable raise, rather than silently leaving
    the schema in a state that does not match the recorded revision.
    """
    if to_revision is None:
        return []
    known = {m.id: m for m in MIGRATIONS}
    if to_revision not in known:
        raise ValueError(f"unknown target revision {to_revision!r}; known: {sorted(known)}")
    order = [m.id for m in MIGRATIONS]
    cutoff = order.index(to_revision)
    reverted: list[str] = []
    for migration in reversed(MIGRATIONS):
        if migration.id not in applied_revisions(engine):
            continue
        if order.index(migration.id) <= cutoff:
            break
        if migration.downgrade is None:
            raise NotImplementedError(
                f"migration {migration.id!r} declares no downgrade; refusing to guess"
            )
        with engine.begin() as connection:
            migration.downgrade(connection)
            connection.execute(text(f"DELETE FROM {VERSION_TABLE} WHERE id = :id"), {"id": migration.id})
        reverted.append(migration.id)
    return reverted


def init_db(engine: Engine) -> dict[str, Any]:
    """Bring a database up to the latest revision; safe to call repeatedly."""
    applied = upgrade(engine)
    return {
        "status": "ok",
        "current_revision": current_revision(engine),
        "applied_now": applied,
        "dialect": engine.dialect.name,
    }
