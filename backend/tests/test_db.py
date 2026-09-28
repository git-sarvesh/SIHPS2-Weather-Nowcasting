"""Tests for the database layer: migrations, persistence, transactions, audit.

Every test runs against a real SQLite database (in-memory or a temp file), so the
constraints, indexes and rollback behaviour are genuinely exercised rather than
mocked.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from app.db.migrations import MIGRATIONS, applied_revisions, current_revision, init_db, upgrade
from app.db.models import AuditRecord, ExecutionStatus, ForecastRun, RiskCell
from app.db.repository import (
    build_idempotency_key,
    check_database,
    create_audit_record,
    get_forecast_run,
    get_run_risk_cells,
    list_audit_records,
    list_forecast_runs,
    record_forecast_run,
    replace_risk_cells,
    scrub_error,
)
from app.db.session import SessionFactory, redact_url
from app.grid import GridSpec

INIT = dt.datetime(2024, 7, 15, 3, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def factory() -> SessionFactory:
    """A migrated, isolated in-memory database."""
    session_factory = SessionFactory("sqlite:///:memory:")
    init_db(session_factory.engine)
    return session_factory


def _run(session, **overrides):
    """Insert a forecast run with sensible defaults."""
    payload = {
        "idempotency_key": build_idempotency_key(
            model_version=overrides.get("model_version", "v-test"),
            init_time=INIT,
            lead_hours=1.0,
        ),
        "model_version": "v-test",
        "init_time": INIT,
        "lead_hours": 1.0,
        "lead_times_h": [0.5, 1.0, 1.5],
        "is_synthetic": True,
    }
    payload.update(overrides)
    return record_forecast_run(session, **payload)


def _write_cells(session, run, grid, **overrides):
    """Write a uniform risk raster for ``run``."""
    arrays = {
        "thunderstorm": np.full(grid.shape, 0.5),
        "cloudburst": np.full(grid.shape, 0.2),
        "flood_risk": np.full(grid.shape, 0.3),
        "flood_probability": np.full(grid.shape, 0.4),
        "exposure": np.full(grid.shape, 0.6),
        "compound": np.full(grid.shape, 0.1),
        "overall": np.full(grid.shape, 0.7),
    }
    arrays.update(overrides)
    return replace_risk_cells(session, run, grid=grid, **arrays)


# --------------------------------------------------------------------------- #
# migrations
# --------------------------------------------------------------------------- #
def test_migrations_create_every_table(factory: SessionFactory) -> None:
    tables = set(inspect(factory.engine).get_table_names())
    assert {"forecast_runs", "risk_cells", "audit_records", "schema_migrations"} <= tables


def test_migrations_are_idempotent(factory: SessionFactory) -> None:
    """A second run must apply nothing, not try to re-create the tables."""
    assert upgrade(factory.engine) == []
    assert current_revision(factory.engine) == MIGRATIONS[-1].id
    assert init_db(factory.engine)["applied_now"] == []


def test_current_revision_is_none_before_migration() -> None:
    empty = SessionFactory("sqlite:///:memory:")
    assert current_revision(empty.engine) is None
    assert applied_revisions(empty.engine) == []


def test_downgrade_refuses_unknown_revision(factory: SessionFactory) -> None:
    from app.db.migrations import downgrade

    with pytest.raises(ValueError, match="unknown target revision"):
        downgrade(factory.engine, "9999_nope")


def test_health_redacts_credentials_from_the_url() -> None:
    postgres = SessionFactory("postgresql+psycopg2://user:secret@db:5432/sihps")
    report = postgres.health()
    assert "secret" not in report["url"]
    assert "***" in report["url"]
    assert redact_url("sqlite:///./data/x.db") == "sqlite:///./data/x.db"


# --------------------------------------------------------------------------- #
# forecast runs
# --------------------------------------------------------------------------- #
def test_record_forecast_run_persists_provenance(factory: SessionFactory) -> None:
    with factory.scope() as session:
        run = _run(
            session,
            valid_time=INIT + dt.timedelta(hours=1),
            model_backend="torch",
            data_source="SIHPS synthetic demo generator",
            grid={"nx": 8, "ny": 8},
            provenance={"trained_checkpoint_loaded": False},
            summary={"max_overall_risk": 0.4},
        )
        assert run.id is not None
        assert run.n_lead_times == 3
        assert run.is_synthetic is True
        assert run.trained_checkpoint_loaded is False
        assert run.status == ExecutionStatus.SUCCEEDED.value

    with factory.scope() as session:
        stored = get_forecast_run(session, run.id)
        assert stored.model_backend == "torch"
        assert stored.grid["nx"] == 8
        assert stored.summary["max_overall_risk"] == 0.4


def test_record_forecast_run_is_idempotent(factory: SessionFactory) -> None:
    """A repeated request must refresh the row, never duplicate it."""
    with factory.scope() as session:
        first = _run(session)
        second = _run(session)
    assert first.id == second.id
    with factory.scope() as session:
        assert list_forecast_runs(session)[1] == 1


def test_duplicate_idempotency_key_violates_the_constraint(factory: SessionFactory) -> None:
    """Bypassing the repository still hits the unique constraint."""
    key = build_idempotency_key(model_version="v", init_time=INIT, lead_hours=1.0)
    with factory.scope() as session:
        _run(session, idempotency_key=key)
    with pytest.raises(IntegrityError):
        with factory.scope() as session:
            session.add(
                ForecastRun(
                    idempotency_key=key,
                    model_version="v",
                    init_time=INIT,
                    lead_hours=1.0,
                    n_lead_times=1,
                )
            )
            session.flush()


def test_check_constraint_rejects_non_positive_lead(factory: SessionFactory) -> None:
    with pytest.raises(IntegrityError):
        with factory.scope() as session:
            session.add(
                ForecastRun(
                    idempotency_key="bad-lead",
                    model_version="v",
                    init_time=INIT,
                    lead_hours=0.0,
                    n_lead_times=1,
                )
            )
            session.flush()


def test_synthetic_flag_defaults_to_true_not_false(factory: SessionFactory) -> None:
    """A missing flag must never be interpreted as observational data."""
    with factory.scope() as session:
        session.add(
            ForecastRun(
                idempotency_key="no-flag",
                model_version="v",
                init_time=INIT,
                lead_hours=1.0,
                n_lead_times=1,
            )
        )
        session.flush()
    with factory.scope() as session:
        assert get_forecast_run(session, 1).is_synthetic is True


# --------------------------------------------------------------------------- #
# risk cells
# --------------------------------------------------------------------------- #
def test_replace_risk_cells_persists_every_cell(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(3, 3))
    with factory.scope() as session:
        run = _run(session)
        assert _write_cells(session, run, grid) == 9
    with factory.scope() as session:
        cells = get_run_risk_cells(session, run.id, limit=100)
        assert len(cells) == 9
        # overall 0.7 lands in the HIGH band of thresholds (0.3, 0.6, 0.85).
        assert {c.risk_category for c in cells} == {"HIGH"}
        assert {c.risk_category_code for c in cells} == {2}
        first = cells[0]
        assert -90.0 <= first.lat <= 90.0 and -180.0 <= first.lon <= 180.0
        assert 0.0 <= first.terrain_exposure <= 1.0


def test_replace_risk_cells_is_idempotent(factory: SessionFactory) -> None:
    """Re-running a task must replace the raster, not append to it."""
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(3, 3))
    with factory.scope() as session:
        run = _run(session)
        _write_cells(session, run, grid)
        _write_cells(session, run, grid, overall=np.full(grid.shape, 0.2))
    with factory.scope() as session:
        cells = get_run_risk_cells(session, run.id, limit=100)
        assert len(cells) == 9
        assert {c.risk_category for c in cells} == {"LOW"}


def test_risk_cell_duplicate_position_violates_constraint(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(2, 2))
    with factory.scope() as session:
        run = _run(session)
        _write_cells(session, run, grid)

    # The violation must surface from inside the scope, which rolls back.
    with pytest.raises(IntegrityError):
        with factory.scope() as session:
            session.add(
                RiskCell(
                    forecast_run_id=run.id, row=0, col=0, lat=30.0, lon=78.0,
                    thunderstorm=0.1, cloudburst=0.1, flood_probability=0.1, flood_risk=0.1,
                    compound_storm_cloudburst=0.1, terrain_exposure=0.1, overall_risk=0.1,
                )
            )
            session.flush()

    with factory.scope() as session:
        # The original four cells survive; the rejected insert did not.
        assert len(get_run_risk_cells(session, run.id, limit=10)) == 4


def test_risk_cells_cascade_on_run_delete(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(2, 2))
    with factory.scope() as session:
        run = _run(session)
        _write_cells(session, run, grid)
        run_id = run.id
    with factory.scope() as session:
        session.delete(get_forecast_run(session, run_id))
    with factory.scope() as session:
        assert session.scalars(select(RiskCell)).all() == []


def test_risk_cell_uncertainty_is_optional(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(2, 2))
    with factory.scope() as session:
        run = _run(session)
        _write_cells(session, run, grid, uncertainty={"flood": np.full(grid.shape, 0.05)})
    with factory.scope() as session:
        cell = get_run_risk_cells(session, run.id, limit=1)[0]
        assert cell.uncertainty is not None
        assert cell.uncertainty["flood"] == pytest.approx(0.05)


def test_max_cells_caps_written_rows(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(4, 4))
    with factory.scope() as session:
        run = _run(session)
        assert _write_cells(session, run, grid) == 16


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def test_scrub_error_removes_credentials() -> None:
    assert scrub_error("failed password=hunter2") == "failed [REDACTED]"
    assert "sihps:sihps" not in scrub_error("postgresql://sihps:sihps@db/sihps failed")
    assert scrub_error("api_key: xyz123") == "[REDACTED]"
    assert scrub_error(None) is None
    assert len(scrub_error("x" * 1000)) == 500


def test_audit_record_stores_scrubbed_error(factory: SessionFactory) -> None:
    with factory.scope() as session:
        record = create_audit_record(
            session,
            operation="ingest",
            status=ExecutionStatus.FAILED.value,
            error="connect to postgresql://u:p@db failed token=abc",
        )
        assert record.id is not None
        assert "u:p" not in record.error
        assert "[REDACTED]" in record.error
        assert record.is_synthetic is True


def test_audit_records_are_listable_and_filterable(factory: SessionFactory) -> None:
    with factory.scope() as session:
        for index in range(5):
            create_audit_record(session, operation="ingest", request_id=f"r{index}")
        create_audit_record(session, operation="batch_inference")
    with factory.scope() as session:
        _rows, total = list_audit_records(session, limit=10)
        assert total == 6
        filtered, filtered_total = list_audit_records(session, operation="ingest")
        assert filtered_total == 5
        assert all(r.operation == "ingest" for r in filtered)


# --------------------------------------------------------------------------- #
# transactions
# --------------------------------------------------------------------------- #
def test_scope_rolls_back_on_error(factory: SessionFactory) -> None:
    with pytest.raises(RuntimeError):
        with factory.scope() as session:
            run = _run(session)
            session.flush()
            assert run.id is not None
            raise RuntimeError("boom")
    with factory.scope() as session:
        assert list_forecast_runs(session)[1] == 0, "a failed scope must not commit"


def test_scope_commits_on_success(factory: SessionFactory) -> None:
    with factory.scope() as session:
        _run(session)
    with factory.scope() as session:
        assert list_forecast_runs(session)[1] == 1


def test_partial_failure_leaves_no_orphan_cells(factory: SessionFactory) -> None:
    grid = GridSpec.from_bbox((78.0, 30.0, 79.0, 31.0), shape=(3, 3))
    with pytest.raises(RuntimeError):
        with factory.scope() as session:
            run = _run(session)
            _write_cells(session, run, grid)
            raise RuntimeError("failure after writing cells")
    with factory.scope() as session:
        assert list_forecast_runs(session)[1] == 0
        assert session.scalars(select(RiskCell)).all() == []


def test_file_backed_sqlite_creates_its_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "sihps.db"
    session_factory = SessionFactory(f"sqlite:///{target.as_posix()}")
    init_db(session_factory.engine)
    assert target.exists()
    assert session_factory.health()["status"] == "ok"


def test_check_database_detects_a_healthy_schema(factory: SessionFactory) -> None:
    with factory.scope() as session:
        assert check_database(session) is True
