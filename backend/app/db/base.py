"""Declarative base and shared column helpers.

Kept dependency-light so Alembic's ``env.py`` can import the metadata without
pulling in the FastAPI layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Explicit naming convention so Alembic can autogenerate reversible migrations
#: on SQLite (which does not support ``ALTER ... DROP CONSTRAINT``).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Declarative base for every SIHPS table."""

    metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of the mapped columns (for JSON responses)."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def utcnow() -> datetime:
    """Timezone-aware current UTC time (mirrors ``app.ingestion.base.utcnow``)."""
    return datetime.now(tz=timezone.utc)


def utcnow_column() -> Mapped[datetime]:
    """A ``created_at`` column defaulting to now (server-side where possible)."""
    return mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
