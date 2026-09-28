"""Database layer: SQLAlchemy models, session management and migrations.

Design notes
------------
* **SQLite for dev/tests, PostgreSQL for production.** The URL comes from
  ``SIHPS_DATABASE_URL``; ``db_backend`` only selects driver conveniences, so a
  PostgreSQL/PostGIS URL works without code changes. No PostGIS-specific column
  types are used, which keeps the schema portable - risk cells store plain
  ``lat``/``lon`` floats and are indexed, so PostGIS can be adopted later by
  adding a generated geometry column.
* **Synthetic data is always flagged.** Every table that can hold model output
  carries ``is_synthetic`` and a provenance JSON payload, so a synthetic
  demonstration can never be read back as an observationally validated forecast.
* **No secrets are persisted.** Audit records store operation, status, provenance
  and an error *message*; the message is scrubbed of anything that looks like a
  credential before it is written.
"""

from app.db.base import Base, metadata
from app.db.models import AuditRecord, ForecastRun, RiskCell
from app.db.session import (
    DatabaseUnavailableError,
    SessionFactory,
    get_session,
    session_scope,
)

__all__ = [
    "AuditRecord",
    "Base",
    "DatabaseUnavailableError",
    "ForecastRun",
    "RiskCell",
    "SessionFactory",
    "get_session",
    "metadata",
    "session_scope",
]
