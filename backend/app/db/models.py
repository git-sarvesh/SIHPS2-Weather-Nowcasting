"""SQLAlchemy models: forecast runs, per-cell risk records and audit trail.

Three tables, deliberately:

``forecast_runs``
    One row per model invocation: issue time, selected lead time and its valid
    time, model/checkpoint identity, provenance JSON, the synthetic flag and the
    execution status. A unique constraint on ``idempotency_key`` is what makes
    scheduled re-runs safe.
``risk_cells``
    Optional per-cell risk rows for a run. Kept apart from the run row because a
    full AOI is tens of thousands of cells and most queries want the run metadata
    without the raster.
``audit_records``
    Append-only trail of operations, statuses, provenance and error messages.
    Never stores credentials.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

__all__ = [
    "AuditRecord",
    "DatasetProvenanceRow",
    "ExecutionStatus",
    "ForecastRun",
    "RiskCategory",
    "RiskCell",
    "RunKind",
]


class ExecutionStatus(str, Enum):
    """Lifecycle of a run or task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"

    @property
    def is_terminal(self) -> bool:
        return self.value in {
            self.SUCCEEDED.value,
            self.FAILED.value,
            self.SKIPPED.value,
            self.DISABLED.value,
        }


class RunKind(str, Enum):
    """What produced a forecast run."""

    API = "api"
    BATCH = "batch"
    TRAINING_EVAL = "training_eval"
    DEMO = "demo"


class RiskCategory(str, Enum):
    """Mirrors ``app.services.risk_engine.RISK_CATEGORIES`` (0-3)."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

    @classmethod
    def from_code(cls, code: int | None) -> str:
        """Map a numeric category code to its name (``None`` -> LOW)."""
        if code is None:
            return cls.LOW.value
        order = [cls.LOW.value, cls.MODERATE.value, cls.HIGH.value, cls.EXTREME.value]
        index = int(code)
        return order[index] if 0 <= index < len(order) else cls.LOW.value


class ForecastRun(Base):
    """One model invocation and everything needed to reproduce its output."""

    __tablename__ = "forecast_runs"
    __table_args__ = (
        # Idempotency: the same logical request must not create two runs.
        UniqueConstraint("idempotency_key", name="uq_forecast_runs_idempotency_key"),
        CheckConstraint("lead_hours > 0", name="lead_hours_positive"),
        CheckConstraint("n_lead_times >= 0", name="n_lead_times_non_negative"),
        Index("ix_forecast_runs_init_time", "init_time"),
        Index("ix_forecast_runs_status_created", "status", "created_at"),
        Index("ix_forecast_runs_model_version", "model_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: Stable key derived from (event, init time, model, lead) for de-duplication.
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)

    kind: Mapped[str] = mapped_column(String(32), nullable=False, default=RunKind.API.value)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ExecutionStatus.PENDING.value
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- timing -------------------------------------------------------------
    init_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lead_hours: Mapped[float] = mapped_column(Float, nullable=False)
    n_lead_times: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    # -- model identity -----------------------------------------------------
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    checkpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trained_checkpoint_loaded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # -- provenance ---------------------------------------------------------
    #: True for anything derived from the synthetic generator. Never inferred from
    #: the environment: it is written explicitly at insert time.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    data_source: Mapped[str | None] = mapped_column(String(256), nullable=True)
    grid: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    risk_cells: Mapped[list[RiskCell]] = relationship(
        back_populates="forecast_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<ForecastRun id={self.id} key={self.idempotency_key!r} "
            f"status={self.status!r} synthetic={self.is_synthetic}>"
        )


class RiskCell(Base):
    """Terrain-aware risk for one grid cell of one forecast run."""

    __tablename__ = "risk_cells"
    __table_args__ = (
        CheckConstraint("lat >= -90 AND lat <= 90", name="lat_in_range"),
        CheckConstraint("lon >= -180 AND lon <= 180", name="lon_in_range"),
        CheckConstraint(
            "risk_category_code >= 0 AND risk_category_code <= 3", name="category_code_in_range"
        ),
        # A cell may only appear once per run.
        UniqueConstraint("forecast_run_id", "row", "col", name="uq_risk_cells_run_row_col"),
        Index("ix_risk_cells_run", "forecast_run_id"),
        Index("ix_risk_cells_run_risk", "forecast_run_id", "overall_risk"),
        Index("ix_risk_cells_geo", "lat", "lon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="CASCADE"), nullable=False
    )

    # -- location -----------------------------------------------------------
    row: Mapped[int] = mapped_column(Integer, nullable=False)
    col: Mapped[int] = mapped_column(Integer, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)

    # -- hazards and risk ---------------------------------------------------
    thunderstorm: Mapped[float] = mapped_column(Float, nullable=False)
    cloudburst: Mapped[float] = mapped_column(Float, nullable=False)
    flood_probability: Mapped[float] = mapped_column(Float, nullable=False)
    flood_risk: Mapped[float] = mapped_column(Float, nullable=False)
    compound_storm_cloudburst: Mapped[float] = mapped_column(Float, nullable=False)
    terrain_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    overall_risk: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category_code: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_category: Mapped[str] = mapped_column(
        String(16), nullable=False, default=RiskCategory.LOW.value
    )

    #: Per-hazard MC spread when an ensemble was run, else ``{}``.
    uncertainty: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    forecast_run: Mapped[ForecastRun] = relationship(back_populates="risk_cells")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RiskCell run={self.forecast_run_id} row={self.row} col={self.col}>"


class DatasetProvenanceRow(Base):
    """One ingested dataset and everything needed to reproduce it.

    Separate from ``forecast_runs`` because the lifecycle differs: datasets are
    ingested once (and de-duplicated by ``ingest_key``) and then referenced by
    many runs, whereas a run describes one model invocation.

    The unique constraint on ``ingest_key`` is the duplicate-ingestion guard -
    re-running a batch over the same files becomes a no-op rather than a second
    copy. No credential is ever stored: the record carries source metadata and a
    checksum, never a token.
    """

    __tablename__ = "dataset_provenance"
    __table_args__ = (
        UniqueConstraint("ingest_key", name="uq_dataset_provenance_ingest_key"),
        CheckConstraint(
            "data_class in ('observation','reanalysis','derived','interpolated','synthetic')",
            name="data_class_valid",
        ),
        Index("ix_dataset_provenance_source", "source"),
        Index("ix_dataset_provenance_acquired_at", "acquired_at"),
        Index("ix_dataset_provenance_data_class", "data_class"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: SHA-256 over (source, product, acquired_at, checksum); the dedupe key.
    ingest_key: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    product: Mapped[str] = mapped_column(String(128), nullable=False)
    #: observation | reanalysis | derived | interpolated | synthetic
    data_class: Mapped[str] = mapped_column(String(16), nullable=False)

    #: Observation/valid time of the data itself.
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: When this system read it.
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    # -- spatial -----------------------------------------------------------
    min_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    crs: Mapped[str] = mapped_column(String(32), nullable=False, default="EPSG:4326")
    source_resolution: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    processed_resolution: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    # -- content -----------------------------------------------------------
    source_cadence: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    processed_cadence: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    n_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Variables/units, QC results, resampling steps, processing config.
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    attribution: Mapped[str] = mapped_column(Text, nullable=False, default="")
    license_note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ---------------------------------------------- phase 8.3 (migration 0003)
    #: SHA-256 of the ORIGINAL source bytes. ``None``/"" means never recorded,
    #: which every verifier treats as unknown - a failure, never a pass. A digest
    #: is never synthesised or derived from a filename.
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: True only when those bytes still exist and can be re-hashed.
    source_bytes_available: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    #: Organisation that provided the data (e.g. "NCMRWF", "IMD").
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: ``channel name -> physical unit`` for the fields this dataset supplies.
    channel_units: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Explicit spatial coverage declaration, e.g. "77.5-80.5E 29.0-31.5N".
    coverage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Acquisition time of the data, distinct from ``ingested_at``.
    acquired_at_src: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DatasetProvenance {self.source}/{self.product} {self.data_class}>"


class AuditRecord(Base):
    """Append-only operation trail. Stores no credentials."""

    __tablename__ = "audit_records"
    __table_args__ = (
        Index("ix_audit_records_operation", "operation"),
        Index("ix_audit_records_created_at", "created_at"),
        Index("ix_audit_records_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ExecutionStatus.PENDING.value
    )
    #: Correlates an operation across logs, tasks and API responses.
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    forecast_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("forecast_runs.id", ondelete="SET NULL"), nullable=True
    )

    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    data_source: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provenance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    #: Scrubbed error message; never raw exception text containing secrets.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AuditRecord id={self.id} op={self.operation!r} status={self.status!r}>"
