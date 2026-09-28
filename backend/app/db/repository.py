"""Repository helpers: the only place that writes forecast/audit rows.

Every write goes through here so the honesty rules are enforced in one spot:

* ``is_synthetic`` is always written explicitly, never defaulted to ``False``.
* Error messages are scrubbed of credential-looking text before being stored.
* Idempotent inserts return the *existing* row instead of raising, which is what
  makes a re-fired Celery task harmless.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import (
    AuditRecord,
    ExecutionStatus,
    ForecastRun,
    RiskCell,
    RiskCategory,
    RunKind,
)
from app.logging_conf import get_logger

logger = get_logger("db.repository")

__all__ = [
    "build_idempotency_key",
    "create_audit_record",
    "get_forecast_run",
    "get_run_risk_cells",
    "list_dataset_provenance",
    "list_forecast_runs",
    "record_dataset_provenance",
    "record_forecast_run",
    "replace_risk_cells",
    "scrub_detail",
    "scrub_error",
]

#: Patterns that must never reach the database or the logs.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),  # credentials embedded in a URL
)


def scrub_error(message: str | None, *, limit: int = 500) -> str | None:
    """Remove credential-looking fragments from an error message."""
    if not message:
        return message
    cleaned = message
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned[:limit]


#: Combined pattern, used by :func:`scrub_detail` to test a free-text value.
_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]\s*\S+"
)


def record_dataset_provenance(
    session: Session, provenance: "Any"
) -> tuple["Any", bool]:
    """Persist a :class:`~app.ingestion.realtime.provenance.DatasetProvenance`.

    Returns ``(row, created)``. When the same ``(source, product, acquired_at,
    checksum)`` has already been ingested the existing row is returned with
    ``created=False`` - duplicate ingestion is prevented by the unique
    ``ingest_key`` constraint, not by a check-then-act race.

    Detail payloads are scrubbed before storage so a credential can never reach
    the provenance table.
    """
    from app.db.models import DatasetProvenanceRow

    key = provenance.key
    existing = session.execute(
        select(DatasetProvenanceRow).where(DatasetProvenanceRow.ingest_key == key)
    ).scalar_one_or_none()
    if existing is not None:
        logger.info(
            "duplicate ingestion ignored",
            extra={"source": provenance.source, "product": provenance.product},
        )
        return existing, False

    detail = {
        "variables": list(provenance.variables),
        "quality": provenance.quality.to_dict(),
        "resampling": [record.to_dict() for record in provenance.resampling],
        "processing_config": dict(provenance.processing_config),
        "processing_version": provenance.processing_version,
        "pipeline": provenance.pipeline,
        "channels_present": provenance.processing_config.get("channels_present", []),
        "channels_absent": provenance.processing_config.get("channels_absent", []),
    }
    row = DatasetProvenanceRow(
        ingest_key=key,
        source=provenance.source,
        product=provenance.product,
        data_class=provenance.data_class,
        acquired_at=provenance.acquired_at,
        ingested_at=provenance.ingested_at,
        valid_from=provenance.valid_from,
        valid_to=provenance.valid_to,
        path=provenance.path,
        checksum=provenance.checksum,
        min_lon=provenance.min_lon,
        min_lat=provenance.min_lat,
        max_lon=provenance.max_lon,
        max_lat=provenance.max_lat,
        crs=provenance.crs,
        source_resolution=provenance.source_resolution,
        processed_resolution=provenance.processed_resolution,
        source_cadence=provenance.source_cadence,
        processed_cadence=provenance.processed_cadence,
        n_frames=provenance.n_frames,
        detail=scrub_detail(detail),
        attribution=provenance.attribution,
        license_note=provenance.license_note,
    )
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        # A concurrent ingest won the race; return that row instead of failing.
        session.rollback()
        existing = session.execute(
            select(DatasetProvenanceRow).where(DatasetProvenanceRow.ingest_key == key)
        ).scalar_one_or_none()
        if existing is None:  # pragma: no cover - would indicate a real fault
            raise
        return existing, False
    logger.info(
        "dataset provenance recorded",
        extra={
            "source": provenance.source,
            "product": provenance.product,
            "data_class": provenance.data_class,
        },
    )
    return row, True


#: Substrings that mark a mapping key as credential-bearing.
_SECRET_KEY_HINTS = (
    "password",
    "passwd",
    "token",
    "secret",
    "cookie",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


def scrub_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Drop credential-bearing keys from a JSON payload, recursively.

    Operates on the *parsed* structure rather than on a JSON string, so the
    result is always valid JSON. A previous implementation scrubbed a
    serialised string, which corrupted it (``"note": "password=x"`` became
    unparseable) - masking must never break the record it protects.
    """
    if isinstance(detail, dict):
        return {
            key: ("<redacted>" if _is_secret_key(key) else scrub_detail(value))
            for key, value in detail.items()
        }
    if isinstance(detail, (list, tuple)):
        return [scrub_detail(item) for item in detail]
    if isinstance(detail, str) and _SECRET_RE.search(detail):
        return scrub_error(detail) or "<redacted>"
    return detail


def _is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def list_dataset_provenance(
    session: Session,
    *,
    source: str | None = None,
    data_class: str | None = None,
    limit: int = 100,
) -> list["Any"]:
    """Provenance rows, newest first, optionally filtered."""
    from app.db.models import DatasetProvenanceRow

    stmt = select(DatasetProvenanceRow).order_by(DatasetProvenanceRow.acquired_at.desc())
    if source:
        stmt = stmt.where(DatasetProvenanceRow.source == source)
    if data_class:
        stmt = stmt.where(DatasetProvenanceRow.data_class == data_class)
    return list(session.execute(stmt.limit(max(1, int(limit)))).scalars())


def build_idempotency_key(
    *,
    model_version: str,
    init_time: datetime,
    lead_hours: float,
    event_id: str | None = None,
    kind: str = RunKind.API.value,
) -> str:
    """Deterministic key for one logical forecast request.

    Two calls with the same (kind, event, model, issue time, lead) produce the
    same key, so a retried task updates the existing run instead of duplicating it.
    """
    raw = "|".join(
        [kind, event_id or "-", model_version, init_time.isoformat(), f"{float(lead_hours):.4f}"]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


# --------------------------------------------------------------------------- #
# forecast runs
# --------------------------------------------------------------------------- #
def record_forecast_run(
    session: Session,
    *,
    idempotency_key: str,
    model_version: str,
    init_time: datetime,
    lead_hours: float,
    lead_times_h: Sequence[float],
    status: str = ExecutionStatus.SUCCEEDED.value,
    kind: str = RunKind.API.value,
    valid_time: datetime | None = None,
    is_synthetic: bool = True,
    model_backend: str | None = None,
    checkpoint: str | None = None,
    checkpoint_sha256: str | None = None,
    trained_checkpoint_loaded: bool = False,
    data_source: str | None = None,
    grid: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> ForecastRun:
    """Insert or update one forecast run, keyed on ``idempotency_key``.

    Returns the stored row. An existing row with the same key has its status and
    metadata refreshed, so a retry converges instead of duplicating.
    """
    existing = session.scalar(
        select(ForecastRun).where(ForecastRun.idempotency_key == idempotency_key)
    )
    if existing is not None:
        existing.status = status
        existing.error = scrub_error(error)
        existing.duration_seconds = duration_seconds
        existing.summary = summary
        existing.valid_time = valid_time
        existing.trained_checkpoint_loaded = trained_checkpoint_loaded
        if checkpoint is not None:
            existing.checkpoint = checkpoint
        if checkpoint_sha256 is not None:
            existing.checkpoint_sha256 = checkpoint_sha256
        if provenance is not None:
            existing.provenance = provenance
        session.add(existing)
        session.flush()
        return existing

    run = ForecastRun(
        idempotency_key=idempotency_key,
        kind=kind,
        status=status,
        error=scrub_error(error),
        init_time=init_time,
        valid_time=valid_time,
        lead_hours=float(lead_hours),
        n_lead_times=len(lead_times_h),
        duration_seconds=duration_seconds,
        model_version=model_version,
        model_backend=model_backend,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        trained_checkpoint_loaded=bool(trained_checkpoint_loaded),
        is_synthetic=bool(is_synthetic),
        data_source=data_source,
        grid=grid,
        provenance=provenance,
        summary=summary,
    )
    session.add(run)
    try:
        session.flush()
    except IntegrityError:
        # A concurrent writer inserted the same key; adopt the winner's row.
        session.rollback()
        winner = session.scalar(
            select(ForecastRun).where(ForecastRun.idempotency_key == idempotency_key)
        )
        if winner is None:  # pragma: no cover - only if the key vanished
            raise
        return winner
    return run


def list_forecast_runs(
    session: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    is_synthetic: bool | None = None,
    model_version: str | None = None,
) -> tuple[list[ForecastRun], int]:
    """Newest-first page of runs plus the total matching the filters."""
    filters = []
    if status is not None:
        filters.append(ForecastRun.status == status)
    if is_synthetic is not None:
        filters.append(ForecastRun.is_synthetic.is_(bool(is_synthetic)))
    if model_version is not None:
        filters.append(ForecastRun.model_version == model_version)

    total = session.scalar(select(func.count()).select_from(ForecastRun).where(*filters)) or 0
    rows = list(
        session.scalars(
            select(ForecastRun)
            .where(*filters)
            .order_by(ForecastRun.init_time.desc(), ForecastRun.id.desc())
            .limit(max(1, int(limit)))
            .offset(max(0, int(offset)))
        )
    )
    return rows, int(total)


def get_forecast_run(session: Session, run_id: int) -> ForecastRun | None:
    """One run by primary key, or ``None``."""
    return session.get(ForecastRun, int(run_id))


def get_run_risk_cells(
    session: Session, run_id: int, *, limit: int = 500, min_risk: float = 0.0
) -> list[RiskCell]:
    """Risk cells of a run, highest risk first."""
    return list(
        session.scalars(
            select(RiskCell)
            .where(
                RiskCell.forecast_run_id == int(run_id),
                RiskCell.overall_risk >= float(min_risk),
            )
            .order_by(RiskCell.overall_risk.desc())
            .limit(max(1, int(limit)))
        )
    )


# --------------------------------------------------------------------------- #
# risk cells
# --------------------------------------------------------------------------- #
def replace_risk_cells(
    session: Session,
    run: ForecastRun,
    *,
    thunderstorm: Any,
    cloudburst: Any,
    flood_risk: Any,
    flood_probability: Any,
    exposure: Any,
    compound: Any,
    overall: Any,
    grid: Any,
    thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
    uncertainty: dict[str, Any] | None = None,
    max_cells: int = 20000,
) -> int:
    """Replace the run's risk cells with a fresh raster.

    *Replace* rather than append, so re-running a task rewrites the same cells
    instead of duplicating them. The delete is issued as a statement (and
    flushed) before any insert, so the per-run unique index on (row, col) is
    free within the same transaction.
    """
    from sqlalchemy import delete

    from app.services.risk_engine import categorise

    session.execute(delete(RiskCell).where(RiskCell.forecast_run_id == run.id))
    session.flush()
    session.expire(run, ["risk_cells"])

    rows, cols = grid.shape
    written = 0
    for row in range(rows):
        for col in range(cols):
            if written >= max_cells:
                break
            lat, lon = grid.cell_center(row, col)
            category_code = max(0, min(3, int(categorise(float(overall[row, col]), thresholds))))
            session.add(
                RiskCell(
                    forecast_run_id=run.id,
                    row=row,
                    col=col,
                    lat=round(float(lat), 6),
                    lon=round(float(lon), 6),
                    thunderstorm=round(float(thunderstorm[row, col]), 6),
                    cloudburst=round(float(cloudburst[row, col]), 6),
                    flood_probability=round(float(flood_probability[row, col]), 6),
                    flood_risk=round(float(flood_risk[row, col]), 6),
                    compound_storm_cloudburst=round(float(compound[row, col]), 6),
                    terrain_exposure=round(float(exposure[row, col]), 6),
                    overall_risk=round(float(overall[row, col]), 6),
                    risk_category_code=category_code,
                    risk_category=RiskCategory.from_code(category_code),
                    uncertainty=(
                        {k: round(float(v[row, col]), 6) for k, v in uncertainty.items()}
                        if uncertainty
                        else None
                    ),
                )
            )
            written += 1
    session.flush()
    return written


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def create_audit_record(
    session: Session,
    *,
    operation: str,
    status: str = ExecutionStatus.PENDING.value,
    request_id: str | None = None,
    actor: str | None = None,
    forecast_run_id: int | None = None,
    is_synthetic: bool = True,
    model_version: str | None = None,
    data_source: str | None = None,
    provenance: dict[str, Any] | None = None,
    error: str | None = None,
    duration_seconds: float | None = None,
    details: dict[str, Any] | None = None,
) -> AuditRecord:
    """Append one audit row with a scrubbed error message."""
    record = AuditRecord(
        operation=operation,
        status=status,
        request_id=request_id,
        actor=actor,
        forecast_run_id=forecast_run_id,
        is_synthetic=bool(is_synthetic),
        model_version=model_version,
        data_source=data_source,
        provenance=provenance,
        error=scrub_error(error),
        duration_seconds=duration_seconds,
        details=details,
    )
    session.add(record)
    session.flush()
    return record


def list_audit_records(
    session: Session, *, limit: int = 20, offset: int = 0, operation: str | None = None
) -> tuple[list[AuditRecord], int]:
    """Newest-first page of audit rows plus the matching total."""
    filters = [AuditRecord.operation == operation] if operation else []
    total = session.scalar(select(func.count()).select_from(AuditRecord).where(*filters)) or 0
    rows = list(
        session.scalars(
            select(AuditRecord)
            .where(*filters)
            .order_by(AuditRecord.created_at.desc(), AuditRecord.id.desc())
            .limit(max(1, int(limit)))
            .offset(max(0, int(offset)))
        )
    )
    return rows, int(total)


def check_database(session: Session) -> bool:
    """``True`` when a trivial query succeeds; ``False`` otherwise."""
    try:
        session.execute(select(func.count()).select_from(ForecastRun))
        return True
    except SQLAlchemyError as exc:  # pragma: no cover - defensive
        logger.warning("database probe failed", extra={"error": type(exc).__name__})
        return False
