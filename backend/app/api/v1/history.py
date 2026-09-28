"""``/api/v1/forecast/*`` persistence: explicit persist, paginated history, detail.

These are additive routes. The six Phase 1 endpoints are unchanged, so existing
clients keep working; a caller that wants an auditable record asks for it here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import (
    ErrorResponse,
    ForecastDetailResponse,
    ForecastHistoryResponse,
    ForecastRunSummary,
    PersistForecastRequest,
    PersistForecastResponse,
    RiskCellOut,
)
from app.db.models import ExecutionStatus, RunKind
from app.db.repository import get_forecast_run, get_run_risk_cells, list_forecast_runs
from app.db.session import DatabaseUnavailableError, get_session
from app.logging_conf import get_logger
from app.services.inference import ModelUnavailableError, UnknownEventError

logger = get_logger("api.history")

router = APIRouter(prefix="/forecast", tags=["forecast-history"])

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.api.v1.deps import ServiceDep  # noqa: F401 - re-exported for convenience
from app.db.session import get_session

#: Transactional database session (commits on return, rolls back on error).
SessionDep = Annotated[Session, Depends(get_session)]

__all__ = ["ServiceDep", "SessionDep", "get_session"]

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _database_error(exc: DatabaseUnavailableError) -> HTTPException:
    """A persistence failure must never look like a successful forecast."""
    logger.error("database unavailable", extra={"error": str(exc)})
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"database unavailable: {exc}",
    )


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


def _summary(run: Any, risk_cells: int = 0) -> dict[str, Any]:
    return {
        "id": run.id,
        "kind": run.kind,
        "status": run.status,
        "init_time": _iso(run.init_time),
        "valid_time": _iso(run.valid_time),
        "lead_hours": run.lead_hours,
        "n_lead_times": run.n_lead_times,
        "model_version": run.model_version,
        "model_backend": run.model_backend,
        "trained_checkpoint_loaded": run.trained_checkpoint_loaded,
        "is_synthetic": run.is_synthetic,
        "data_source": run.data_source,
        "risk_cells": risk_cells,
    }


@router.post(
    "/persist",
    response_model=PersistForecastResponse,
    summary="Run inference and persist the run (and optionally its risk raster)",
    responses=_ERRORS,
)
def persist(payload: PersistForecastRequest, service: ServiceDep) -> dict[str, Any]:
    """Execute inference and store an auditable forecast run.

    Idempotent: the key is derived from (kind, event, model version, issue time,
    lead), so repeating the same request refreshes the same record rather than
    creating a duplicate.
    """
    from app.tasks.jobs import persist_forecast

    try:
        result = persist_forecast(
            payload.event_id,
            payload.lead_hours,
            service=service,
            kind=RunKind.API.value,
            persist_risk=payload.persist_risk and payload.max_cells > 0,
            max_cells=payload.max_cells,
        )
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except DatabaseUnavailableError as exc:
        raise _database_error(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc

    notice = (
        "SYNTHETIC DEMONSTRATION - persisted from synthetic inputs with an untrained model. "
        "Not an observationally validated forecast and not an official IMD warning."
        if result.get("is_synthetic")
        else "Persisted operational forecast. Not an official IMD warning."
    )
    return {**result, "notice": notice}


@router.get(
    "/history",
    response_model=ForecastHistoryResponse,
    summary="Paginated forecast-run history, newest first",
    responses=_ERRORS,
)
def history(
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    run_status: str | None = Query(default=None, alias="status"),
    is_synthetic: bool | None = Query(default=None),
    model_version: str | None = Query(default=None),
) -> dict[str, Any]:
    """A page of persisted runs with total/offset/has_more pagination metadata.

    ``is_synthetic`` is filterable so a dashboard can separate a synthetic
    demonstration from an operational run.
    """
    from sqlalchemy import func, select

    from app.db.models import RiskCell

    try:
        runs, total = list_forecast_runs(
            session,
            limit=limit,
            offset=offset,
            status=run_status,
            is_synthetic=is_synthetic,
            model_version=model_version,
        )
        counts = dict(
            session.execute(
                select(RiskCell.forecast_run_id, func.count())
                .where(RiskCell.forecast_run_id.in_([r.id for r in runs] or [-1]))
                .group_by(RiskCell.forecast_run_id)
            ).all()
        )
    except DatabaseUnavailableError as exc:
        raise _database_error(exc) from exc

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(runs),
        "has_more": offset + len(runs) < total,
        "runs": [_summary(run, int(counts.get(run.id, 0))) for run in runs],
        "filters": {
            "status": run_status,
            "is_synthetic": is_synthetic,
            "model_version": model_version,
        },
    }


@router.get(
    "/{run_id}",
    response_model=ForecastDetailResponse,
    summary="One persisted run with provenance and (optionally) risk cells",
    responses=_ERRORS,
)
def detail(
    run_id: int,
    session: SessionDep,
    include_risk: bool = Query(default=True),
    risk_limit: int = Query(default=100, ge=0, le=5000),
    min_risk: float = Query(default=0.0, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Detail for a single run.

    ``include_risk=false`` returns metadata only, which is what a list view or a
    large AOI should use.
    """
    try:
        run = get_forecast_run(session, run_id)
        if run is None:
            raise not_found(ValueError(f"forecast run {run_id} not found"))
        cells = (
            get_run_risk_cells(session, run_id, limit=risk_limit, min_risk=min_risk)
            if include_risk and risk_limit > 0
            else []
        )
        total_cells = len(run.risk_cells)
    except DatabaseUnavailableError as exc:
        raise _database_error(exc) from exc

    return {
        "run": _summary(run, total_cells),
        "created_at": _iso(run.created_at),
        "updated_at": _iso(run.updated_at),
        "duration_seconds": run.duration_seconds,
        "checkpoint": run.checkpoint,
        "checkpoint_sha256": run.checkpoint_sha256,
        "grid": run.grid,
        "provenance": run.provenance,
        "risk_summary": run.summary,
        "error": run.error,
        "risk_cell_count": total_cells,
        "risk_cells": [
            RiskCellOut(
                id=c.id,
                row=c.row,
                col=c.col,
                lat=c.lat,
                lon=c.lon,
                thunderstorm=c.thunderstorm,
                cloudburst=c.cloudburst,
                flood_probability=c.flood_probability,
                flood_risk=c.flood_risk,
                compound_storm_cloudburst=c.compound_storm_cloudburst,
                terrain_exposure=c.terrain_exposure,
                overall_risk=c.overall_risk,
                risk_category=c.risk_category,
                uncertainty=c.uncertainty,
            ).model_dump()
            for c in cells
        ],
    }


@router.get(
    "/{run_id}/status",
    summary="Execution status of one run (mirrors the Celery task view)",
    responses=_ERRORS,
)
def run_status(run_id: int, session: SessionDep) -> dict[str, Any]:
    """Lightweight status lookup for polling a run."""
    try:
        run = get_forecast_run(session, run_id)
    except DatabaseUnavailableError as exc:
        raise _database_error(exc) from exc
    if run is None:
        raise not_found(ValueError(f"forecast run {run_id} not found"))
    return {
        "id": run.id,
        "status": run.status,
        "is_terminal": run.status in {
            ExecutionStatus.SUCCEEDED.value,
            ExecutionStatus.FAILED.value,
            ExecutionStatus.SKIPPED.value,
            ExecutionStatus.DISABLED.value,
        },
        "is_synthetic": run.is_synthetic,
        "model_version": run.model_version,
        "error": run.error,
    }
