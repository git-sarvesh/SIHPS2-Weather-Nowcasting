"""``GET /api/v1/health`` - application and component health."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep
from app.api.v1.schemas import ErrorResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application and component health",
    responses={500: {"model": ErrorResponse}},
)
def health(service: ServiceDep) -> dict:
    """Report component readiness.

    Never raises: an unavailable model is reported as ``status="degraded"`` with
    the reason in ``error`` so that orchestrators can distinguish "starting up"
    from "broken". The database probe and the task schedule are reported the same
    way and never turn a healthy API into a failed one.
    """
    payload = service.health()
    payload["database"] = _database_health()
    payload["schedule"] = _schedule_health()
    payload["connectors"] = _connector_health()
    payload["data_sources"] = _data_source_health()
    return payload


def _data_source_health() -> dict:
    """Real-data access state plus the year-split feasibility verdict.

    Reports the shortfall in historical coverage rather than quietly presenting
    the synthetic demo as if it satisfied the required 2015-2025 evaluation
    window.
    """
    try:
        from app.ingestion.realtime.sources import describe_real_sources

        return describe_real_sources()
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"any_real_available": False, "error": type(exc).__name__}


def _database_health() -> dict:
    """Database connectivity, or a graceful ``unavailable`` report."""
    try:
        from app.db.migrations import current_revision
        from app.db.session import get_session_factory

        factory = get_session_factory()
        report = factory.health()
        if report["status"] == "ok":
            report["current_revision"] = current_revision(factory.engine)
        return report
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"status": "unavailable", "error": type(exc).__name__}


def _schedule_health() -> dict:
    """Celery schedule description (no broker contact)."""
    try:
        from app.tasks.celery_app import describe_schedule

        return describe_schedule()
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"schedule_enabled": False, "error": type(exc).__name__}


def _connector_health() -> dict:
    """Which data sources can actually be ingested."""
    try:
        from app.tasks.connectors import describe_connectors

        return describe_connectors()
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return {"any_live_available": False, "error": type(exc).__name__}


__all__ = ["router"]
