"""``POST /api/v1/forecast`` - multi-hazard nowcast over the synthetic demo event."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import ErrorResponse, ForecastRequest, ForecastResponse
from app.services.inference import ModelUnavailableError, UnknownEventError

router = APIRouter(tags=["forecast"])

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Run inference and return per-lead-time hazard statistics",
    responses=_ERRORS,
)
def forecast(payload: ForecastRequest, service: ServiceDep) -> dict:
    """Run one deterministic pass (plus optional MC ensemble) and summarise it.

    Returns per-step ``max``/``mean``/``p95`` statistics per hazard, the selected
    lead time with its valid time, and the terrain-aware compound-risk summary.
    All values come from the untrained demo model unless a trained checkpoint is
    present - see ``accuracy_claim`` in the response.
    """
    try:
        return service.forecast(
            event_id=payload.event_id,
            lead_hours=payload.lead_hours,
            include_uncertainty=payload.include_uncertainty,
            mc_samples=payload.mc_samples,
        )
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


__all__ = ["router"]
