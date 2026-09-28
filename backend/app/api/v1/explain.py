"""``POST /api/v1/explain`` - Grad-CAM attribution and physical-consistency audit."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import ErrorResponse, ExplainRequest, ExplainResponse
from app.services.inference import ModelUnavailableError, UnknownEventError

router = APIRouter(tags=["explainability"])

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Spatial attribution, channel importance and consistency audit",
    responses=_ERRORS,
)
def explain(payload: ExplainRequest, service: ServiceDep) -> dict:
    """Explain one hazard decision of the demo model.

    Returns the Grad-CAM style saliency summary, per-input-channel attributions,
    the physical-consistency audit of the prediction, and (optionally) a
    counterfactual "what-if" shift under perturbed channels. With untrained
    weights the attributions describe the model, not real atmospheric behaviour.
    """
    try:
        return service.explain(
            hazard=payload.hazard,
            step=payload.step,
            event_id=payload.event_id,
            include_consistency=payload.include_consistency,
            include_what_if=payload.include_what_if,
            perturbations=payload.perturbations,
        )
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


__all__ = ["router"]
