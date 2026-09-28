"""``GET /api/v1/model/checkpoint`` - active checkpoint identity and provenance."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import CheckpointInfoResponse, ErrorResponse
from app.services.inference import ModelUnavailableError, UnknownEventError

router = APIRouter(prefix="/model", tags=["model"])

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.get(
    "/checkpoint",
    response_model=CheckpointInfoResponse,
    summary="Active checkpoint version, training provenance and calibration status",
    responses=_ERRORS,
)
def checkpoint_info(service: ServiceDep) -> dict:
    """Describe the checkpoint currently backing the model.

    The response always states ``observational_validation: false`` and carries the
    training run's own ``validation_status``; a synthetic training run is never
    relabelled as independent validation.
    """
    try:
        return service.active_checkpoint_info()
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


__all__ = ["router"]
