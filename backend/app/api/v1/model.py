"""``GET /api/v1/model/describe`` - model configuration and metadata."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import ErrorResponse, ModelDescribeResponse
from app.services.inference import ModelUnavailableError, UnknownEventError

router = APIRouter(prefix="/model", tags=["model"])

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.get(
    "/describe",
    response_model=ModelDescribeResponse,
    summary="Model configuration, input shapes and provenance",
    responses=_ERRORS,
)
def describe_model(service: ServiceDep) -> dict:
    """Return the resolved model configuration and demo metadata.

    The payload states explicitly whether a *trained* checkpoint was loaded; in
    demo mode with fresh weights this is ``false``.
    """
    try:
        return service.describe_model()
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


__all__ = ["router"]
