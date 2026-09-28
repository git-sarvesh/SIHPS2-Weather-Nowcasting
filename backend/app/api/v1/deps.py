"""Shared FastAPI dependencies and error translation for the v1 routers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.services.inference import (
    InferenceService,
    ModelUnavailableError,
    UnknownEventError,
    get_inference_service,
)

ServiceDep = Annotated[InferenceService, Depends(get_inference_service)]


def unavailable(exc: ModelUnavailableError) -> HTTPException:
    """503 for "no usable predictor" situations (e.g. non-demo mode, no checkpoint)."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


def not_found(exc: LookupError) -> HTTPException:
    """404 for unknown synthetic event ids."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


def bad_request(exc: ValueError) -> HTTPException:
    """400 for semantically invalid but schema-valid input (e.g. bad lead time)."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


__all__ = [
    "ModelUnavailableError",
    "ServiceDep",
    "UnknownEventError",
    "bad_request",
    "not_found",
    "unavailable",
]

