"""``/api/v1/risk/*`` - point risk and GeoJSON risk polygons."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.deps import ServiceDep, bad_request, not_found, unavailable
from app.api.v1.schemas import (
    ErrorResponse,
    PointRiskRequest,
    PointRiskResponse,
    RiskGeoJSONRequest,
    RiskGeoJSONResponse,
)
from app.services.inference import ModelUnavailableError, UnknownEventError

router = APIRouter(prefix="/risk", tags=["risk"])

_ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


@router.post(
    "/point",
    response_model=PointRiskResponse,
    summary="Terrain-aware compound risk at a single coordinate",
    responses=_ERRORS,
)
def point_risk(payload: PointRiskRequest, service: ServiceDep) -> dict:
    """Return the per-hazard breakdown, exposure and risk category at ``(lat, lon)``.

    A coordinate outside the configured AOI yields ``400`` (the point is well
    formed but not coverable), while an unknown ``event_id`` yields ``404``.
    """
    try:
        return service.point_risk(
            lat=payload.lat,
            lon=payload.lon,
            event_id=payload.event_id,
            lead_hours=payload.lead_hours,
        )
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


@router.post(
    "/geojson",
    response_model=RiskGeoJSONResponse,
    summary="Risk polygons as a GeoJSON FeatureCollection",
    responses=_ERRORS,
)
def risk_geojson(payload: RiskGeoJSONRequest, service: ServiceDep) -> dict:
    """Vectorise the categorised risk raster into merged rectangle polygons.

    ``min_category`` filters out low-risk cells (0 = LOW .. 3 = EXTREME), so a
    dashboard can request only the actionable areas.
    """
    try:
        return service.risk_geojson(
            event_id=payload.event_id,
            lead_hours=payload.lead_hours,
            min_category=payload.min_category,
            risk_field=payload.risk_field,
            max_features=payload.max_features,
        )
    except UnknownEventError as exc:
        raise not_found(exc) from exc
    except ModelUnavailableError as exc:
        raise unavailable(exc) from exc
    except ValueError as exc:
        raise bad_request(exc) from exc


__all__ = ["router"]
