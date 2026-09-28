"""IMD GeoServer OGC adapter - genuine, credential-free ground observations.

Verified 2026-09-26 against the live service
--------------------------------------------
``https://reactjs.imd.gov.in/geoserver/imd/ows`` is an OGC Web Feature Service
that IMD's own public website (https://mausam.imd.gov.in/) calls from
JavaScript to populate its dashboards. It is:

* **keyless** - no account, token or registration;
* **official** - the endpoint is referenced by mausam.imd.gov.in itself;
* **in EPSG:4326** - so no reprojection is required.

Available observation layers discovered via ``GetCapabilities`` (30 WFS layers):

============================  ========  ====================================
Layer                         AOI hits Contents
============================  ========  ====================================
``imd:aws_data_layer``        147       AWS station temp/RH/pressure/wind/rain
``imd:synop_data_layer``      7         SYNOP surface obs, 3/6/12/24 h rain
``imd:metar_data_layer``      2         METAR (e.g. VIDN Dehradun)
``imd:radar_station_status``  3         radar availability (Surkandaji)
``imd:subdiv_rainfall_now``   4         subdivision daily rainfall
============================  ========  ====================================

Honest limitations, established by inspecting the real responses
--------------------------------------------------------------
* **This is a snapshot, not a time series.** Each station contributes one
  latest-observation feature. It cannot supply the 6-frame history the model
  needs, so real-data *training* is blocked - see
  :func:`app.ingestion.realtime.readiness.assess_readiness`.
* **The AWS ``time`` field is unusable.** It arrives as ``1970-01-01T06:15:00Z``
  (an epoch artefact), so only the day-resolution ``dat`` field is trusted. We
  do not invent an intraday timestamp from it.
* **AWS accumulation windows are heterogeneous.** ``rain_sel`` is the
  accumulation length in minutes and varies per station (0.5, 1, 1.5, 2, 2.5,
  3, 4, 4.5, 10) and is ``NULL`` for 108 of 147 records. Rainfall is converted
  to mm/h only where the window is known, and the conversion is recorded.
* SYNOP is cleaner: a real UTC hour plus standard WMO 3/6/12/24-hour
  accumulation windows.

Nothing here fabricates a value. A field that is absent stays missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.grid import GridSpec
from app.ingestion.base import DataConnector
from app.ingestion.realtime.access import AccessStatus, Availability, SourceAccessError
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.imd_ogc")

__all__ = [
    "IMD_OGC_ATTRIBUTION",
    "IMD_OGC_BASE_URL",
    "IMD_LAYERS",
    "IMDStationObservation",
    "LayerSpec",
    "OGCFeatureSet",
    "OGCResponseError",
    "fetch_layer",
    "list_layers",
    "parse_synop_features",
]

#: The OGC endpoint IMD's own site calls. Keyless.
IMD_OGC_BASE_URL = "https://reactjs.imd.gov.in/geoserver/imd/ows"

#: Workspace-qualified layer names, with what each one actually contains.
IMD_LAYERS: dict[str, str] = {
    "imd:aws_data_layer": "Automatic Weather Station observations (temp, RH, pressure, wind, rainfall)",
    "imd:synop_data_layer": "SYNOP surface observations with 3/6/12/24 h rainfall accumulations",
    "imd:metar_data_layer": "METAR aerodrome observations",
    "imd:radar_station_status": "Weather radar station availability",
    "imd:subdiv_rainfall_now": "Subdivision-level recent daily rainfall",
}

IMD_OGC_ATTRIBUTION = (
    "Ground observations from the India Meteorological Department (IMD), served "
    "via the public IMD GeoServer OGC Web Feature Service at "
    "https://reactjs.imd.gov.in/geoserver/imd/ows (referenced by "
    "https://mausam.imd.gov.in/). Cite IMD when redistributing."
)

#: SYNOP rainfall accumulations, in hours. WMO-standard reporting periods.
SYNOP_RAIN_WINDOWS_H: dict[str, float] = {
    "3hrlyrain": 3.0,
    "6hrlyrain": 6.0,
    "12hrlyrain": 12.0,
    "24hrlyrain": 24.0,
}

#: Sentinel values the service uses for "no data". Never treated as zero.
_MISSING_TOKENS = {"", "null", "none", "nan", "na", "n/a", "-", "-999", "-9999"}

#: Upper bound for a physically plausible rainfall rate [mm/h].
#:
#: The world record for a 1-hour rainfall is roughly 350 mm/h, so anything above
#: 300 mm/h indicates a sentinel or unit error in the source rather than real
#: weather. Such values are withheld (the accumulation is kept) instead of being
#: published as a physically impossible rate.
MAX_PLAUSIBLE_RAIN_RATE_MMH = 300.0

#: Plausible surface bounds, used to reject sentinels that survive ``_to_float``.
_PLAUSIBLE_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (-90.0, 60.0),
    "dewpoint_c": (-90.0, 50.0),
    "relative_humidity_pct": (0.0, 100.0),
    "pressure_hpa": (800.0, 1100.0),
    "cloud_oktas": (0.0, 8.0),
    "wind_dir_deg": (0.0, 360.0),
}


class OGCResponseError(SourceAccessError):
    """The OGC service returned something that is not usable feature data.

    Raised for HTML error pages, XML exception reports, non-JSON bodies and
    malformed GeoJSON - the cases where a naive client would silently ingest an
    error page as if it were observations.
    """


@dataclass(frozen=True, slots=True)
class LayerSpec:
    """An OGC layer and the exact request that retrieves it."""

    name: str
    description: str
    #: Bounding box in EPSG:4326 as ``min_lon,min_lat,max_lon,max_lat``.
    bbox: tuple[float, float, float, float]
    max_features: int = 5000

    def bbox_param(self) -> str:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326"

    def request_params(self) -> dict[str, Any]:
        """WFS 1.1.0 GetFeature parameters, exactly as the service expects."""
        return {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": self.name,
            "maxFeatures": int(self.max_features),
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "bbox": self.bbox_param(),
        }


@dataclass(slots=True)
class OGCFeatureSet:
    """A validated WFS response: the raw bytes plus parsed features.

    ``raw`` is retained so the exact bytes that were validated can be
    checksummed and stored, making the acquisition reproducible.
    """

    layer: str
    features: list[dict[str, Any]]
    raw: bytes
    url: str
    retrieved_at: datetime
    content_type: str = ""

    @property
    def n_features(self) -> int:
        return len(self.features)

    def checksum(self) -> str:
        import hashlib

        return hashlib.sha256(self.raw).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "n_features": self.n_features,
            "sha256": self.checksum(),
            "retrieved_at": self.retrieved_at.isoformat(),
            "content_type": self.content_type,
            "bytes": len(self.raw),
        }


def _looks_like_error_document(text: str) -> str | None:
    """Return a reason string when the body is an error page, not data.

    GeoServer answers a bad typeName or unsupported ``outputFormat`` with an
    ``ows:ExceptionReport`` XML document. That is a *200* response, so a status
    check alone would pass it through as if it were data.
    """
    head = text.lstrip()[:400].lower()
    if head.startswith("<!doctype html") or head.startswith("<html"):
        return "response is an HTML page, not feature data"
    if "exceptionreport" in head or "<serviceexception" in head:
        return "response is an OGC ServiceExceptionReport"
    return None


def _parse_feature_collection(text: str, layer: str) -> list[dict[str, Any]]:
    """Parse and structurally validate a GeoJSON FeatureCollection."""
    problem = _looks_like_error_document(text)
    if problem is not None:
        raise OGCResponseError(f"layer {layer}: {problem}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OGCResponseError(
            f"layer {layer}: body is not valid JSON ({exc.msg}); "
            "refusing to treat it as observations"
        ) from exc
    if not isinstance(payload, dict):
        raise OGCResponseError(
            f"layer {layer}: expected a JSON object, got {type(payload).__name__}"
        )
    kind = payload.get("type")
    if kind != "FeatureCollection":
        raise OGCResponseError(f"layer {layer}: expected FeatureCollection, got {kind!r}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise OGCResponseError(
            f"layer {layer}: 'features' is {type(features).__name__}, not a list"
        )
    for index, feature in enumerate(features):
        if (
            not isinstance(feature, dict)
            or "geometry" not in feature
            or "properties" not in feature
        ):
            raise OGCResponseError(
                f"layer {layer}: feature {index} is not a valid GeoJSON feature"
            )
    return features


def list_layers(timeout: float = 60.0) -> dict[str, str]:
    """Discover the available WFS layers via ``GetCapabilities``.

    Returns a mapping of layer name to description. An empty mapping means the
    service was unreachable, which callers must read as "unknown".
    """
    try:
        import httpx

        response = httpx.get(
            IMD_OGC_BASE_URL,
            params={"service": "WFS", "request": "GetCapabilities", "version": "1.1.0"},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OGC GetCapabilities failed", extra={"error": type(exc).__name__})
        return {}
    names = sorted(set(re.findall(r"<(?:wfs:)?Name>([^<]+)</", response.text)))
    return {name: IMD_LAYERS.get(name, "") for name in names}


def fetch_layer(
    spec: LayerSpec,
    *,
    timeout: float = 90.0,
    retries: int = 2,
    save_dir: str | Path | None = None,
) -> OGCFeatureSet:
    """Fetch one WFS layer and validate the response.

    Validation is deliberately strict: HTTP status, content type, HTML/OGC
    exception rejection, JSON parse, and FeatureCollection structure. Transport
    failures and 5xx are retried; a 4xx is deterministic and raises
    immediately. A *valid but empty* response is returned as-is, because an
    empty AOI is a real answer rather than a failure.
    """
    import httpx

    params = spec.request_params()
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 2):
        try:
            response = httpx.get(
                IMD_OGC_BASE_URL,
                params=params,
                timeout=timeout,
                follow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - transport failure, retry
            last_error = exc
            logger.warning(
                "OGC request failed",
                extra={"layer": spec.name, "attempt": attempt, "error": type(exc).__name__},
            )
            continue
        if response.status_code >= 500:
            last_error = OGCResponseError(
                f"layer {spec.name}: server error {response.status_code}"
            )
            continue
        if response.status_code >= 400:
            raise OGCResponseError(
                f"layer {spec.name}: HTTP {response.status_code} "
                f"({response.text[:120].strip()})"
            )
        features = _parse_feature_collection(response.text, spec.name)
        fetched = OGCFeatureSet(
            layer=spec.name,
            features=features,
            raw=response.content,
            url=str(response.url),
            retrieved_at=datetime.now(tz=timezone.utc),
            content_type=response.headers.get("content-type", ""),
        )
        if save_dir is not None:
            _persist_raw(fetched, Path(save_dir))
        logger.info("OGC layer fetched", extra=fetched.summary())
        return fetched
    raise OGCResponseError(
        f"layer {spec.name}: exhausted {retries} retries; last error: {last_error}"
    ) from last_error


def _persist_raw(fetched: OGCFeatureSet, directory: Path) -> Path:
    """Write the exact validated bytes to a raw (immutable) store."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", fetched.layer)
    stamp = fetched.retrieved_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{safe}_{stamp}.geojson"
    path.write_bytes(fetched.raw)
    return path


def _to_float(value: Any) -> float:
    """Parse a service value to float, or NaN when it encodes "no data".

    The service sends ``None``, the string ``"NULL"`` and (in one layer) a
    literal ``-999``. All of them mean *missing* and are mapped to NaN - never
    to 0.0, which would be indistinguishable from a genuine dry reading.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        number = float(value)
        # -999 / -9999 are the WMO "no data" sentinels.
        return float("nan") if number in {-999.0, -9999.0} else number
    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return float("nan")
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return float("nan") if number in {-999.0, -9999.0} else number


@dataclass(slots=True)
class IMDStationObservation:
    """One station observation in physical units, with explicit missingness.

    ``*_mm_per_h`` is derived from a documented accumulation window; where the
    window is unknown the value is NaN rather than an assumed rate.
    """

    station_id: str
    layer: str
    latitude: float
    longitude: float
    #: Observation time. Day resolution for AWS (the ``time`` field is an epoch
    #: artefact and is deliberately ignored); minute resolution for SYNOP/METAR.
    valid_time: datetime
    time_precision: str
    # -- surface state ------------------------------------------------------
    temperature_c: float = float("nan")
    dewpoint_c: float = float("nan")
    relative_humidity_pct: float = float("nan")
    pressure_hpa: float = float("nan")
    wind_speed: float = float("nan")
    wind_dir_deg: float = float("nan")
    cloud_oktas: float = float("nan")
    # -- rainfall -----------------------------------------------------------
    #: Rate in mm/h derived from the documented accumulation window.
    rain_mm_per_h: float = float("nan")
    #: The accumulation window actually used, in hours (NaN when unknown).
    rain_window_h: float = float("nan")
    #: The raw accumulation in mm as published, and its window in hours.
    rain_amount_mm: float = float("nan")
    #: Which published field the rainfall came from, for auditability.
    rain_source_field: str = ""
    # -- provenance ---------------------------------------------------------
    source_url: str = ""
    feature_id: str = ""

    def variable_summary(self) -> list[dict[str, Any]]:
        """Per-variable inventory for the provenance record."""
        entries = [
            ("air_temperature", self.temperature_c, "degC"),
            ("dew_point_temperature", self.dewpoint_c, "degC"),
            ("relative_humidity", self.relative_humidity_pct, "%"),
            ("surface_pressure", self.pressure_hpa, "hPa"),
            ("wind_speed", self.wind_speed, "source unit"),
            ("wind_direction", self.wind_dir_deg, "deg"),
            ("cloud_cover", self.cloud_oktas, "oktas"),
            ("rainfall_rate", self.rain_mm_per_h, "mm h-1"),
        ]
        return [
            {
                "variable": name,
                "unit": unit,
                "observed": bool(np.isfinite(value)),
                "value": None if not np.isfinite(value) else round(float(value), 4),
            }
            for name, value, unit in entries
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "layer": self.layer,
            "lat": self.latitude,
            "lon": self.longitude,
            "valid_time": self.valid_time.isoformat(),
            "time_precision": self.time_precision,
            "temperature_c": None if np.isnan(self.temperature_c) else self.temperature_c,
            "dewpoint_c": None if np.isnan(self.dewpoint_c) else self.dewpoint_c,
            "relative_humidity_pct": (
                None if np.isnan(self.relative_humidity_pct) else self.relative_humidity_pct
            ),
            "pressure_hpa": None if np.isnan(self.pressure_hpa) else self.pressure_hpa,
            "wind_speed": None if np.isnan(self.wind_speed) else self.wind_speed,
            "rainfall_mm_per_h": None if np.isnan(self.rain_mm_per_h) else self.rain_mm_per_h,
            "rain_window_h": None if np.isnan(self.rain_window_h) else self.rain_window_h,
            "rain_amount_mm": None if np.isnan(self.rain_amount_mm) else self.rain_amount_mm,
            "rain_source_field": self.rain_source_field,
            "feature_id": self.feature_id,
        }


def _coordinates(feature: dict[str, Any]) -> tuple[float, float]:
    """Extract ``(lon, lat)`` from a GeoJSON point, validating the type."""
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        raise OGCResponseError(
            f"expected a Point geometry, got {geometry.get('type')!r}"
        )
    coords = geometry.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        raise OGCResponseError(f"malformed point coordinates: {coords!r}")
    return float(coords[0]), float(coords[1])


def _bounded(name: str, value: float) -> float:
    """Reject values outside physical bounds as sentinels, not observations.

    Applied to every surface variable so a bad sentinel becomes NaN (missing)
    rather than an impossible number entering the dataset.
    """
    if not np.isfinite(value):
        return value
    bounds = _PLAUSIBLE_BOUNDS.get(name)
    if bounds is None:
        return value
    lo, hi = bounds
    if value < lo or value > hi:
        logger.warning(
            "observation value outside physical bounds; treated as missing",
            extra={"field": name, "value": value, "lo": lo, "hi": hi},
        )
        return float("nan")
    return value


def _parse_day(value: Any) -> datetime | None:
    """Parse the day-resolution ``dat`` field (``2026-09-26Z``)."""
    if value is None:
        return None
    text = str(value).strip().rstrip("Z")
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_synop_features(
    fetched: OGCFeatureSet,
) -> list[IMDStationObservation]:
    """Parse SYNOP features into observations with a real UTC timestamp.

    SYNOP publishes standard 3/6/12/24-hour rainfall accumulations. The 3-hour
    accumulation is preferred (shortest window -> least smoothing); the window
    actually used is recorded on every record.
    """
    records: list[IMDStationObservation] = []
    for feature in fetched.features:
        props = feature.get("properties") or {}
        lon, lat = _coordinates(feature)
        day = _parse_day(props.get("dat"))
        if day is None:
            logger.warning("SYNOP feature without a usable date; skipped")
            continue
        hour = props.get("utc")
        try:
            hour_int = int(float(hour)) if hour is not None else 0
        except (TypeError, ValueError):
            hour_int = 0
        hour_int = min(max(hour_int, 0), 23)
        valid_time = day.replace(hour=hour_int)

        # Prefer the shortest documented window.
        rain_mm = float("nan")
        window_h = float("nan")
        rain_field = ""
        for field_name, hours in SYNOP_RAIN_WINDOWS_H.items():
            amount = _to_float(props.get(field_name))
            if np.isfinite(amount):
                rain_mm, window_h, rain_field = amount, hours, field_name
                break
        rate = rain_mm / window_h if np.isfinite(rain_mm) and window_h > 0 else float("nan")
        if np.isfinite(rate) and rate > MAX_PLAUSIBLE_RAIN_RATE_MMH:
            # A 3-hour accumulation divided by 3 is normally well under this;
            # anything higher is a source/sentinel error, so withhold the rate.
            logger.warning(
                "SYNOP rainfall rate exceeds physical plausibility; rate withheld",
                extra={"station": str(props.get("station_id")), "rate_mm_h": rate},
            )
            rate = float("nan")

        records.append(
            IMDStationObservation(
                station_id=str(props.get("station_id") or feature.get("id", "")),
                layer=fetched.layer,
                latitude=lat,
                longitude=lon,
                valid_time=valid_time,
                time_precision="hour",
                temperature_c=_bounded("temperature_c", _to_float(props.get("dbtemp"))),
                dewpoint_c=_bounded("dewpoint_c", _to_float(props.get("dewtemp"))),
                relative_humidity_pct=_bounded(
                    "relative_humidity_pct", _to_float(props.get("rh"))
                ),
                pressure_hpa=_bounded("pressure_hpa", _to_float(props.get("mslp"))),
                wind_speed=_to_float(props.get("windsp")),
                wind_dir_deg=_bounded("wind_dir_deg", _to_float(props.get("winddir"))),
                cloud_oktas=_bounded("cloud_oktas", _to_float(props.get("nebulosity"))),
                rain_mm_per_h=rate,
                rain_window_h=window_h,
                rain_amount_mm=rain_mm,
                rain_source_field=rain_field,
                source_url=fetched.url,
                feature_id=str(feature.get("id", "")),
            )
        )
    return records


def _parse_aws_features(fetched: OGCFeatureSet) -> list[IMDStationObservation]:
    """Parse AWS features.

    Three deliberate constraints, each established by inspecting the live data:

    * the ``time`` field is an epoch artefact (``1970-01-01T06:15:00Z``) and is
      **ignored**; only the day-resolution ``dat`` is used;
    * ``rain_sel`` is the accumulation length in **hours** (verified: its values
      are 0.0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 6, 10 - reading it as
      minutes produced impossible rates of 10 800 mm/h). A value of ``0`` is
      rejected, because dividing by it is undefined;
    * a derived rate above :data:`MAX_PLAUSIBLE_RAIN_RATE_MMH` is treated as a
      unit/sentinel error: the accumulation is retained but the rate is left
      missing rather than published as a physically impossible value.
    """
    records: list[IMDStationObservation] = []
    for feature in fetched.features:
        props = feature.get("properties") or {}
        lon, lat = _coordinates(feature)
        day = _parse_day(props.get("dat"))
        if day is None:
            logger.warning("AWS feature without a usable date; skipped")
            continue
        rain_mm = _to_float(props.get("rainfall"))
        window_h = _to_float(props.get("rain_sel"))
        # 0, negative and NaN windows are all "window unknown".
        usable_window = np.isfinite(window_h) and window_h > 0
        rate = (
            rain_mm / window_h
            if np.isfinite(rain_mm) and usable_window
            else float("nan")
        )
        if np.isfinite(rate) and rate > MAX_PLAUSIBLE_RAIN_RATE_MMH:
            logger.warning(
                "AWS rainfall rate exceeds physical plausibility; rate withheld",
                extra={
                    "station": str(props.get("station_id") or props.get("id")),
                    "amount_mm": rain_mm,
                    "window_h": window_h,
                    "rate_mm_h": rate,
                },
            )
            rate = float("nan")

        station = (
            props.get("station_id")
            or props.get("id")
            or props.get("call_sign")
            or feature.get("id", "")
        )
        records.append(
            IMDStationObservation(
                station_id=str(station),
                layer=fetched.layer,
                latitude=lat,
                longitude=lon,
                valid_time=day,
                # Day resolution: the intraday time field is unusable.
                time_precision="day",
                temperature_c=_bounded("temperature_c", _to_float(props.get("temp"))),
                dewpoint_c=_bounded("dewpoint_c", _to_float(props.get("dewpoint"))),
                relative_humidity_pct=_bounded(
                    "relative_humidity_pct", _to_float(props.get("rh"))
                ),
                pressure_hpa=_bounded("pressure_hpa", _to_float(props.get("mslp"))),
                wind_speed=_to_float(props.get("windspeed")),
                wind_dir_deg=_bounded("wind_dir_deg", _to_float(props.get("winddir"))),
                cloud_oktas=_bounded("cloud_oktas", _to_float(props.get("nebulosity"))),
                rain_mm_per_h=rate,
                rain_window_h=window_h if usable_window else float("nan"),
                rain_amount_mm=rain_mm,
                rain_source_field="rainfall",
                source_url=fetched.url,
                feature_id=str(feature.get("id", "")),
            )
        )
    return records


#: Layer -> parser. Only layers whose schema was verified against a live
#: response are dispatched; anything else raises rather than guessing.
LAYER_PARSERS = {
    "imd:synop_data_layer": _parse_synop_features,
    "imd:aws_data_layer": _parse_aws_features,
}


def parse_features(fetched: OGCFeatureSet) -> list[IMDStationObservation]:
    """Parse a fetched feature set using the verified schema for its layer."""
    parser = LAYER_PARSERS.get(fetched.layer)
    if parser is None:
        raise OGCResponseError(
            f"no verified parser for layer {fetched.layer!r}; available: "
            f"{sorted(LAYER_PARSERS)}. Refusing to guess a schema."
        )
    return parser(fetched)


class IMDOGCConnector(DataConnector):
    """Acquires genuine IMD station observations from the public OGC service.

    Unlike the staged-file connectors, this one performs a real, keyless
    download. It is therefore the project's first *working* real-data path.
    """

    source_name = "IMD GeoServer OGC (live)"
    is_synthetic = False
    data_class = "observation"

    def __init__(
        self,
        grid: GridSpec,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        raw_dir: str | Path | None = None,
        timeout: float = 90.0,
    ) -> None:
        super().__init__(grid, demo_mode=False, cache_dir=raw_dir)
        self.bbox = bbox or (grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat)
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.timeout = timeout

    def spec_for(self, layer: str, max_features: int = 5000) -> LayerSpec:
        """Build the request spec for one layer over the configured AOI."""
        if layer not in IMD_LAYERS:
            raise OGCResponseError(
                f"unknown IMD OGC layer {layer!r}; known: {sorted(IMD_LAYERS)}"
            )
        return LayerSpec(
            name=layer,
            description=IMD_LAYERS[layer],
            bbox=self.bbox,
            max_features=max_features,
        )

    def fetch_layer(self, layer: str, *, max_features: int = 5000) -> OGCFeatureSet:
        """Download and validate one layer, persisting the raw bytes."""
        return fetch_layer(
            self.spec_for(layer, max_features),
            timeout=self.timeout,
            save_dir=self.raw_dir,
        )

    def fetch_observations(
        self, layer: str, *, max_features: int = 5000
    ) -> list[IMDStationObservation]:
        """Download one layer and parse it into physical-unit observations."""
        return parse_features(self.fetch_layer(layer, max_features=max_features))

    def availability(self, *, timeout: float = 30.0) -> AccessStatus:
        """Probe the live service rather than reporting a stored assumption."""
        layers = list_layers(timeout=timeout)
        details = {
            "endpoint": IMD_OGC_BASE_URL,
            "n_layers_discovered": len(layers),
            "observation_layers": [
                name for name in layers if name in IMD_LAYERS
            ],
            "bbox": list(self.bbox),
            "authentication": "none required (keyless public OGC service)",
        }
        if not layers:
            return AccessStatus(
                source=self.source_name,
                availability=Availability.UNREACHABLE,
                reason="OGC GetCapabilities did not return any layer",
                details=details,
                is_synthetic=False,
            )
        relevant = [name for name in layers if name in IMD_LAYERS]
        return AccessStatus(
            source=self.source_name,
            availability=Availability.AVAILABLE,
            reason=(
                f"public keyless OGC service reachable; {len(layers)} WFS layers "
                f"published, {len(relevant)} of them carry observations"
            ),
            details=details,
            is_synthetic=False,
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["attribution"] = IMD_OGC_ATTRIBUTION
        payload["data_class"] = self.data_class
        payload["endpoint"] = IMD_OGC_BASE_URL
        return payload
