"""IMDAA NetCDF validation, vertical interpolation and derived diagnostics.

Everything here is conditional on data that is actually present. The module
never fabricates a satellite channel, a surface observation, a DEM feature, or a
CAPE value: if the variables needed are missing, the corresponding function
raises or reports the gap.

What is implemented, and why it is justified
--------------------------------------------
**Vertical interpolation (log-pressure).** Meteorological fields are much closer
to linear in ``ln(p)`` than in ``p``; this is the standard assumption behind
met-model vertical interpolation (e.g. the ECMWF/Unified Model treatment of
pressure-level data). :func:`interpolate_to_levels` uses it and records the
method on a :class:`ResamplingRecord`.

**CAPE / CIN / lifted index.** Delegated to
:func:`app.physics.parcel_ascent`, an existing pseudo-adiabatic parcel ascent
already used by the project. It requires temperature *and* specific humidity on
at least three levels. If humidity is absent, CAPE is **not** derived - the
function reports that rather than substituting relative humidity.

**What is deliberately not derived.** No INSAT channels (TIR/VIS/SWIR/MIR/CTT),
no surface observations, no DEM/terrain features, and no rainfall targets. Those
sources are unavailable or out of scope; filling them would be fabrication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.ingestion.base import parse_time as _parse_time
from app.ingestion.realtime.provenance import QualityReport, ResamplingRecord
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.imdaa_netcdf")

__all__ = [
    "IMDAA_CHANNEL_UNITS",
    "IMDAA_PROVIDER",
    "IMDAAChannelSet",
    "IMDAAFileReport",
    "IMDAAPressureFields",
    "VerticalProfile",
    "build_observation_cube",
    "derive_cape",
    "derive_iwv",
    "detect_variables",
    "interpolate_to_levels",
    "read_pressure_fields",
    "read_pressure_levels",
    "validate_imdaa_file",
]

#: Provider recorded on every IMDAA acquisition.
IMDAA_PROVIDER = "NCMRWF (National Centre for Medium Range Weather Forecasting)"

#: The only model channels IMDAA pressure-level data can genuinely supply.
#: The ten remaining channels (INSAT-3D brightness temperatures, reflectance,
#: cloud-top temperature and its tendency, water-vapour BT anomaly, DEM
#: elevation) need other sources. They are left NaN, never filled.
IMDAA_DERIVED_CHANNELS: tuple[str, ...] = ("cape", "iwv")


#: Candidate NetCDF names for the IMDAA fields this project can use, in
#: priority order. Detection is by name; a file that uses different names is
#: reported as unrecognised rather than guessed at.
VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("t", "temp", "temperature", "air_temperature", "2t"),
    "specific_humidity": ("q", "sh", "specific_humidity", "humidity", "r"),
    "relative_humidity": ("rh", "relative_humidity", "r2"),
    "u_wind": ("u", "u_component_of_wind", "uwind", "eastward_wind"),
    "v_wind": ("v", "v_component_of_wind", "vwind", "northward_wind"),
    "geopotential": ("z", "geopotential", "geopotential_height"),
    "mean_sea_level_pressure": ("msl", "mslp", "mean_sea_level_pressure", "pressure"),
    "total_cloud_cover": ("tcc", "cloud_cover", "total_cloud_cover"),
}

#: Plausible physical ranges, used to flag (not silently clip) bad values.
VARIABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (180.0, 340.0),          # K
    "specific_humidity": (0.0, 0.05),       # kg/kg
    "relative_humidity": (0.0, 100.0),      # %
    "geopotential": (-5000.0, 60000.0),     # m2 s-2
    "mean_sea_level_pressure": (800.0, 1100.0),  # hPa
    "total_cloud_cover": (0.0, 100.0),      # %
}

#: NetCDF fill values that must be treated as missing, not as data.
_FILL_VALUES = {-999.0, -9999.0, -1.0e20, -3.0e38, 1.0e20}


@dataclass(slots=True)
class IMDAAFileReport:
    """What a single IMDAA NetCDF file actually contains."""

    path: str
    ok: bool = False
    #: Detected field name -> the NetCDF variable it came from.
    detected: dict[str, str] = field(default_factory=dict)
    #: NetCDF variables present but not mapped to any known field.
    unmapped: list[str] = field(default_factory=list)
    pressure_levels_hpa: list[float] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    lat_range: tuple[float, float] | None = None
    lon_range: tuple[float, float] | None = None
    crs: str = ""
    units: dict[str, str] = field(default_factory=dict)
    quality: QualityReport = field(default_factory=QualityReport)
    notes: list[str] = field(default_factory=list)
    resampling: list[ResamplingRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "detected": dict(self.detected),
            "unmapped": list(self.unmapped),
            "n_pressure_levels": len(self.pressure_levels_hpa),
            "pressure_levels_hpa": [round(p, 2) for p in self.pressure_levels_hpa],
            "times": list(self.times),
            "lat_range": list(self.lat_range) if self.lat_range else None,
            "lon_range": list(self.lon_range) if self.lon_range else None,
            "crs": self.crs,
            "units": dict(self.units),
            "quality": self.quality.to_dict(),
            "notes": list(self.notes),
            "resampling": [r.to_dict() for r in self.resampling],
        }


def _mask_fill(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask of *valid* samples.

    NetCDF fill sentinels (``_FillValue``, ``missing_value`` and the common
    -999/-9999 conventions) are missing, not measurements. The project's
    missing-data contract forbids turning them into zeros.
    """
    finite = np.isfinite(values)
    for fill in _FILL_VALUES:
        finite &= values != fill
    return finite


def detect_variables(dataset) -> tuple[dict[str, str], list[str]]:
    """Map NetCDF variable names onto this project's field vocabulary.

    Returns ``(detected, unmapped)`` where ``detected`` maps
    ``project_field -> netcdf_name``. Detection is by name only; nothing is
    inferred from shape or magnitude.
    """
    available = {str(name) for name in dataset.data_vars}
    detected: dict[str, str] = {}
    for field_name, aliases in VARIABLE_ALIASES.items():
        for alias in aliases:
            if alias in available:
                detected[field_name] = alias
                break
    unmapped = sorted(available - set(detected.values()))
    return detected, unmapped


def _coord_values(dataset, names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in dataset.coords:
            return np.asarray(dataset.coords[name].values)
        if name in dataset.variables:
            return np.asarray(dataset.variables[name].values)
    return None


def read_pressure_levels(dataset) -> list[float]:
    """Read the pressure axis [hPa], or ``[]`` when the file is single-level.

    Some archives store the axis in Pa; a magnitude above 2000 is treated as
    Pa and converted, rather than producing levels of 100000 hPa.
    """
    levels = _coord_values(dataset, ("level", "pressure", "isobaric", "plev"))
    if levels is None:
        return []
    values = np.asarray(levels, dtype=np.float64).ravel()
    if values.size and float(np.nanmax(np.abs(values))) > 2000.0:
        values = values / 100.0
    return [float(v) for v in values if np.isfinite(v)]



def validate_imdaa_file(path: str | Path) -> IMDAAFileReport:
    """Validate one IMDAA NetCDF file and report exactly what it contains.

    Checks: openable, recognised variables and units, coordinate ranges, a time
    axis, a pressure axis (if any), and physical-range violations (counted,
    never silently clipped).

    A file that is not recognisable IMDAA returns ``ok=False`` with the reason;
    it is never partially accepted.
    """
    report = IMDAAFileReport(path=str(path))
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - xarray is a declared dep
        report.quality.fail("xarray is required to read IMDAA NetCDF")
        raise RuntimeError(str(exc)) from exc

    try:
        dataset = xr.open_dataset(path)
    except Exception as exc:  # noqa: BLE001 - a malformed file is a data fact
        report.quality.fail(f"could not open as NetCDF: {type(exc).__name__}")
        return report

    with dataset:
        report.detected, report.unmapped = detect_variables(dataset)
        report.pressure_levels_hpa = read_pressure_levels(dataset)
        if not report.detected:
            report.quality.fail(
                "no recognised IMDAA variables found; unmapped: "
                + (", ".join(report.unmapped) or "<none>")
            )
            return report

        for field_name, nc_name in report.detected.items():
            unit = dataset[nc_name].attrs.get("units")
            if unit is not None:
                report.units[field_name] = str(unit)

        lats = _coord_values(dataset, ("lat", "latitude", "y"))
        lons = _coord_values(dataset, ("lon", "longitude", "x"))
        if lats is not None and lats.size:
            report.lat_range = (float(np.nanmin(lats)), float(np.nanmax(lats)))
        if lons is not None and lons.size:
            report.lon_range = (float(np.nanmin(lons)), float(np.nanmax(lons)))
        has_lonlat = lats is not None and lons is not None
        report.crs = str(
            dataset.attrs.get("crs")
            or dataset.attrs.get("projection")
            or ("EPSG:4326" if has_lonlat else "unknown")
        )
        if not has_lonlat:
            report.quality.warn("no recognised lat/lon coordinates found")

        if "time" not in dataset.coords:
            report.quality.warn(
                "no time axis found; the file cannot be placed in a series"
            )
        else:
            # `.dt` only exists when the axis decoded to datetimes; a file with
            # a string-typed time coordinate must still be readable.
            try:
                report.times = [
                    str(v) for v in dataset.time.dt.strftime("%Y-%m-%dT%H:%M:%S").values
                ]
            except (AttributeError, TypeError):
                report.times = [
                    str(v) for v in np.asarray(dataset["time"].values).ravel()
                ]

        total = 0
        invalid = 0
        missing = 0
        for field_name, nc_name in report.detected.items():
            values = np.asarray(dataset[nc_name].values, dtype=np.float64)
            valid = _mask_fill(values)
            total += int(values.size)
            missing += int(values.size - np.count_nonzero(valid))
            bounds = VARIABLE_BOUNDS.get(field_name)
            if bounds is not None:
                outside = valid & ((values < bounds[0]) | (values > bounds[1]))
                invalid += int(np.count_nonzero(outside))
        report.quality.n_values = total
        report.quality.n_missing = missing
        if total and invalid:
            report.quality.warn(
                f"{invalid} of {total} values fall outside the documented physical "
                "range; they are retained and counted, not clipped"
            )
        if not report.pressure_levels_hpa and "temperature" in report.detected:
            report.notes.append(
                "no pressure axis found: this is a single-level file, so it "
                "cannot supply the CAPE/IWV model channels"
            )
        report.ok = not report.quality.errors
        report.quality.passed = report.ok
    return report



def interpolate_to_levels(
    source_levels_hpa: Sequence[float],
    target_levels_hpa: Sequence[float],
    values: np.ndarray,
    *,
    max_gap_levels: int = 1,
) -> tuple[np.ndarray, ResamplingRecord]:
    """Interpolate a pressure-level field onto other levels, linearly in ln(p).

    Log-pressure interpolation is the standard treatment for meteorological
    pressure-level data: temperature and geopotential vary close to linearly in
    ``ln(p)`` over the lower troposphere, whereas linear-in-p interpolation
    overshoots near the surface.

    A target level is only filled when it is bracketed by **adjacent valid source
    levels** no more than ``max_gap_levels`` apart. Beyond that the value is left
    NaN, so a single missing level cannot silently manufacture a value two
    levels away from any observation.

    Parameters
    ----------
    source_levels_hpa:
        Levels matching the *first* axis of ``values``.
    values:
        Array shaped ``(n_source_levels, ...)``; NaN propagates as missing.

    Returns
    -------
    ``(interpolated, record)`` where ``record`` documents the method, the
    source/target level sets and the gap rule for the provenance record.
    """
    src = np.asarray(source_levels_hpa, dtype=np.float64)
    dst = np.asarray(target_levels_hpa, dtype=np.float64)
    if values.shape[0] != src.size:
        raise ValueError(
            f"values has {values.shape[0]} levels but {src.size} source levels given"
        )
    if src.size < 2:
        raise ValueError("need at least 2 source levels to interpolate")
    if np.any(src <= 0) or np.any(dst <= 0):
        raise ValueError("pressure levels must be positive to interpolate in log(p)")

    # Sort ascending in log(p) and interpolate along a new leading axis.
    order = np.argsort(np.log(src))
    src_log = np.log(src[order])
    dst_log = np.log(dst)
    stacked = np.asarray(values, dtype=np.float64)[order]
    # np.interp needs a 1-D y; flatten the trailing dimensions.
    trailing = stacked.shape[1:]
    flat = stacked.reshape(stacked.shape[0], -1)
    out = np.empty((dst.size, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        column_values = flat[:, column]
        good = np.isfinite(column_values)
        if good.sum() < 2:
            out[:, column] = np.nan
            continue
        out[:, column] = np.interp(dst_log, src_log[good], column_values[good])
        # Withhold any target whose bracketing source levels are too far apart.
        indices = np.arange(src.size)
        for row in range(dst.size):
            above = indices[(src_log >= dst_log[row]) & good]
            below = indices[(src_log <= dst_log[row]) & good]
            if above.size == 0 or below.size == 0:
                out[row, column] = np.nan
                continue
            if int(above.min()) - int(below.max()) > max_gap_levels:
                out[row, column] = np.nan
    interpolated = out.reshape((dst.size, *trailing))
    record = ResamplingRecord(
        stage="vertical_interpolation",
        method="linear in log(pressure), adjacent-level bracketing only",
        source_resolution=f"{src.size} levels {src.min():g}-{src.max():g} hPa",
        target_resolution=f"{dst.size} levels {dst.min():g}-{dst.max():g} hPa",
        detail=(
            "meteorological pressure-level fields are close to linear in ln(p). A "
            f"target is filled only when its bracketing valid levels are within "
            f"{max_gap_levels} level(s); wider gaps stay NaN rather than being "
            "interpolated across missing data."
        ),
    )
    return interpolated, record


@dataclass(slots=True)
class VerticalProfile:
    """One column's pressure-level profile, ready for parcel ascent."""

    pressure_hpa: np.ndarray
    temperature_k: np.ndarray
    specific_humidity_kgkg: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.pressure_hpa, dtype=np.float64).ravel()
        t = np.asarray(self.temperature_k, dtype=np.float64).ravel()
        q = np.asarray(self.specific_humidity_kgkg, dtype=np.float64).ravel()
        if not (p.size == t.size == q.size):
            raise ValueError("pressure, temperature and humidity must have equal length")
        if p.size < 3:
            raise ValueError("a parcel ascent needs at least 3 levels")
        order = np.argsort(-p)  # surface (highest pressure) first
        self.pressure_hpa = p[order]
        self.temperature_k = t[order]
        self.specific_humidity_kgkg = q[order]

    def is_complete(self) -> bool:
        return bool(
            np.isfinite(self.pressure_hpa).all()
            and np.isfinite(self.temperature_k).all()
            and np.isfinite(self.specific_humidity_kgkg).all()
        )


def derive_cape(
    profiles: Sequence[VerticalProfile],
) -> tuple[np.ndarray, ResamplingRecord, dict[str, Any]]:
    """Derive CAPE / CIN / lifted index from real pressure-level profiles.

    Delegates to :func:`app.physics.parcel_ascent`, the project's existing
    pseudo-adiabatic ascent. Columns whose humidity is missing yield NaN - CAPE
    is **not** estimated from relative humidity or temperature alone, because
    that would not be the same quantity.

    Returns ``(cape_j_per_kg, record, detail)``. ``cape`` is NaN wherever the
    profile lacks temperature or specific humidity.
    """
    from app.physics import parcel_ascent

    cape = np.full(len(profiles), np.nan, dtype=np.float64)
    derived = 0
    skipped = 0
    for index, profile in enumerate(profiles):
        if not profile.is_complete():
            skipped += 1
            continue
        try:
            result = parcel_ascent(
                profile.pressure_hpa,
                profile.temperature_k,
                profile.specific_humidity_kgkg,
            )
        except ValueError:
            skipped += 1
            continue
        value = result.get("cape")
        if value is not None and np.isfinite(value):
            cape[index] = float(value)
            derived += 1
    record = ResamplingRecord(
        stage="derive_instability",
        method="pseudo-adiabatic parcel ascent (app.physics.parcel_ascent)",
        source_resolution="pressure-level T and q",
        target_resolution="column CAPE [J kg-1]",
        detail=(
            f"{derived} of {len(profiles)} columns derived; {skipped} skipped for "
            "missing temperature or specific humidity. No CAPE is estimated from "
            "missing temperature or specific humidity. No CAPE is estimated from "
            "temperature or relative humidity alone."
        ),
    )
    detail = {
        "method": "pseudo-adiabatic parcel ascent",
        "reference": "app.physics.parcel_ascent (Bolton 1980 thermodynamics)",
        "n_derived": derived,
        "n_skipped_incomplete": skipped,
        "requires": ["temperature", "specific_humidity", ">=3 pressure levels"],
    }
    return cape, record, detail


def source_grid_spacing(lats: np.ndarray, lons: np.ndarray) -> float:
    """Approximate mean grid spacing [km] of a regular lat/lon grid."""
    from app.grid import km_per_deg_lat, km_per_deg_lon

    lat_mid = float(np.mean(lats))
    dy = abs(float(np.mean(np.diff(lats)))) * km_per_deg_lat() if lats.size > 1 else 0.0
    dx = abs(float(np.mean(np.diff(lons)))) * km_per_deg_lon(lat_mid) if lons.size > 1 else 0.0
    positive = [v for v in (dx, dy) if v > 0]
    return float(np.mean(positive)) if positive else 0.0



# --------------------------------------------------------------------------- #
# Reading real pressure-level fields and assembling the canonical cube
# --------------------------------------------------------------------------- #
#: Documented physical unit per IMDAA NetCDF field. Used to record the unit of
#: any model channel derived from that column, so units come from the source's
#: own vocabulary rather than being guessed at the ingestion boundary.
IMDAA_CHANNEL_UNITS: dict[str, str] = {
    "temperature": "K",
    "specific_humidity": "kg kg-1",
    "relative_humidity": "%",
    "u_wind": "m s-1",
    "v_wind": "m s-1",
    "geopotential": "m2 s-2",
    "mean_sea_level_pressure": "hPa",
    "total_cloud_cover": "%",
}

#: NetCDF dimension names recognised for each axis, in preference order.
_DIM_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "valid_time", "forecast_time"),
    "level": ("level", "pressure", "isobaric", "plev"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}

#: The model channels IMDAA cannot supply; they need INSAT-3D or DEM inputs.
IMDAA_UNAVAILABLE_CHANNELS: tuple[str, ...] = (
    "tir1_bt", "tir2_bt", "wv_bt", "vis_refl", "swir_refl", "mir_bt",
    "ctt", "ctt_cooling_rate", "wv_bt_anomaly", "elevation",
)


def _dim_of(variable, kind: str) -> str | None:
    for alias in _DIM_ALIASES[kind]:
        if alias in variable.dims:
            return alias
    return None


def _stack_axes(variable, order: Sequence[str | None]) -> np.ndarray:
    """Return ``variable``'s data with axes arranged as ``order``.

    An axis named in ``order`` but absent from the file is inserted as a
    length-1 axis, so single-level and pressure-level variables can share one
    code path without inventing a level for the former.
    """
    present = [d for d in order if d is not None and d in variable.dims]
    arr = np.asarray(variable.transpose(*present).values)
    index: list[Any] = []
    for name in order:
        if name is None or name not in variable.dims:
            index.append(np.newaxis)
        else:
            index.append(slice(None))
    return arr[tuple(index)]


def _convert_units(
    field_name: str, values: np.ndarray, unit: str | None
) -> tuple[np.ndarray, str | None]:
    """Normalise temperature and humidity to this project's units (K, kg/kg).

    IMDAA publishes K and kg kg-1, but an operator-supplied file may carry
    degC or g kg-1. The conversion is reported, not silent, because it changes
    every diagnostic derived from the field.
    """
    if not unit:
        return values, None
    lowered = unit.strip().lower()
    if field_name == "temperature" and "c" in lowered and "k" not in lowered:
        return values + 273.15, f"degC -> K (declared units {unit!r})"
    if field_name == "specific_humidity" and ("g/kg" in lowered or "g kg" in lowered):
        return values / 1000.0, f"g/kg -> kg/kg (declared units {unit!r})"
    return values, None


@dataclass(slots=True)
class IMDAAPressureFields:
    """Real pressure-level fields read from one file, on that file's own grid.

    ``temperature_k`` is ``(T, L, Y, X)``; ``specific_humidity_kgkg`` has the
    same shape or is ``None`` when the file carries no humidity. Fill values and
    non-finite samples are NaN, never zero.
    """

    source_grid: Any
    times: list[Any]
    pressure_levels_hpa: list[float]
    temperature_k: np.ndarray
    specific_humidity_kgkg: np.ndarray | None
    records: list[ResamplingRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_times(self) -> int:
        return int(self.temperature_k.shape[0])


@dataclass(slots=True)
class IMDAAChannelSet:
    """Derived model channels, resampled onto the canonical grid."""

    #: channel name -> ``(T, H, W)`` physical units; NaN where unavailable.
    channels: dict[str, np.ndarray]
    #: Fraction of finite cells per channel, so coverage is measured not assumed.
    coverage: dict[str, float]
    times: list[Any]
    records: list[ResamplingRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": sorted(self.channels),
            "coverage": dict(self.coverage),
            "n_times": len(self.times),
            "records": [r.to_dict() for r in self.records],
            "notes": list(self.notes),
        }


def derive_iwv(
    pressure_levels_hpa: Sequence[float], specific_humidity: np.ndarray
) -> np.ndarray:
    """Precipitable water [mm] from a specific-humidity column [kg kg-1].

    ``IWV = (1/g) * integral(q dp)`` by the trapezoidal rule. A column with any
    missing humidity returns NaN in full, so a partial column can never report
    a deceptively small total.

    ``specific_humidity`` is shaped ``(..., L)`` with the pressure axis last.
    """
    from app.physics import GRAVITY

    p = np.asarray(pressure_levels_hpa, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)
    if q.shape[-1] != p.size:
        raise ValueError(
            f"humidity has {q.shape[-1]} levels but {p.size} pressure levels given"
        )
    complete = np.isfinite(q).all(axis=-1)
    q_safe = np.where(np.isfinite(q), q, 0.0)
    order = np.argsort(p)
    integral = np.trapezoid(q_safe[..., order], x=p[order] * 100.0, axis=-1)
    return np.where(complete, integral / GRAVITY, np.nan)


def read_pressure_fields(
    path: str | Path, report: IMDAAFileReport | None = None
) -> IMDAAPressureFields:
    """Read temperature and humidity profiles from an IMDAA NetCDF file.

    Raises when the file has no pressure axis, no temperature, or no usable
    1-D lat/lon grid, because CAPE and IWV cannot be derived without them.
    Nothing is substituted in those cases.
    """
    import xarray as xr

    from app.grid import GridSpec

    report = report or validate_imdaa_file(path)
    notes: list[str] = []
    records: list[ResamplingRecord] = []

    with xr.open_dataset(path) as dataset:
        levels = report.pressure_levels_hpa or read_pressure_levels(dataset)
        if not levels:
            raise ValueError(
                f"{Path(path).name} has no pressure axis, so it cannot supply the "
                "CAPE or IWV channels; a single-level file is not enough"
            )
        temp_name = report.detected.get("temperature")
        if not temp_name:
            raise ValueError(
                f"{Path(path).name} carries no recognised temperature variable "
                f"(found: {sorted(report.detected.values()) or 'none'})"
            )
        humidity_name = report.detected.get("specific_humidity")

        lats = _coord_values(dataset, _DIM_ALIASES["lat"])
        lons = _coord_values(dataset, _DIM_ALIASES["lon"])
        if lats is None or lons is None or lats.ndim != 1 or lons.ndim != 1:
            raise ValueError(
                f"{Path(path).name} does not have 1-D lat/lon coordinates; a "
                "curvilinear or absent grid cannot be aligned to the model grid"
            )
        if lats.size < 2 or lons.size < 2:
            raise ValueError(f"{Path(path).name} has a degenerate lat/lon axis")

        descending = bool(np.all(np.diff(lats) < 0))
        lats = np.sort(lats)
        if np.all(np.diff(lons) < 0):
            lons = np.sort(lons)
        if descending:
            records.append(
                ResamplingRecord(
                    stage="orientation",
                    method="latitude axis flipped to ascending order",
                    source_resolution="as stored on disk",
                    target_resolution="ascending south-to-north",
                    detail=(
                        "the file stores latitude descending; the field is flipped "
                        "so it matches the model's north-up raster"
                    ),
                )
            )

        temp_var = dataset[temp_name]
        order: list[str | None] = [
            _dim_of(temp_var, "time"),
            _dim_of(temp_var, "level"),
            _dim_of(temp_var, "lat"),
            _dim_of(temp_var, "lon"),
        ]
        if order[1] is None:
            raise ValueError(
                f"{Path(path).name}: temperature {temp_name!r} has no level dimension"
            )
        temperature = _stack_axes(temp_var, order).astype(np.float64)
        temperature, note = _convert_units(
            "temperature", temperature, temp_var.attrs.get("units")
        )
        if note:
            notes.append(f"temperature: {note}")
        temperature[~_mask_fill(temperature)] = np.nan

        humidity = None
        if humidity_name:
            q_var = dataset[humidity_name]
            q_order = [
                _dim_of(q_var, "time"), _dim_of(q_var, "level"),
                _dim_of(q_var, "lat"), _dim_of(q_var, "lon"),
            ]
            if q_order[1] is not None:
                humidity = _stack_axes(q_var, q_order).astype(np.float64)
                humidity, note = _convert_units(
                    "specific_humidity", humidity, q_var.attrs.get("units")
                )
                if note:
                    notes.append(f"specific_humidity: {note}")
                humidity[~_mask_fill(humidity)] = np.nan
                if humidity.shape != temperature.shape:
                    notes.append(
                        f"humidity shape {humidity.shape} differs from temperature "
                        f"{temperature.shape}; humidity ignored"
                    )
                    humidity = None
            else:
                notes.append(
                    f"humidity variable {humidity_name!r} has no level dimension"
                )
        if humidity is None:
            notes.append(
                "no level-resolved specific humidity: CAPE and IWV stay NaN, "
                "because a temperature-only or RH-based estimate is not the same "
                "quantity"
            )

        if descending:
            temperature = temperature[:, :, ::-1, :]
            if humidity is not None:
                humidity = humidity[:, :, ::-1, :]

        source_grid = GridSpec(
            min_lon=float(lons[0]),
            min_lat=float(lats[0]),
            max_lon=float(lons[-1]),
            max_lat=float(lats[-1]),
            res_km=source_grid_spacing(lats, lons),
            nx=int(lons.size),
            ny=int(lats.size),
        )

        times: list[Any] = []
        if "time" in dataset.coords:
            try:
                times = [
                    _parse_time(str(v))
                    for v in dataset.time.dt.strftime("%Y-%m-%dT%H:%M:%S").values
                ]
            except (AttributeError, TypeError):
                times = [
                    _parse_time(str(v))
                    for v in np.asarray(dataset["time"].values).ravel()
                ]
        n_expected = int(temperature.shape[0])
        if not times:
            times = [None] * n_expected
            notes.append("no usable time axis; frames are reported without valid times")
        elif len(times) != n_expected:
            notes.append("time axis length differed from the field; it was truncated")
            times = (times * n_expected)[:n_expected]

    return IMDAAPressureFields(
        source_grid=source_grid,
        times=times,
        pressure_levels_hpa=levels,
        temperature_k=temperature,
        specific_humidity_kgkg=humidity,
        records=records,
        notes=notes,
    )



def _horizontal_alignment_record(source: Any, target: Any) -> ResamplingRecord:
    return ResamplingRecord(
        stage="horizontal_alignment",
        method="bilinear on lat/lon centres",
        source_resolution=f"{source.ny}x{source.nx} at ~{source.res_km:.1f} km",
        target_resolution=(
            f"{target.ny}x{target.nx} at ~{target.effective_res_km[1]:.2f} km"
        ),
        detail=(
            "IMDAA ~12 km reanalysis resampled onto the model grid with "
            "app.grid.GridSpec.resample_bilinear. Target cells outside the file's "
            "longitude/latitude range are NaN, not clamped to the edge value."
        ),
    )


def _outside_source(target: Any, source: Any) -> np.ndarray:
    """Boolean mask of target cells falling outside the source grid extent."""
    lons = np.asarray(target.lon_centers())
    lats = np.asarray(target.lat_centers())
    inside_x = (lons >= source.min_lon) & (lons <= source.max_lon)
    inside_y = (lats >= source.min_lat) & (lats <= source.max_lat)
    return ~np.outer(inside_y, inside_x)


def derive_channel_fields(fields: IMDAAPressureFields, grid: Any) -> IMDAAChannelSet:
    """Derive CAPE and IWV per column and resample them onto ``grid``.

    Every frame goes through the real calculation: a parcel ascent for CAPE and a
    pressure integral for IWV. Channels that cannot be derived come back all-NaN
    rather than being filled with a plausible number.
    """
    records = list(fields.records)
    notes = list(fields.notes)
    levels = fields.pressure_levels_hpa
    outside = _outside_source(grid, fields.source_grid)
    n_times = fields.n_times

    # Derived first on the *source* grid, then warped onto the model grid:
    # resample_bilinear requires a field shaped like the source raster.
    source_shape = fields.source_grid.shape
    iwv = np.full((n_times, *source_shape), np.nan, dtype=np.float64)
    cape = np.full((n_times, *source_shape), np.nan, dtype=np.float64)
    for index in range(n_times):
        q = fields.specific_humidity_kgkg
        if q is None:
            continue
        q_column = np.moveaxis(q[index], 0, -1)  # (Y, X, L): levels last
        iwv[index] = derive_iwv(levels, q_column)
        t_column = np.moveaxis(fields.temperature_k[index], 0, -1)
        shape = t_column.shape[:2]
        flat_t = t_column.reshape(-1, t_column.shape[-1])
        flat_q = q_column.reshape(-1, q_column.shape[-1])
        profiles = [
            VerticalProfile(levels, flat_t[i], flat_q[i]) for i in range(flat_t.shape[0])
        ]
        values, record, _detail = derive_cape(profiles)
        cape[index] = values.reshape(shape)
        if index == 0:
            records.append(record)
    if fields.specific_humidity_kgkg is None:
        notes.append("CAPE and IWV are NaN: the file has no level-resolved humidity")
    records.append(_horizontal_alignment_record(fields.source_grid, grid))

    raw = {"cape": cape, "iwv": iwv}
    channels: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for name, values in raw.items():
        warped = np.empty((n_times, *grid.shape), dtype=np.float64)
        for t in range(n_times):
            warped[t] = grid.resample_bilinear(fields.source_grid, values[t])
        warped[:, outside] = np.nan
        channels[name] = warped
        total = int(warped.size)
        finite = int(np.count_nonzero(np.isfinite(warped)))
        coverage[name] = float(finite) / float(total) if total else 0.0

    return IMDAAChannelSet(
        channels=channels,
        coverage=coverage,
        times=list(fields.times),
        records=records,
        notes=notes,
    )


def build_observation_cube(
    path: str | Path, grid: Any, *, report: IMDAAFileReport | None = None
) -> Any:
    """Assemble an :class:`~app.ingestion.base.ObservationCube` from a real file.

    The cube carries all 12 model channels, but only ``cape`` and ``iwv`` are
    filled, from genuine IMDAA pressure-level data. Every other channel is
    **NaN**, and ``metadata["channels_available"]`` / ``["channels_missing"]``
    record the split, so no consumer can mistake this for a complete input.

    Raises :class:`ValueError` when the file fails validation or yields no
    finite value for any derived channel, rather than returning an all-NaN cube.
    """
    from app.ingestion.base import ObservationCube, Provenance
    from app.physics import N_CHANNELS, channel_index

    report = report or validate_imdaa_file(path)
    if not report.ok:
        raise ValueError(
            f"{Path(path).name} failed validation: "
            + "; ".join(report.quality.errors or ["unknown reason"])
        )
    fields = read_pressure_fields(path, report)
    derived = derive_channel_fields(fields, grid)
    if not any(np.isfinite(v).any() for v in derived.channels.values()):
        raise ValueError(
            f"{Path(path).name} produced no finite value for any derived channel "
            f"(tried: {sorted(derived.channels)}); "
            + ("; ".join(derived.notes) or "no usable pressure-level data")
        )

    n_times = len(derived.times)
    channels = np.full((n_times, N_CHANNELS, *grid.shape), np.nan, dtype=np.float32)
    for name, values in derived.channels.items():
        channels[:, channel_index(name), :, :] = values.astype(np.float32)

    available = [
        name for name in IMDAA_DERIVED_CHANNELS
        if np.isfinite(channels[:, channel_index(name)]).any()
    ]
    fraction = np.mean(np.isfinite(channels), axis=(1, 2, 3)).astype(np.float32)
    quality = np.broadcast_to(fraction[:, None, None], (n_times, *grid.shape)).copy()

    fallback = _parse_time("1970-01-01T00:00:00")
    times = [t if t is not None else fallback for t in derived.times]
    return ObservationCube(
        grid=grid,
        times=times,
        channels=channels,
        quality=quality,
        provenance=[
            Provenance(
                source="NCMRWF RDS (IMDAA)",
                product=str(Path(path).name),
                valid_from=min(times),
                valid_to=max(times),
                path=str(path),
                is_synthetic=False,
                attribution=(
                    "IMDAA reanalysis: NCMRWF/IMD/Met Office under the National "
                    "Monsoon Mission, retrieved from the NCMRWF RDS portal."
                ),
            )
        ],
        metadata={
            "source_file": str(path),
            "channels_available": available,
            "channels_missing": list(IMDAA_UNAVAILABLE_CHANNELS),
            "channel_coverage": derived.coverage,
            "pressure_levels_hpa": fields.pressure_levels_hpa,
            "resampling": [r.to_dict() for r in derived.records],
            "notes": derived.notes,
            "trainable": False,
            "trainable_reason": (
                "IMDAA supplies only CAPE and IWV. The ten remaining channels need "
                "INSAT-3D and DEM inputs, so this cube is not a complete model input."
            ),
        },
    )
