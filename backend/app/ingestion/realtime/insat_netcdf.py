"""INSAT-3D / INSAT-3DR imagery to model channels (Phase 8.5).

The six INSAT imager channels map one-to-one onto the model's satellite
channels, and three further model channels are *derived* from them using exactly
the derivations :data:`app.physics.CHANNEL_META` already declares:

===================  ===========================================  ==========
Model channel        Source                                       Kind
===================  ===========================================  ==========
``tir1_bt``          10.8 um brightness temperature               observed
``tir2_bt``          12.0 um brightness temperature               observed
``wv_bt``            6.8 um brightness temperature                observed
``vis_refl``         0.65 um reflectance                          observed
``swir_refl``        1.6 um reflectance                           observed
``mir_bt``           3.9 um brightness temperature                observed
``ctt``              ``tir1_bt`` (window brightness temperature)  derived
``ctt_cooling_rate`` finite difference of ``ctt`` over frames    derived
``wv_bt_anomaly``    ``wv_bt`` minus the field median             derived
===================  ===========================================  ==========

Honesty rules enforced here
---------------------------
* Nothing is synthesised. A file that cannot be opened, carries no recognised
  channel variable, or declares no 1-D lat/lon grid fails validation and yields
  no channel values at all.
* A variable declared in **radiance** units is refused. Converting radiance to
  brightness temperature needs the published per-channel calibration
  coefficients ``(c1, c2)``, and this project does not hardcode them: using the
  placeholder defaults would emit a numerically wrong temperature that looks
  real. Such a granule must be re-encoded by the operator, or the coefficients
  supplied explicitly.
* Values outside the physical range declared in
  :data:`app.physics.CHANNEL_META` are masked to NaN and counted. They are never
  clipped into range, because a clipped value is an invented value.
* A channel that is unavailable stays NaN for the whole series. Nothing is
  borrowed from a neighbouring channel or frame.

Documented, not verified
------------------------
The variable-name aliases below are the *documented* MOSDAC spellings. They have
**not** been confirmed against a downloaded granule, so an unrecognised layout
fails closed and reports the variables that were actually found instead of
guessing a mapping. The same applies to geolocation: full-disc L1B swath
products carry curvilinear 2-D latitude/longitude arrays, whose resampling is
explicitly **not implemented** here and refuses rather than approximating.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import FRAME_MINUTES, GridSpec
from app.ingestion.base import parse_time, source_file_from_disk
from app.ingestion.realtime.imdaa_netcdf import (
    _mask_fill,
    _stack_axes,
    source_grid_spacing,
)
from app.ingestion.realtime.provenance import QualityReport, ResamplingRecord, file_sha256
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.insat_netcdf")

__all__ = [
    "INSAT_ATTRIBUTION",
    "INSAT_DERIVED_CHANNELS",
    "INSAT_OBSERVED_CHANNELS",
    "INSAT_PROVIDER",
    "INSAT_VARIABLE_ALIASES",
    "INSATChannelSet",
    "INSATFileReport",
    "build_insat_cube",
    "derive_ctt",
    "derive_ctt_cooling_rate",
    "derive_wv_bt_anomaly",
    "read_insat_channels",
    "validate_insat_file",
]

#: Provider recorded on every INSAT acquisition.
INSAT_PROVIDER = "ISRO / MOSDAC (INSAT-3D & INSAT-3DR)"

#: Attribution text reproduced on every INSAT-derived artefact.
INSAT_ATTRIBUTION = (
    "INSAT-3D/INSAT-3DR imagery: Indian Remote Sensing Satellite, operated by "
    "ISRO and archived by MOSDAC (Space Applications Centre). Cite MOSDAC and "
    "the product/dataset identifier used."
)

#: The six directly measured INSAT channels, in model-channel order.
INSAT_OBSERVED_CHANNELS: tuple[str, ...] = (
    "tir1_bt",
    "tir2_bt",
    "wv_bt",
    "vis_refl",
    "swir_refl",
    "mir_bt",
)

#: Model channels derived from the six observed ones (see ``CHANNEL_META``).
INSAT_DERIVED_CHANNELS: tuple[str, ...] = (
    "ctt",
    "ctt_cooling_rate",
    "wv_bt_anomaly",
)

#: Model channel -> accepted dataset variable spellings, matched exactly but
#: case-insensitively. Order within each tuple is preference order.
#: DELIBERATELY NOT VERIFIED against a live granule: see the module docstring.
INSAT_VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "tir1_bt": ("TIR1_BT", "BT_TIR1", "TIR1", "tir1_bt"),
    "tir2_bt": ("TIR2_BT", "BT_TIR2", "TIR2", "tir2_bt"),
    "wv_bt": ("WV_BT", "BT_WV", "WV", "wv_bt"),
    "vis_refl": ("VIS_REFL", "VIS", "VISIBLE_REFLECTANCE", "vis_refl"),
    "swir_refl": ("SWIR_REFL", "SWIR", "swir_refl"),
    "mir_bt": ("MIR_BT", "BT_MIR", "MIR", "mir_bt"),
}

#: Unit spellings that mean "already an absolute temperature in kelvin".
_KELVIN_UNITS = frozenset({"k", "kelvin", "degk", "deg k", "k "})

#: Substrings that identify a spectral-radiance unit, which cannot be converted
#: here without the published calibration coefficients.
_RADIANCE_MARKERS = ("mw m-2", "sr-1", "sr^-1", "(cm-1)", "cm-1", "radiance")

#: Unit spellings that mean "already a fraction in [0, 1]".
_FRACTION_UNITS = frozenset({"", "-", "1", "none", "fraction", "unitless", "dimensionless"})

#: Unit spellings that mean "percent", divided by 100 and recorded.
_PERCENT_UNITS = frozenset({"%", "percent", "percentage"})

#: Coordinate aliases for *geolocation values*, matched case-insensitively.
#: Deliberately excludes ``y``/``x``: those are index axes in most granules and
#: treating them as degrees would place a swath on the grid at nonsense
#: coordinates. A granule without real lat/lon must fail, not be mis-geolocated.
_LAT_COORD_ALIASES: tuple[str, ...] = ("lat", "latitude")
_LON_COORD_ALIASES: tuple[str, ...] = ("lon", "longitude")

#: Dimension aliases, including the index axes, used only to find a variable's
#: own lat/lon dimensions for stacking.
_LAT_ALIASES: tuple[str, ...] = (*_LAT_COORD_ALIASES, "y")
_LON_ALIASES: tuple[str, ...] = (*_LON_COORD_ALIASES, "x")
_TIME_ALIASES: tuple[str, ...] = ("time", "valid_time", "scantime", "scan_time", "date_time")


# --------------------------------------------------------------------------- #
# Axis and coordinate lookup (case-insensitive: HDF-sourced granules vary)
# --------------------------------------------------------------------------- #
def _coord_name(dataset: Any, aliases: Sequence[str]) -> str | None:
    """First coordinate, dimension or data variable matching ``aliases``.

    Data variables are searched last so a swath granule that stores its
    geolocation as 2-D ``latitude``/``longitude`` variables is still *found* -
    and then correctly rejected - rather than being reported as having no
    geolocation at all.
    """
    wanted = {alias.lower() for alias in aliases}
    containers = (list(dataset.coords), list(dataset.dims), list(dataset.data_vars))
    for names in containers:
        for name in names:
            if str(name).lower() in wanted:
                return str(name)
    return None


def _variable_name(dataset: Any, aliases: Sequence[str]) -> str | None:
    """First data variable whose name matches ``aliases`` (case-insensitive).

    Exact matching only. A substring rule would let ``MIR`` match ``SWIR`` and
    attach the wrong channel, so no fuzzy matching is used anywhere.
    """
    for alias in aliases:
        for name in dataset.data_vars:
            if str(name).lower() == alias.lower():
                return str(name)
    return None


def _coord_values(dataset: Any, aliases: Sequence[str]) -> np.ndarray | None:
    name = _coord_name(dataset, aliases)
    if name is None or name not in dataset.variables:
        return None
    return np.asarray(dataset[name].values)


def _dim_name(variable: Any, aliases: Sequence[str]) -> str | None:
    wanted = {alias.lower() for alias in aliases}
    for dim in variable.dims:
        if str(dim).lower() in wanted:
            return str(dim)
    return None


def _classify_unit(unit: str | None) -> str:
    """``kelvin``, ``radiance``, ``percent``, ``fraction`` or ``unknown``."""
    text = (unit or "").strip().lower()
    if text in _KELVIN_UNITS:
        return "kelvin"
    if any(marker in text for marker in _RADIANCE_MARKERS):
        return "radiance"
    if text in _PERCENT_UNITS:
        return "percent"
    if text in _FRACTION_UNITS:
        return "fraction"
    return "unknown"




# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class INSATFileReport:
    """What one INSAT granule actually contains, measured rather than assumed."""

    path: str
    ok: bool = False
    #: Model channel -> variable name found in the file.
    detected: dict[str, str] = field(default_factory=dict)
    #: Data variables present that no alias claimed.
    unmapped: list[str] = field(default_factory=list)
    times: list[str] = field(default_factory=list)
    lat_range: tuple[float, float] | None = None
    lon_range: tuple[float, float] | None = None
    crs: str = ""
    #: Model channel -> declared unit, verbatim from the file.
    units: dict[str, str] = field(default_factory=dict)
    #: Nominal grid spacing [km] derived from the coordinates, or ``None``.
    resolution_km: float | None = None
    product_id: str = ""
    quality: QualityReport = field(default_factory=QualityReport)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "detected": dict(self.detected),
            "unmapped": list(self.unmapped),
            "times": list(self.times),
            "lat_range": list(self.lat_range) if self.lat_range else None,
            "lon_range": list(self.lon_range) if self.lon_range else None,
            "crs": self.crs,
            "units": dict(self.units),
            "resolution_km": self.resolution_km,
            "product_id": self.product_id,
            "quality": self.quality.to_dict(),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class INSATChannelSet:
    """Model channels read from one granule, already on the model grid.

    ``channels`` maps model channel name -> ``(T, H, W)`` float array in that
    channel's physical unit. A channel the file does not supply is *absent* from
    the mapping; it is never present as zeros.
    """

    channels: dict[str, np.ndarray]
    times: list[datetime]
    coverage: dict[str, float]
    source_grid: Any = None
    records: list[ResamplingRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_insat_file(path: str | Path, *, product_id: str = "") -> INSATFileReport:
    """Inspect one INSAT granule and report what is genuinely usable.

    Fails ``ok=False`` when the file cannot be opened, carries no recognised
    channel variable, declares no time, or uses a layout this reader cannot place
    on the model grid (curvilinear swath geolocation). Every failure is reported
    distinctly; nothing is silently defaulted.
    """
    from app.physics import CHANNEL_META_INDEX

    report = INSATFileReport(path=str(path), product_id=product_id)
    source = Path(path)
    if not source.is_file():
        report.quality.fail("the file does not exist, so no channel can be read")
        return report
    try:
        import xarray as xr

        dataset = xr.open_dataset(source)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a data fact
        report.quality.fail(
            f"the file could not be opened as NetCDF/HDF ({type(exc).__name__}); "
            "no channel is read and nothing is substituted"
        )
        return report

    try:
        for channel, aliases in INSAT_VARIABLE_ALIASES.items():
            found = _variable_name(dataset, aliases)
            if found:
                report.detected[channel] = found
                unit = dataset[found].attrs.get("units")
                if unit is not None:
                    report.units[channel] = str(unit)
        report.unmapped = sorted(
            str(name)
            for name in dataset.data_vars
            if str(name) not in set(report.detected.values())
        )
        if not report.detected:
            report.quality.fail(
                "no recognised INSAT channel variable "
                f"(found: {sorted(str(n) for n in dataset.data_vars) or 'none'}). "
                "The alias list is documented but unverified against a real "
                "granule, so an unfamiliar layout is reported rather than guessed at."
            )

        # ---- units: radiance-encoded variables are refused, not converted ----
        for channel, unit in report.units.items():
            kind = _classify_unit(unit)
            if kind == "radiance":
                report.quality.fail(
                    f"{channel} is stored as spectral radiance (units {unit!r}); "
                    "converting it needs the published calibration coefficients "
                    "(c1, c2), which are not hardcoded, so no brightness "
                    "temperature is derived"
                )
            elif kind == "unknown":
                report.quality.warn(
                    f"{channel} declares unit {unit!r}, which is neither kelvin "
                    "nor a fraction; values are used verbatim and range-checked"
                )
            elif kind == "percent" and channel.endswith("_refl"):
                report.notes.append(
                    f"{channel}: declared in percent, divided by 100 to a fraction"
                )

        _check_geolocation(dataset, report)
        _check_time(dataset, report)

        # ---- range sanity against the declared channel contract --------------
        for channel, variable in report.detected.items():
            meta = CHANNEL_META_INDEX[channel]
            values = np.asarray(dataset[variable].values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size and (finite.min() < meta.vmin or finite.max() > meta.vmax):
                report.quality.warn(
                    f"{channel} spans [{finite.min():.3g}, {finite.max():.3g}] "
                    f"outside the declared range [{meta.vmin:g}, {meta.vmax:g}]; "
                    "out-of-range cells are masked to NaN"
                )
        report.quality.n_values = len(report.detected)
        report.quality.n_missing = len(INSAT_VARIABLE_ALIASES) - len(report.detected)
        report.ok = report.quality.passed
        return report
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _check_geolocation(dataset: Any, report: INSATFileReport) -> None:
    """Record the granule's lat/lon layout, failing closed on a swath grid."""
    lats = _coord_values(dataset, _LAT_COORD_ALIASES)
    lons = _coord_values(dataset, _LON_COORD_ALIASES)
    if lats is None or lons is None:
        report.quality.fail(
            "no latitude/longitude coordinates: the granule cannot be geolocated "
            "onto the model grid"
        )
        return
    if lats.ndim == 2 or lons.ndim == 2:
        report.quality.fail(
            "curvilinear (2-D) latitude/longitude found: this is a swath product "
            "and swath geolocation is not implemented, so no channel is produced "
            "rather than resampling it incorrectly"
        )
        return
    if lats.size < 2 or lons.size < 2:
        report.quality.fail("degenerate 1-D latitude/longitude axis")
        return
    report.lat_range = (float(np.nanmin(lats)), float(np.nanmax(lats)))
    report.lon_range = (float(np.nanmin(lons)), float(np.nanmax(lons)))
    report.resolution_km = round(source_grid_spacing(lats, lons), 4)
    report.crs = str(
        dataset.attrs.get("crs") or dataset.attrs.get("projection") or "EPSG:4326"
    )


def _check_time(dataset: Any, report: INSATFileReport) -> None:
    """Record valid times, preferring the coordinate over a header attribute."""
    time_name = _coord_name(dataset, _TIME_ALIASES)
    if time_name is None:
        for attr in ("date", "scan_time", "time"):
            if attr in dataset.attrs:
                report.times = [str(dataset.attrs[attr])]
                report.notes.append(
                    f"no time coordinate; the {attr!r} attribute supplies the "
                    "valid time of the granule"
                )
                return
    else:
        try:
            report.times = [
                str(v) for v in dataset[time_name].dt.strftime("%Y-%m-%dT%H:%M:%S").values
            ]
        except (AttributeError, TypeError):
            report.times = [
                str(v) for v in np.asarray(dataset[time_name].values).ravel()
            ]
    if not report.times:
        report.quality.fail(
            "no valid time could be read from the granule; a snapshot with no "
            "time cannot be placed in an observational series"
        )



# --------------------------------------------------------------------------- #
# Reading one granule onto the model grid
# --------------------------------------------------------------------------- #
def _outside_source(target: GridSpec, source: GridSpec) -> np.ndarray:
    """Boolean mask of target cells outside the source grid's extent."""
    lons = np.asarray(target.lon_centers())
    lats = np.asarray(target.lat_centers())
    inside_x = (lons >= source.min_lon) & (lons <= source.max_lon)
    inside_y = (lats >= source.min_lat) & (lats <= source.max_lat)
    return ~np.outer(inside_y, inside_x)


def read_insat_channels(
    path: str | Path, grid: GridSpec, *, report: INSATFileReport | None = None
) -> INSATChannelSet:
    """Read the six INSAT channel fields and put them on ``grid``.

    Raises :class:`NotImplementedError` for a curvilinear swath granule and
    :class:`ValueError` when the file fails validation or supplies no finite
    value for any channel. Nothing is approximated in either case.
    """
    import xarray as xr

    report = report or validate_insat_file(path)
    source = Path(path)

    with xr.open_dataset(source) as dataset:
        lats = _coord_values(dataset, _LAT_COORD_ALIASES)
        lons = _coord_values(dataset, _LON_COORD_ALIASES)
        if (lats is not None and lats.ndim == 2) or (lons is not None and lons.ndim == 2):
            raise NotImplementedError(
                f"swath geolocation is not implemented: {source.name} carries "
                "curvilinear 2-D latitude/longitude, so its channels cannot be "
                "placed on the model grid; nothing is approximated"
            )
        if not report.ok:
            raise ValueError(
                f"{source.name} failed validation: "
                + "; ".join(report.quality.errors or ["unknown reason"])
            )
        if lats is None or lons is None or lats.ndim != 1 or lons.ndim != 1:
            raise ValueError(
                f"{source.name} does not have a usable 1-D lat/lon grid; a "
                "curvilinear or absent grid cannot be aligned to the model grid"
            )

        notes: list[str] = []
        records: list[ResamplingRecord] = []
        descending = bool(np.all(np.diff(lats) < 0))
        lats = np.sort(np.asarray(lats, dtype=np.float64))
        lons = np.asarray(lons, dtype=np.float64)
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
                        "the granule stores latitude descending; the field is "
                        "flipped so it matches the model's north-up raster"
                    ),
                )
            )

        source_grid = GridSpec(
            min_lon=float(lons[0]),
            min_lat=float(lats[0]),
            max_lon=float(lons[-1]),
            max_lat=float(lats[-1]),
            res_km=source_grid_spacing(lats, lons),
            nx=int(lons.size),
            ny=int(lats.size),
        )
        outside = _outside_source(grid, source_grid)
        records.append(_horizontal_alignment_record(source_grid, grid, report))
        channels, n_times, notes = _read_channel_fields(
            dataset, report, grid, source_grid, outside, descending, notes
        )

        if not channels:
            raise ValueError(
                f"{source.name} supplied no usable INSAT channel: "
                + ("; ".join(notes) or "no recognised channel variable")
            )

    times = _valid_times(report, n_times, notes)
    coverage = {
        name: round(float(np.mean(np.isfinite(values))), 6)
        for name, values in channels.items()
    }
    return INSATChannelSet(
        channels=channels,
        times=times,
        coverage=coverage,
        source_grid=source_grid,
        records=records,
        notes=notes,
    )



def _read_channel_fields(
    dataset: Any,
    report: INSATFileReport,
    target_grid: GridSpec,
    source_grid: GridSpec,
    outside: np.ndarray,
    descending: bool,
    notes: list[str],
) -> tuple[dict[str, np.ndarray], int, list[str]]:
    """Geolocate, resample and range-check every detected channel.

    A channel whose frame count disagrees with the series, whose layout carries
    no lat/lon axes, or whose values are all missing is *omitted* and explained -
    never padded, averaged or filled from a neighbour.
    """
    from app.physics import CHANNEL_META_INDEX

    channels: dict[str, np.ndarray] = {}
    n_times = 0
    for channel, variable_name in report.detected.items():
        variable = dataset[variable_name]
        lat_dim = _dim_name(variable, _LAT_ALIASES)
        lon_dim = _dim_name(variable, _LON_ALIASES)
        if lat_dim is None or lon_dim is None:
            notes.append(
                f"{channel}: variable {variable_name!r} carries no lat/lon "
                "dimensions, so it cannot be geolocated; omitted"
            )
            continue
        order: list[str | None] = [
            _dim_name(variable, _TIME_ALIASES), None, lat_dim, lon_dim,
        ]
        values = _stack_axes(variable, order).astype(np.float64)[:, 0, :, :]
        if descending:
            values = values[:, ::-1, :]
        values[~_mask_fill(values)] = np.nan
        if channel.endswith("_refl") and _classify_unit(report.units.get(channel)) == "percent":
            values = values / 100.0
            notes.append(f"{channel}: percent converted to a fraction")

        field = _resample_frames(values, source_grid, target_grid, outside)
        if n_times == 0:
            n_times = int(field.shape[0])
        elif field.shape[0] != n_times:
            notes.append(
                f"{channel}: {field.shape[0]} frame(s) against {n_times} for the "
                "other channels; omitted rather than padded"
            )
            continue

        masked, mask_note = _mask_out_of_range(channel, field, CHANNEL_META_INDEX)
        if mask_note:
            notes.append(mask_note)
        if not np.isfinite(masked).any():
            notes.append(
                f"{channel}: no finite value survived geolocation and range "
                "checks; the channel is omitted and stays NaN"
            )
            continue
        channels[channel] = masked
    return channels, n_times, notes


def _resample_frames(
    values: np.ndarray, source_grid: GridSpec, target_grid: GridSpec, outside: np.ndarray
) -> np.ndarray:
    """Bilinear-resample each frame onto the model grid, NaN outside the granule."""
    frames = [
        _set_outside_nan(
            np.asarray(target_grid.resample_bilinear(source_grid, values[index]), dtype=np.float64),
            outside,
        )
        for index in range(values.shape[0])
    ]
    if not frames:
        return np.empty((0, *target_grid.shape), dtype=np.float64)
    return np.stack(frames, axis=0)


def _set_outside_nan(frame: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """NaN any target cell the granule's own extent does not cover."""
    frame[outside] = np.nan
    return frame


def _mask_out_of_range(
    channel: str, field: np.ndarray, index: dict[str, Any]
) -> tuple[np.ndarray, str | None]:
    """NaN every cell outside the channel's declared physical range."""
    meta = index[channel]
    out_of_range = np.isfinite(field) & ((field < meta.vmin) | (field > meta.vmax))
    count = int(np.count_nonzero(out_of_range))
    if not count:
        return field, None
    masked = np.array(field, copy=True)
    masked[out_of_range] = np.nan
    return masked, (
        f"{channel}: {count} cell(s) outside the declared range "
        f"[{meta.vmin:g}, {meta.vmax:g}] masked to NaN"
    )


def _horizontal_alignment_record(
    source: GridSpec, target: GridSpec, report: INSATFileReport
) -> ResamplingRecord:
    nominal = "not declared" if report.resolution_km is None else f"{report.resolution_km:g} km"
    return ResamplingRecord(
        stage="horizontal_alignment",
        method="bilinear on lat/lon centres",
        source_resolution=f"{source.ny}x{source.nx} at ~{source.res_km:.2f} km",
        target_resolution=(
            f"{target.ny}x{target.nx} at ~{target.effective_res_km[1]:.2f} km"
        ),
        detail=(
            "INSAT imagery resampled onto the model grid with "
            "app.grid.GridSpec.resample_bilinear. Target cells outside the "
            "granule's longitude/latitude range are NaN, not clamped to the edge "
            f"value. Granule-declared grid spacing: {nominal}."
        ),
    )


def _valid_times(report: INSATFileReport, n_times: int, notes: list[str]) -> list[datetime]:
    """Parse the granule's declared times, one per data frame, or refuse."""
    if n_times == 0:
        return []
    parsed: list[datetime] = []
    for text in report.times:
        try:
            parsed.append(parse_time(text))
        except ValueError:
            notes.append(f"time {text!r} could not be parsed and was dropped")
    if not parsed:
        raise ValueError(
            f"{Path(report.path).name} declares times {report.times!r}, none of "
            "which could be parsed, so no frame can be placed on the time axis"
        )
    if len(parsed) == n_times:
        return parsed
    if len(parsed) > n_times:
        notes.append(
            f"{len(parsed)} valid time(s) for {n_times} frame(s); the tail was "
            "dropped rather than guessed"
        )
        return parsed[:n_times]
    raise ValueError(
        f"{Path(report.path).name} declares {len(parsed)} valid time(s) for "
        f"{n_times} data frame(s); a frame with no time is not placed on the axis"
    )



# --------------------------------------------------------------------------- #
# Derived channels (the derivations CHANNEL_META already declares)
# --------------------------------------------------------------------------- #
def derive_ctt(tir1_bt: np.ndarray) -> np.ndarray:
    """Cloud-top temperature from the 10.8 um window channel.

    Exactly the relation :data:`app.physics.CHANNEL_META` records: CTT *is* the
    TIR1 brightness temperature. No atmospheric correction or split-window
    adjustment is applied, because that would be a different quantity and this
    project has no verified coefficients for one. Cells outside
    ``CTT_BOUNDS`` or already NaN stay NaN.
    """
    from app.physics import CHANNEL_META_INDEX

    field = np.asarray(tir1_bt, dtype=np.float64)
    masked, _note = _mask_out_of_range("ctt", field, CHANNEL_META_INDEX)
    return np.where(np.isfinite(masked), masked, np.nan)


def derive_ctt_cooling_rate(
    ctt: np.ndarray, times: Sequence[datetime]
) -> tuple[np.ndarray, list[str]]:
    """``dCTT/dt`` [K h-1] between consecutive frames.

    The rate is a real finite difference over the *actual* time separation of the
    two frames, so a series that is not on the model's 30-minute cadence produces
    a correct hourly rate plus an explicit note rather than a rescaled number.
    The first frame is always NaN: a rate needs two samples.
    """
    from app.physics import CHANNEL_META_INDEX

    field = np.asarray(ctt, dtype=np.float64)
    out = np.full_like(field, np.nan, dtype=np.float64)
    notes: list[str] = []
    if field.shape[0] < 2:
        return out, ["ctt_cooling_rate: fewer than two frames, so the rate is NaN everywhere"]
    off_cadence = False
    for index in range(1, field.shape[0]):
        hours = (times[index] - times[index - 1]).total_seconds() / 3600.0
        if hours <= 0:
            notes.append(
                f"ctt_cooling_rate: non-increasing time step at frame {index}; "
                "left NaN"
            )
            continue
        if abs(hours * 60.0 - FRAME_MINUTES) > 1e-6:
            off_cadence = True
        out[index] = (field[index] - field[index - 1]) / hours
    if off_cadence:
        notes.append(
            "ctt_cooling_rate: the frames are not on the model's 30-minute "
            "cadence; the rate is computed over the real time step and only the "
            "first difference is valid"
        )
    masked, note = _mask_out_of_range("ctt_cooling_rate", out, CHANNEL_META_INDEX)
    if note:
        notes.append(note)
    return masked, notes


def derive_wv_bt_anomaly(wv_bt: np.ndarray) -> np.ndarray:
    """``wv_bt`` minus the per-frame median of its own finite values.

    The background is measured from the frame itself, which is what
    :data:`app.physics.CHANNEL_META` states. A frame with no finite value stays
    entirely NaN - the anomaly is never computed against an assumed background.
    """
    from app.physics import CHANNEL_META_INDEX

    field = np.asarray(wv_bt, dtype=np.float64)
    out = np.full_like(field, np.nan, dtype=np.float64)
    for index in range(field.shape[0]):
        finite = field[index][np.isfinite(field[index])]
        if finite.size == 0:
            continue
        out[index] = field[index] - float(np.median(finite))
    masked, _note = _mask_out_of_range("wv_bt_anomaly", out, CHANNEL_META_INDEX)
    return masked


def add_derived_channels(channels: dict[str, np.ndarray], times: Sequence[datetime]) -> list[str]:
    """Add ``ctt``, ``ctt_cooling_rate`` and ``wv_bt_anomaly`` in place.

    A derived channel is produced only when its parent channel is present. It is
    never produced from a substitute parent, and never when the parent is all
    NaN. Returns the notes describing what was and was not derived.
    """
    notes: list[str] = []
    if "tir1_bt" in channels:
        channels["ctt"] = derive_ctt(channels["tir1_bt"])
        notes.append("ctt: taken as TIR1 10.8 um brightness temperature (no correction)")
        rate, rate_notes = derive_ctt_cooling_rate(channels["ctt"], times)
        channels["ctt_cooling_rate"] = rate
        notes.extend(rate_notes)
    else:
        notes.append("ctt: tir1_bt is unavailable, so CTT is not derived")
    if "wv_bt" in channels:
        channels["wv_bt_anomaly"] = derive_wv_bt_anomaly(channels["wv_bt"])
        notes.append("wv_bt_anomaly: wv_bt minus the per-frame median of its finite cells")
    else:
        notes.append(
            "wv_bt_anomaly: wv_bt is unavailable, so the anomaly is not derived"
        )
    return notes




# --------------------------------------------------------------------------- #
# Cube assembly with Phase 8.3/8.4 provenance
# --------------------------------------------------------------------------- #
def _coverage_string(report: INSATFileReport) -> str | None:
    """``"minLon-maxLonE minLat-maxLatN"`` from the granule's own extent."""
    if not report.lon_range or not report.lat_range:
        return None
    (min_lon, max_lon), (min_lat, max_lat) = report.lon_range, report.lat_range
    return f"{min_lon}-{max_lon}E {min_lat}-{max_lat}N"


def build_insat_cube(
    path: str | Path,
    grid: GridSpec,
    *,
    report: INSATFileReport | None = None,
    product_id: str = "",
) -> Any:
    """Assemble an :class:`~app.ingestion.base.ObservationCube` from one granule.

    Nine of the twelve model channels can be supplied: the six observed INSAT
    channels plus the three derivations ``CHANNEL_META`` declares. The remaining
    three (``iwv``, ``cape``, ``elevation``) need IMDAA and DEM inputs and stay
    **NaN**; ``metadata["channels_available"]`` / ``["channels_missing"]`` record
    the split so no consumer can mistake this for a complete model input.

    The provenance record carries the SHA-256 of the granule's *actual bytes* and
    a one-entry :class:`~app.ingestion.base.SourceFile` manifest, so the Phase 8.3
    verifier can re-hash the file later.

    Raises :class:`ValueError` when the granule fails validation or yields no
    finite value for any channel, rather than returning an all-NaN cube.
    """
    from app.ingestion.base import ObservationCube, Provenance
    from app.physics import (
        CHANNEL_META_INDEX,
        N_CHANNELS,
        channel_index,
        channel_units,
    )

    source = Path(path)
    report = report or validate_insat_file(source, product_id=product_id)
    if not report.ok:
        raise ValueError(
            f"{source.name} failed validation: "
            + "; ".join(report.quality.errors or ["unknown reason"])
        )

    read = read_insat_channels(source, grid, report=report)
    notes = list(read.notes)
    notes.extend(add_derived_channels(read.channels, read.times))
    times = read.times
    n_times = next(iter(read.channels.values())).shape[0]

    channels = np.full((n_times, N_CHANNELS, *grid.shape), np.nan, dtype=np.float32)
    for name, values in read.channels.items():
        channels[:, channel_index(name), :, :] = values.astype(np.float32)

    available = [
        name for name in channel_units()
        if np.isfinite(channels[:, channel_index(name)]).any()
    ]
    missing = [name for name in channel_units() if name not in available]
    for name in missing:
        report.quality.warn(
            f"{name} is not supplied by INSAT and stays NaN "
            f"(declared source: {CHANNEL_META_INDEX[name].source})"
        )

    fraction = np.mean(np.isfinite(channels), axis=(1, 2, 3)).astype(np.float32)
    quality = np.broadcast_to(fraction[:, None, None], (n_times, *grid.shape)).copy()

    # -- Phase 8.3/8.4: digest and manifest from the bytes on disk -------------
    try:
        digest = file_sha256(source)
    except OSError:  # pragma: no cover - a file that vanished mid-read
        digest = None
    acquired_at = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc)
    source_file = (
        source_file_from_disk(
            source,
            provider=INSAT_PROVIDER,
            role="insat_l1b",
            source_identity=report.product_id or source.stem,
        )
        if digest is not None
        else None
    )
    coverage = _coverage_string(report)
    declared_units = {name: channel_units()[name] for name in available}
    return ObservationCube(
        grid=grid,
        times=times,
        channels=channels,
        quality=quality,
        provenance=[
            Provenance(
                source="MOSDAC INSAT-3D/3DR",
                product=report.product_id or source.name,
                valid_from=min(times),
                valid_to=max(times),
                path=str(source),
                is_synthetic=False,
                attribution=INSAT_ATTRIBUTION,
                source_sha256=digest,
                source_path=str(source),
                source_bytes_available=digest is not None,
                provider=INSAT_PROVIDER,
                acquired_at=acquired_at,
                channel_units=declared_units,
                coverage=coverage,
                source_files=[source_file] if source_file else [],
            )
        ],
        metadata={
            "source_file": str(source),
            "spatial_coverage": coverage,
            "channel_units": declared_units,
            "channels_available": available,
            "channels_missing": missing,
            "channel_coverage": read.coverage,
            "product_id": report.product_id,
            "declared_resolution_km": report.resolution_km,
            "granule_times": list(report.times),
            "acquired_at_source": (
                "granule file modification time (the earliest verifiable instant "
                "associated with the bytes)"
            ),
            "resampling": [r.to_dict() for r in read.records],
            "notes": notes,
            "trainable": False,
            "trainable_reason": (
                f"INSAT supplied {len(available)} of {N_CHANNELS} channels. The "
                f"remaining {len(missing)} need IMDAA (iwv, cape) and DEM "
                "(elevation) inputs, so this cube is not a complete model input."
            ),
        },
    )
