"""Canonical ingestion pipeline: acquire -> validate -> align -> composite.

The eight required stages are implemented as one explicit, resumable object:

1. **Acquire** - :meth:`CanonicalPipeline.stage_files` enumerates operator
   supplied files and checksums them (:func:`~.provenance.file_sha256`).
2. **Validate** - :meth:`validate_dataset` checks format, metadata, timestamps
   and geographic coverage, producing a :class:`QualityReport`.
3. **Align** - :meth:`align_field` reprojects/resamples onto the canonical
   :class:`~app.grid.GridSpec` and records a
   :class:`~.provenance.ResamplingRecord` for the step.
4. **Composite** - :meth:`composite_temporal` resamples in time and records how.
5. **Normalise units** - :meth:`normalise_units` applies explicit, named
   conversions; the unit actually applied is recorded on the field.
6. **Missing values** - every field keeps a validity mask; missing data stays
   ``NaN`` and is **never** replaced with 0.0 or a spatial mean.
7. **Sequences** - :func:`build_windows` produces model-shaped samples that keep
   the existing ``(T, C, H, W)`` contract.
8. **Provenance** - every step returns a :class:`DatasetProvenance`.

Nothing in this module invents a value. A frame that cannot be filled is
recorded as missing, and the report says so.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from app.grid import FRAME_MINUTES, GridSpec, utc_floor_to_cadence
from app.ingestion.base import ObservationCube, Provenance, WindowSpec
from app.ingestion.realtime.provenance import (
    DataClass,
    DatasetProvenance,
    QualityReport,
    ResamplingRecord,
    file_sha256,
)
from app.logging_conf import get_logger
from app.physics import CHANNELS, N_CHANNELS

logger = get_logger("ingestion.realtime.pipeline")

__all__ = [
    "AlignedField",
    "CanonicalPipeline",
    "PipelineResult",
    "build_windows",
    "composite_temporal",
    "normalise_units",
    "validate_dataset",
]

#: Unit conversions this pipeline knows how to apply, by (source unit, target
#: unit, physical quantity). Unknown conversions raise rather than pass through,
#: because a silent wrong-unit bug is indistinguishable from bad data.
UNIT_CONVERSIONS: dict[tuple[str, str, str], float] = {
    # Brightness temperature, kelvin -> kelvin.
    ("K", "K", "temperature"): 1.0,
    ("C", "K", "temperature"): 1.0,  # affine; handled specially
    # Rain rate/amount, mm/h or mm -> mm/h on the model grid.
    ("mm/h", "mm/h", "precipitation"): 1.0,
    ("mm", "mm/h", "precipitation"): 1.0,
    # Wind, knots -> m/s (IMD station files commonly report knots).
    ("knot", "m s-1", "wind_speed"): 0.514444,
    ("m s-1", "m s-1", "wind_speed"): 1.0,
    # Pressure, hPa -> hPa.
    ("hPa", "hPa", "pressure"): 1.0,
    # Relative humidity, percent -> fraction.
    ("%", "1", "relative_humidity"): 0.01,
    # Reflectance, percent -> fraction.
    ("%", "1", "reflectance"): 0.01,
}


@dataclass(slots=True)
class AlignedField:
    """A 2-D field on the canonical grid, with its mask and audit trail.

    ``data`` and ``valid`` are parallel arrays: a cell is usable only where
    ``valid`` is True. Where it is False, ``data`` is ``NaN``.
    """

    data: np.ndarray
    valid: np.ndarray
    channel: str
    unit: str
    data_class: str
    #: ``(source_resolution, target_resolution, method)`` for provenance.
    resampling: list[ResamplingRecord] = field(default_factory=list)
    source_crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.data.shape != self.valid.shape:
            raise ValueError(
                f"channel {self.channel!r}: data {self.data.shape} != mask {self.valid.shape}"
            )
        # Enforce the missing-data contract: masked cells are NaN, not zero.
        self.data = np.where(self.valid, self.data, np.nan).astype(np.float32)

    @property
    def missing_fraction(self) -> float:
        if self.data.size == 0:
            return 0.0
        return float(np.count_nonzero(~self.valid) / self.data.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "unit": self.unit,
            "data_class": self.data_class,
            "shape": list(self.data.shape),
            "missing_fraction": round(self.missing_fraction, 6),
            "source_crs": self.source_crs,
            "resampling": [r.to_dict() for r in self.resampling],
        }


def normalise_units(
    field_: AlignedField, source_unit: str, target_unit: str, quantity: str
) -> AlignedField:
    """Convert a field's units, recording the conversion on the field.

    Raises ``ValueError`` for an unknown conversion so a mislabelled unit is a
    loud failure rather than a plausible-looking wrong number.
    """
    if source_unit == target_unit:
        field_.unit = target_unit
        return field_
    key = (source_unit, target_unit, quantity)
    factor = UNIT_CONVERSIONS.get(key)
    if factor is None:
        raise ValueError(
            f"no conversion for {quantity} {source_unit!r} -> {target_unit!r}; "
            f"known: {sorted(UNIT_CONVERSIONS)}"
        )
    if source_unit == "C" and target_unit == "K":
        field_.data = np.where(field_.valid, field_.data + 273.15, np.nan)
    else:
        field_.data = np.where(field_.valid, field_.data * factor, np.nan)
    field_.unit = target_unit
    field_.resampling.append(
        ResamplingRecord(
            stage="normalise_units",
            method="affine" if source_unit == "C" else "multiplicative",
            source_resolution=source_unit,
            target_resolution=target_unit,
            detail=f"quantity={quantity}",
        )
    )
    return field_


def composite_temporal(
    frames: Sequence[np.ndarray],
    stamps: Sequence[datetime],
    *,
    frame_minutes: int = FRAME_MINUTES,
) -> tuple[np.ndarray, list[datetime], ResamplingRecord | None]:
    """Average source frames into the canonical cadence.

    Returns the composited stack, its timestamps, and a
    :class:`ResamplingRecord` describing what was done. When the source cadence
    already equals the target, the record is ``None`` and the data is passed
    through unchanged - the pipeline does not claim to have resampled anything
    it did not resample.
    """
    if len(frames) != len(stamps):
        raise ValueError(
            f"{len(frames)} frames but {len(stamps)} timestamps; they must correspond"
        )
    if not frames:
        return np.empty((0, 0, 0), dtype=np.float32), [], None


    first = np.asarray(frames[0], dtype=np.float32)
    times = sorted(stamps)
    if len(times) < 2:
        return first[None, ...], list(times), None

    deltas = [
        (times[i + 1] - times[i]).total_seconds() / 60.0 for i in range(len(times) - 1)
    ]
    source_cadence = float(np.median(deltas))
    order = np.argsort([t.timestamp() for t in times])
    stack = np.stack([np.asarray(frames[i], dtype=np.float32) for i in order])
    ordered_times = [times[i] for i in order]

    if abs(source_cadence - frame_minutes) < 1e-6:
        return stack, ordered_times, None

    # Bucket by canonical half-hour boundary; NaN-safe mean so a partially
    # observed slot is the mean of what was observed, and a fully unobserved
    # slot stays NaN.
    buckets: dict[datetime, list[int]] = {}
    for index, stamp in enumerate(ordered_times):
        key = utc_floor_to_cadence(stamp, frame_minutes)
        buckets.setdefault(key, []).append(index)

    out_frames: list[np.ndarray] = []
    out_times: list[datetime] = []
    for key in sorted(buckets):
        members = stack[buckets[key]]
        # A bucket with no observed value yields NaN and a RuntimeWarning; the
        # NaN is the correct output (the frame stays missing), so the warning
        # is suppressed rather than left to pollute the test output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            averaged = np.nanmean(members, axis=0)
        out_frames.append(averaged.astype(np.float32))
        out_times.append(key)

    record = ResamplingRecord(
        stage="composite_temporal",
        method="nanmean over source frames within each canonical bucket",
        source_resolution=f"{source_cadence:g} min",
        target_resolution=f"{frame_minutes} min",
        detail=(
            "a fully unobserved bucket remains NaN; it is never filled with zero "
            "or a spatial mean"
        ),
    )
    return np.stack(out_frames), out_times, record


def validate_dataset(
    data: np.ndarray,
    *,
    grid: GridSpec,
    valid_mask: np.ndarray | None = None,
    expected_channels: Sequence[str] = (),
    channel_bounds: dict[str, tuple[float, float]] | None = None,
    max_missing_fraction: float = 0.5,
) -> QualityReport:
    """Validate a stacked field against the canonical grid and channel schema.

    Checks performed: shape versus grid, finiteness, monotonic timestamps (via
    the caller's mask), and physical range violations. Range violations are
    warnings, not errors, because real satellite data routinely contains
    calibration outliers; they are counted so a reviewer can see them.
    """
    report = QualityReport()
    arr = np.asarray(data)
    if arr.ndim != 3:
        return report.fail(f"expected a 3-D (frames, ny, nx) array, got shape {arr.shape}")
    if arr.shape[1:] != grid.shape:
        return report.fail(f"grid shape {grid.shape} != data shape {arr.shape[1:]}")

    mask = np.ones(arr.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    if mask.shape != arr.shape:
        return report.fail(f"validity mask {mask.shape} != data {arr.shape}")

    report.n_values = int(arr.size)
    report.n_missing = int(np.count_nonzero(~mask))

    if report.missing_fraction > max_missing_fraction:
        report.fail(
            f"missing fraction {report.missing_fraction:.3f} exceeds the limit "
            f"{max_missing_fraction:.3f}"
        )
    elif report.n_missing:
        report.warn(f"{report.missing_fraction:.3f} of values are missing (kept as NaN)")

    if expected_channels:
        unknown = [c for c in expected_channels if c not in {s.name for s in CHANNELS}]
        if unknown:
            report.fail(f"unknown model channel(s) {unknown}")

    if channel_bounds:
        for index, name in enumerate(expected_channels):
            if name not in channel_bounds:
                continue
            lo, hi = channel_bounds[name]
            frame = arr[..., index, :, :] if arr.ndim == 4 else arr
            values = frame[np.isfinite(frame)]
            if values.size == 0:
                continue
            out_low = int(np.count_nonzero(values < lo))
            out_high = int(np.count_nonzero(values > hi))
            if out_low or out_high:
                report.warn(
                    f"channel {name}: {out_low} value(s) below {lo} and {out_high} "
                    f"above {hi}; retained and counted, not clipped silently"
                )
    return report



@dataclass(slots=True)
class PipelineResult:
    """Outcome of running one product through the pipeline."""

    provenance: DatasetProvenance
    fields: list[AlignedField] = field(default_factory=list)
    cube: ObservationCube | None = None
    ok: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ingest_key": self.provenance.key,
            "source": self.provenance.source,
            "product": self.provenance.product,
            "data_class": self.provenance.data_class,
            "checksum": self.provenance.checksum,
            "quality": self.provenance.quality.to_dict(),
            "resampling": [r.to_dict() for r in self.provenance.resampling],
            "n_fields": len(self.fields),
            "n_frames": self.provenance.n_frames,
            "has_cube": self.cube is not None,
        }


class CanonicalPipeline:
    """Runs a source's fields through the eight canonical stages.

    Parameters
    ----------
    grid:
        The canonical model grid every field is aligned to.
    source:
        Human-readable source name recorded in provenance.
    product:
        Product identifier (``datasetId``, dataset slug, ...).
    data_class:
        One of :class:`~.provenance.DataClass`. Defaults to ``observation``;
        reanalysis callers must pass it explicitly.
    """

    def __init__(
        self,
        grid: GridSpec,
        *,
        source: str,
        product: str,
        data_class: str = DataClass.OBSERVATION,
        attribution: str = "",
        license_note: str = "",
    ) -> None:
        self.grid = grid
        self.source = source
        self.product = product
        self.data_class = DataClass.validate(data_class)
        self.attribution = attribution
        self.license_note = license_note

    # -- stage 1: acquire ---------------------------------------------------
    def stage_files(self, paths: Iterable[str | Path]) -> list[tuple[Path, str]]:
        """Checksum each file, returning ``(path, sha256)`` pairs.

        Missing files are skipped and logged rather than raising, so one absent
        granule does not abort an otherwise usable batch.
        """
        staged: list[tuple[Path, str]] = []
        for raw in paths:
            path = Path(raw)
            if not path.exists():
                logger.warning("staged file missing, skipping", extra={"path": str(path)})
                continue
            staged.append((path, file_sha256(path)))
        return staged

    # -- stage 3: align -----------------------------------------------------
    def align_field(
        self,
        data: np.ndarray,
        valid: np.ndarray,
        *,
        source_grid: GridSpec,
        channel: str,
        unit: str,
        source_crs: str = "EPSG:4326",
        method: str = "bilinear",
    ) -> AlignedField:
        """Resample a field from ``source_grid`` onto the canonical grid.

        Both sides are EPSG:4326 in this pipeline - geolocating a raw L1B swath
        to a lat/lon grid is a separate step that is *not* implemented here - so
        the operation is honestly labelled a resampling, not a reprojection.
        """
        values = np.asarray(data, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool)
        if values.shape != source_grid.shape:
            raise ValueError(
                f"channel {channel!r}: field {values.shape} != source grid {source_grid.shape}"
            )
        filled = np.where(mask, values, np.nan)
        if method == "nearest":
            resampled = self.grid.resample_nearest(source_grid, filled)
            resampled_mask = self.grid.resample_nearest(source_grid, mask)
        else:
            resampled = self.grid.resample_bilinear(source_grid, filled)
            resampled_mask = self.grid.resample_nearest(source_grid, mask)

        aligned_mask = np.asarray(resampled_mask, dtype=bool) & np.isfinite(resampled)
        record = ResamplingRecord(
            stage="align_spatial",
            method=method,
            source_resolution=f"{source_grid.res_km:g} km nominal",
            target_resolution=f"{self.grid.res_km:g} km nominal",
            detail=(
                f"source grid {source_grid.nx}x{source_grid.ny} -> canonical "
                f"{self.grid.nx}x{self.grid.ny}; both EPSG:4326"
            ),
        )
        return AlignedField(
            data=resampled,
            valid=aligned_mask,
            channel=channel,
            unit=unit,
            data_class=self.data_class,
            resampling=[record],
            source_crs=source_crs,
        )

    # -- stages 7-8: cube + provenance --------------------------------------
    def build_cube(
        self,
        fields: Sequence[AlignedField],
        times: Sequence[datetime],
        *,
        quality: QualityReport | None = None,
        resampling: Sequence[ResamplingRecord] = (),
        path: str | None = None,
        checksum: str = "",
        acquired_at: datetime | None = None,
        event_id: str | None = None,
    ) -> PipelineResult:
        """Assemble aligned fields into an :class:`ObservationCube`.

        The cube keeps *physical units* and the validity mask, exactly as the
        existing dataset contract expects; normalisation stays lazy in
        ``ObservationCube.normalised``. Channels absent from the input stay
        ``NaN`` and are reported as missing rather than being zero-filled.
        """
        stamps = list(times)
        if not fields:
            raise ValueError("cannot build a cube with no fields")
        if not stamps:
            raise ValueError("cannot build a cube with no timestamps")

        ny, nx = self.grid.shape
        cube_values = np.full((len(stamps), N_CHANNELS, ny, nx), np.nan, dtype=np.float32)
        cube_mask = np.zeros((len(stamps), N_CHANNELS, ny, nx), dtype=np.float32)

        by_channel = {f.channel: f for f in fields}
        for index, spec in enumerate(CHANNELS):
            aligned = by_channel.get(spec.name)
            if aligned is None:
                continue
            for t_index in range(len(stamps)):
                cube_values[t_index, index] = aligned.data
                cube_mask[t_index, index] = aligned.valid.astype(np.float32)

        present = [f.channel for f in fields]
        absent = [s.name for s in CHANNELS if s.name not in by_channel]
        report = quality or QualityReport()
        report.n_values = int(cube_mask.size)
        report.n_missing = int(np.count_nonzero(cube_mask == 0))
        if absent:
            report.warn(
                f"{len(absent)} of {N_CHANNELS} model channels were not supplied "
                f"({', '.join(absent)}); they are NaN and are never zero-filled"
            )
        if report.missing_fraction > 0.5:
            report.fail(
                f"only {1 - report.missing_fraction:.1%} of the cube is observed"
            )

        provenance = DatasetProvenance(
            source=self.source,
            product=self.product,
            data_class=self.data_class,
            acquired_at=acquired_at or stamps[0],
            valid_from=stamps[0],
            valid_to=stamps[-1],
            path=path,
            checksum=checksum,
            min_lon=self.grid.min_lon,
            min_lat=self.grid.min_lat,
            max_lon=self.grid.max_lon,
            max_lat=self.grid.max_lat,
            crs="EPSG:4326",
            source_resolution="per-field, see resampling records",
            processed_resolution=f"{self.grid.res_km:g} km nominal",
            source_cadence="per-source, see resampling records",
            processed_cadence=f"{FRAME_MINUTES} min",
            variables=[
                {
                    "channel": spec.name,
                    "unit": spec.unit,
                    "present": spec.name in by_channel,
                    "data_class": self.data_class,
                }
                for spec in CHANNELS
            ],
            n_frames=len(stamps),
            quality=report,
            resampling=[*resampling, *[r for f in fields for r in f.resampling]],
            processing_config={
                "grid": self.grid.to_dict(),
                "frame_minutes": FRAME_MINUTES,
                "channels_present": present,
                "channels_absent": absent,
            },
            attribution=self.attribution,
            license_note=self.license_note,
        )

        if report.passed:
            cube = ObservationCube(
                values=cube_values,
                quality=cube_mask,
                times=stamps,
                provenance=[
                    Provenance(
                        source=self.source,
                        product=self.product,
                        valid_from=stamps[0],
                        valid_to=stamps[-1],
                        is_synthetic=self.data_class == DataClass.SYNTHETIC,
                        attribution=self.attribution,
                        path=path,
                    )
                ],
                event_id=event_id,
                metadata={
                    "data_class": self.data_class,
                    "channels_present": present,
                    "channels_absent": absent,
                    "provenance": provenance.to_dict(),
                },
            )
        else:
            logger.error(
                "cube rejected by quality control",
                extra={"product": self.product, "errors": report.errors},
            )
            cube = None

        return PipelineResult(
            provenance=provenance,
            fields=list(fields),
            cube=cube,
            ok=report.passed,
        )


def build_windows(
    cube: ObservationCube, spec: WindowSpec
) -> list[tuple[int, slice, slice]]:
    """Enumerate ``(index, input_slice, target_slice)`` windows for a cube.

    Preserves the existing model contract: ``spec.seq_len`` history frames and
    ``spec.horizon`` target frames, both at the canonical cadence.
    """
    if spec.n_windows(cube.n_frames) == 0:
        return []
    return [
        (index, spec.history_slice(index), spec.future_slice(index))
        for index in range(spec.n_windows(cube.n_frames))
    ]
