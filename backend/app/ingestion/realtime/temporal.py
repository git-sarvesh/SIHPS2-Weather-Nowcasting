"""Temporal feasibility of acquired data for the model's input contract (Phase 8).

The network consumes **six consecutive frames on a fixed 30-minute cadence**
(``app.grid.FRAME_MINUTES`` and ``app.ingestion.base.WindowSpec.seq_len``). This
module answers, from the times actually present in a dataset, whether that
contract can be met by genuine data - and refuses to help manufacture it.

The central prohibition
-----------------------
A 3-hourly reanalysis **cannot** be turned into six 30-minute frames by
repetition or interpolation and then presented as observed data. Repeating a
field fabricates temporal structure no instrument measured; interpolating
between 3-hourly states invents two of every six frames outright. Either would
inflate the apparent sample count and let the model learn the repetition pattern
instead of storm evolution. This module therefore *measures* cadence and reports
a verdict. It offers no upsampling path, and
:func:`assess_temporal_feasibility` has no resampling parameter at all.

Data classes are kept distinct, because they must not be conflated:

``observation``
    Measurements from an instrument.
``reanalysis``
    Model output constrained by observations (IMDAA, MERA). Not synthetic, but
    also not an observation.
``interpolation``
    Values the system computed from other values. Listed so a source can be
    explicitly excluded from it.
``model``
    Output of a forecasting or generative model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.grid import FRAME_MINUTES
from app.ingestion.base import parse_time
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.temporal")

__all__ = [
    "DATA_CLASSES",
    "NATIVE_FRAME_SOURCES",
    "TemporalFeasibility",
    "assess_temporal_feasibility",
    "cadence_matches_contract",
    "longest_contiguous_run",
]

#: The data classes this project distinguishes.
#:
#: ``synthetic`` is a fifth class, separate from all of them. A generated field
#: is not an observation, not a reanalysis and not a model run: it is none of
#: those, and the distinction has to survive into the readiness report rather
#: than being rounded to "observation".
DATA_CLASSES = ("observation", "reanalysis", "interpolation", "model", "synthetic")

#: Data classes that can legitimately supply genuine instrument frames.
NATIVE_FRAME_SOURCES = ("observation",)

#: Classes that carry no observational content at all.
NON_OBSERVATIONAL = ("interpolation", "model", "synthetic")


def cadence_matches_contract(native_cadence_minutes: float | None, required: int) -> bool:
    """Whether ``native_cadence`` equals the required cadence exactly.

    Exact, not "close enough": a source delivering frames every 28 or 32 minutes
    is not delivering the contract, and rounding it away would hide the mismatch.
    """
    if native_cadence_minutes is None:
        return False
    return abs(native_cadence_minutes - required) < 1e-6


def longest_contiguous_run(
    times: Sequence[datetime], cadence_minutes: int = FRAME_MINUTES
) -> int:
    """Longest run of times spaced exactly ``cadence_minutes`` apart."""
    if not times:
        return 0
    step = timedelta(minutes=cadence_minutes)
    best = run = 1
    for previous, current in zip(times, times[1:], strict=False):
        run = run + 1 if current - previous == step else 1
        best = max(best, run)
    return best



@dataclass(slots=True)
class TemporalFeasibility:
    """Whether genuine data can supply the model's six 30-minute input frames."""

    source: str
    data_class: str
    required_frames: int
    required_cadence_minutes: int
    distinct_times: int
    max_contiguous_run: int
    native_cadence_minutes: float | None
    largest_gap_minutes: float | None
    first_time: datetime | None
    last_time: datetime | None
    time_span_minutes: float | None
    #: Channels this source can genuinely supply at the required cadence.
    channels_available: list[str] = field(default_factory=list)
    channels_missing: list[str] = field(default_factory=list)
    #: Optional spatial description, e.g. ``"77.5-80.5E 29.0-31.5N"``.
    spatial_coverage: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cadence_matches(self) -> bool:
        return cadence_matches_contract(
            self.native_cadence_minutes, self.required_cadence_minutes
        )

    @property
    def provides_required_frames(self) -> bool:
        return self.max_contiguous_run >= self.required_frames

    @property
    def feasible(self) -> bool:
        """True only when real data supplies the full input window at cadence.

        Strict by construction: the cadence must match, the run must be long
        enough, and the data class must carry observational content. Nothing here
        can be satisfied by repeating or interpolating fields, because this
        module offers no way to do so.
        """
        if not self.cadence_matches or not self.provides_required_frames:
            return False
        return self.data_class in NATIVE_FRAME_SOURCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "data_class": self.data_class,
            "feasible": self.feasible,
            "cadence_matches_contract": self.cadence_matches,
            "provides_required_frames": self.provides_required_frames,
            "required_frames": self.required_frames,
            "required_cadence_minutes": self.required_cadence_minutes,
            "native_cadence_minutes": self.native_cadence_minutes,
            "max_contiguous_run": self.max_contiguous_run,
            "distinct_times": self.distinct_times,
            "largest_gap_minutes": self.largest_gap_minutes,
            "first_time": self.first_time.isoformat() if self.first_time else None,
            "last_time": self.last_time.isoformat() if self.last_time else None,
            "time_span_minutes": self.time_span_minutes,
            "channels_available": list(self.channels_available),
            "channels_missing": list(self.channels_missing),
            "spatial_coverage": self.spatial_coverage,
            "notes": list(self.notes),
        }



def assess_temporal_feasibility(
    times: Sequence[datetime | str],
    *,
    source: str,
    data_class: str,
    channels_available: Sequence[str] = (),
    channels_missing: Sequence[str] = (),
    required_frames: int = 6,
    required_cadence_minutes: int = FRAME_MINUTES,
    spatial_coverage: str | None = None,
) -> TemporalFeasibility:
    """Measure whether ``times`` can supply the model's input window.

    ``times`` are the genuine valid times present in the dataset. Duplicates are
    collapsed and the sequence sorted, so an unsorted or repeated index cannot
    change the verdict.

    This function never interpolates, repeats, or resamples anything; it only
    counts and measures.
    """
    if data_class not in DATA_CLASSES:
        raise ValueError(
            f"unknown data_class {data_class!r}; expected one of {DATA_CLASSES}"
        )
    parsed = sorted({parse_time(t) for t in times})
    notes: list[str] = []

    if not parsed:
        notes.append(
            "no valid times are present, so no input window can be formed; this "
            "is a snapshot or an unparsed time axis, not a series"
        )
        return TemporalFeasibility(
            source=source,
            data_class=data_class,
            required_frames=required_frames,
            required_cadence_minutes=required_cadence_minutes,
            distinct_times=0,
            max_contiguous_run=0,
            native_cadence_minutes=None,
            largest_gap_minutes=None,
            first_time=None,
            last_time=None,
            time_span_minutes=None,
            channels_available=list(channels_available),
            channels_missing=list(channels_missing),
            spatial_coverage=spatial_coverage,
            notes=notes,
        )

    gaps = [
        (b - a).total_seconds() / 60.0
        for a, b in zip(parsed, parsed[1:], strict=False)
    ]
    native_cadence = min(gaps) if gaps else None
    largest_gap = max(gaps) if gaps else None
    run = longest_contiguous_run(parsed, required_cadence_minutes)
    span = (parsed[-1] - parsed[0]).total_seconds() / 60.0

    if not cadence_matches_contract(native_cadence, required_cadence_minutes):
        notes.append(
            f"native cadence is {native_cadence:.0f} min but the model requires "
            f"{required_cadence_minutes} min. This field is NOT upsampled: "
            "repeating it would fabricate temporal structure, and interpolating "
            "it would invent two of every six frames. Both are refused, so this "
            "source cannot supply the six-frame input window."
        )
    if run < required_frames:
        notes.append(
            f"longest run of consecutive {required_cadence_minutes}-minute frames "
            f"is {run}, short of the {required_frames} the model consumes"
        )
    if largest_gap is not None and largest_gap > required_cadence_minutes * 1.5:
        notes.append(
            f"largest gap between valid times is {largest_gap:.0f} min, so the "
            "series is discontinuous"
        )
    if data_class == "synthetic":
        notes.append(
            "data class is 'synthetic': these fields were generated, not observed. "
            "They can demonstrate the pipeline but can never satisfy an "
            "observational training requirement."
        )
    elif data_class not in NATIVE_FRAME_SOURCES:
        notes.append(
            f"data class is {data_class!r}, which is not an instrument "
            "observation. It may be legitimate as a target or a diagnostic "
            "input, but it must not be presented as observed 30-minute data."
        )

    return TemporalFeasibility(
        source=source,
        data_class=data_class,
        required_frames=required_frames,
        required_cadence_minutes=required_cadence_minutes,
        distinct_times=len(parsed),
        max_contiguous_run=run,
        native_cadence_minutes=native_cadence,
        largest_gap_minutes=largest_gap,
        first_time=parsed[0],
        last_time=parsed[-1],
        time_span_minutes=span,
        channels_available=list(channels_available),
        channels_missing=list(channels_missing),
        spatial_coverage=spatial_coverage,
        notes=notes,
    )
