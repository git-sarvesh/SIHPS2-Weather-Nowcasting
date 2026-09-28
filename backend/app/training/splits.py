"""Temporal splitting helpers with explicit leakage protection.

The nowcasting problem is a sliding-window supervised task: one sample consumes
``seq_len`` past frames and predicts ``horizon`` future frames. Two windows that
overlap in time share observations *and* target frames, so a naive random split
lets the validation target period leak into training.

The split implemented here is:

* **chronological** - only earlier times train, later times validate/test (the
  regime a real nowcaster faces), and
* **embargoed** - windows whose *target period* would cross a split boundary are
  dropped, so no training target is ever scored on validation or vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Sequence

from app.ingestion.base import WindowSpec

__all__ = ["SPLIT_NAMES", "SplitAssignment", "SplitPlan", "assign_splits", "assert_no_leakage", "describe_split"]

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Which split one candidate window belongs to (``None`` = embargoed)."""

    index: int
    init_time: datetime
    target_start: datetime
    target_end: datetime
    split: SplitName | None

    @property
    def embargoed(self) -> bool:
        return self.split is None


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Result of :func:`assign_splits`: assignments plus per-split indices."""

    assignments: tuple[SplitAssignment, ...]
    indices: dict[str, tuple[int, ...]]
    embargoed: tuple[int, ...]
    fractions: tuple[float, float, float]
    embargo_frames: int

    def indices_for(self, split: str) -> tuple[int, ...]:
        return self.indices.get(split, ())

    def coverage(self) -> dict[str, int]:
        """Window counts per split plus the embargoed total (for reporting)."""
        return {name: len(self.indices.get(name, ())) for name in SPLIT_NAMES} | {
            "embargoed": len(self.embargoed)
        }


def _window_times(
    times: Sequence[datetime], spec: WindowSpec, index: int
) -> tuple[datetime, datetime, datetime]:
    """``(init_time, target_start, target_end)`` for window ``index``.

    ``init_time`` is the last input frame (the forecast issue time);
    ``target_start``/``target_end`` bracket the predicted frames.
    """
    if spec.n_windows(len(times)) <= index:
        raise IndexError(f"window {index} out of range for {len(times)} frames")
    init_time = times[index + spec.seq_len - 1]
    target_start = times[index + spec.seq_len]
    target_end = times[index + spec.total_frames - 1]
    return init_time, target_start, target_end


def assign_splits(
    times: Sequence[datetime],
    spec: WindowSpec,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    embargo_frames: int | None = None,
) -> SplitPlan:
    """Chronological, leakage-safe split of one event's windows.

    Parameters
    ----------
    times:
        Frame timestamps of a single contiguous sequence, oldest first.
    spec:
        Window geometry (``seq_len`` history, ``horizon`` target).
    fractions:
        ``(train, val, test)`` proportions of the *assignable* windows.
    embargo_frames:
        Frames excluded just before each of the val/test boundaries. Defaults to
        ``spec.horizon``, the minimum that guarantees disjoint target periods.
    """
    if len(fractions) != 3 or any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be three non-negative values, got {fractions!r}")
    total_fraction = float(sum(fractions))
    if total_fraction <= 0:
        raise ValueError("at least one split fraction must be positive")
    embargo = spec.horizon if embargo_frames is None else int(embargo_frames)
    if embargo < 0:
        raise ValueError("embargo_frames must be >= 0")
    if embargo < spec.horizon:
        # A window predicts ``horizon`` frames, so a narrower embargo cannot
        # guarantee disjoint target periods across the boundary. Failing loudly
        # here is safer than producing a split that silently leaks.
        raise ValueError(
            f"embargo_frames={embargo} is smaller than horizon={spec.horizon}, which would let "
            "training target frames overlap the validation/test target period. Use "
            f"embargo_frames >= {spec.horizon} (or omit it for the default)."
        )

    n_windows = spec.n_windows(len(times))
    assignments: list[SplitAssignment] = []
    for index in range(n_windows):
        init_time, target_start, target_end = _window_times(times, spec, index)
        assignments.append(
            SplitAssignment(
                index=index,
                init_time=init_time,
                target_start=target_start,
                target_end=target_end,
                split=None,
            )
        )
    if n_windows == 0:
        empty = {name: () for name in SPLIT_NAMES}
        return SplitPlan(tuple(assignments), empty, (), fractions, embargo)

    # Chronological cut points (index order == time order for one sequence).
    n_train = int(round(fractions[0] / total_fraction * n_windows))
    n_val = int(round(fractions[1] / total_fraction * n_windows))
    n_train = min(max(n_train, 0), n_windows)
    n_val = min(max(n_val, 0), n_windows - n_train)
    # Chronological cut points (index order == time order for one sequence).
    n_train = int(round(fractions[0] / total_fraction * n_windows))
    n_val = int(round(fractions[1] / total_fraction * n_windows))
    n_train = min(max(n_train, 0), n_windows)
    n_val = min(max(n_val, 0), n_windows - n_train)
    boundaries = {"val": n_train, "test": n_train + n_val}
    # Each boundary consumes ``embargo`` windows from the *end of the preceding
    # split* so their target periods cannot reach into the next split.
    regions = {
        "val": (0, max(0, n_train - embargo)),
        "test": (n_train, max(n_train, boundaries["test"] - embargo)),
    }

    # Embargo rationale: a window ``i`` predicts frames
    # ``[i+seq_len, i+total_frames-1]``. Keeping the last usable train index
    # ``m`` and the first val index ``n_train`` requires
    # ``m + total_frames - 1 < n_train + seq_len``, i.e. ``m < n_train - horizon``;
    # cutting the train region at ``n_train - embargo`` with
    # ``embargo = horizon`` satisfies this with equality as the worst case.
    # A split too short to absorb the embargo is reported empty rather than
    # silently overlapping its neighbour (see ``describe_split``).
    updated: list[SplitAssignment] = []
    indices: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    embargoed_set: set[int] = set()
    for item in assignments:
        index = item.index
        if regions["val"][0] <= index < regions["val"][1]:
            split: str | None = "train"
        elif regions["test"][0] <= index < regions["test"][1]:
            split = "val"
        elif index >= boundaries["test"]:
            split = "test"
        else:
            split = None
            embargoed_set.add(index)
        if split is not None:
            indices[split].append(index)
        updated.append(
            SplitAssignment(
                index=index,
                init_time=item.init_time,
                target_start=item.target_start,
                target_end=item.target_end,
                split=split,
            )
        )
    return SplitPlan(
        assignments=tuple(updated),
        indices={name: tuple(values) for name, values in indices.items()},
        embargoed=tuple(sorted(embargoed_set)),
        fractions=fractions,
        embargo_frames=embargo,
    )


def assert_no_leakage(plan: SplitPlan) -> None:
    """Raise if any two splits share a target period in time.

    The embargo exists to protect this invariant, so it is verified explicitly
    rather than assumed.
    """
    ranges: dict[str, tuple[datetime, datetime]] = {}
    for item in plan.assignments:
        if item.split is None:
            continue
        if item.split not in ranges:
            ranges[item.split] = (item.target_start, item.target_end)
        else:
            lo, hi = ranges[item.split]
            ranges[item.split] = (min(lo, item.target_start), max(hi, item.target_end))
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a >= b or a not in ranges or b not in ranges:
                continue
            start_a, end_a = ranges[a]
            start_b, end_b = ranges[b]
            if start_a <= end_b and start_b <= end_a:
                raise AssertionError(
                    f"target leakage between {a} [{start_a}, {end_a}] and {b} [{start_b}, {end_b}]"
                )


def describe_split(plan: SplitPlan, *, frame_minutes: int = 30) -> dict:
    """JSON-serialisable split summary with per-split time coverage."""
    payload: dict = {
        "fractions": list(plan.fractions),
        "embargo_frames": plan.embargo_frames,
        "embargo_minutes": plan.embargo_frames * frame_minutes,
        "n_windows": len(plan.assignments),
        "counts": plan.coverage(),
    }
    for name in SPLIT_NAMES:
        members = [a for a in plan.assignments if a.split == name]
        if not members:
            payload[name] = {
                "n": 0,
                "init_start": None,
                "init_end": None,
                "target_start": None,
                "target_end": None,
            }
            continue
        payload[name] = {
            "n": len(members),
            "init_start": min(m.init_time for m in members).isoformat(),
            "init_end": max(m.init_time for m in members).isoformat(),
            "target_start": min(m.target_start for m in members).isoformat(),
            "target_end": max(m.target_end for m in members).isoformat(),
        }
    return payload
