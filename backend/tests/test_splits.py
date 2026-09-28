"""Tests for leakage-safe chronological splits and the windowed dataset."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

pytest.importorskip("torch")

from app.grid import GridSpec
from app.ingestion.base import ObservationCube, Provenance, WindowSpec
from app.training.dataset import (
    NowcastWindowDataset,
    build_dataloaders,
    build_datasets,
    collate_samples,
    split_summary,
)
from app.training.splits import assign_splits, assert_no_leakage, describe_split

#: Long enough that 70/15/15 fractions yield a val and test split that can each
#: absorb the default embargo (``horizon`` frames) without being emptied.
SEQ_LEN, HORIZON, N_FRAMES = 6, 6, 90
FRAME_MINUTES = 30


def _times(n_frames: int = N_FRAMES) -> list[dt.datetime]:
    start = dt.datetime(2024, 7, 15, 3, 0, tzinfo=dt.timezone.utc)
    return [start + dt.timedelta(minutes=FRAME_MINUTES * i) for i in range(n_frames)]


def _cube(
    shape: tuple[int, int] = (8, 8),
    n_frames: int = N_FRAMES,
    seed: int = 3,
    bbox: tuple[float, float, float, float] = (78.0, 30.0, 79.0, 31.0),
) -> ObservationCube:
    """Small cube with the label contract the dataset requires."""
    rng = np.random.RandomState(seed)
    grid = GridSpec.from_bbox(bbox, shape=shape)
    times = _times(n_frames)
    channels = rng.uniform(0.0, 1.0, (n_frames, 12, *shape)).astype(np.float32) * 100.0
    labels = {
        "thunderstorm": (rng.rand(n_frames, *shape) > 0.8).astype(np.float32),
        "rain_class": rng.randint(0, 4, (n_frames, *shape)).astype(np.int8),
        "cloudburst": (rng.rand(n_frames, *shape) > 0.95).astype(np.float32),
        "flood": (rng.rand(n_frames, *shape) > 0.9).astype(np.float32),
        "flood_soft": rng.rand(n_frames, *shape).astype(np.float32),
    }
    provenance = [
        Provenance(
            source="SYNTHETIC (unit test)",
            product="test-cube",
            valid_from=times[0],
            is_synthetic=True,
            attribution="synthetic test fixture",
        )
    ]
    return ObservationCube(grid=grid, times=times, channels=channels, labels=labels, provenance=provenance)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_split_is_chronological_and_exhaustive() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    plan = assign_splits(_times(), spec)

    counts = plan.coverage()
    assert counts["train"] > counts["val"] >= 0
    assert counts["val"] > 0 and counts["test"] > 0
    # Every window is accounted for exactly once.
    assert counts["train"] + counts["val"] + counts["test"] + counts["embargoed"] == len(plan.assignments)


def test_split_boundaries_are_time_ordered() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    plan = assign_splits(_times(), spec)
    assert_no_leakage(plan)

    def last_init(name: str) -> dt.datetime:
        return max(a.init_time for a in plan.assignments if a.split == name)

    def first_init(name: str) -> dt.datetime:
        return min(a.init_time for a in plan.assignments if a.split == name)

    assert last_init("train") < first_init("val")
    assert last_init("val") < first_init("test")


def test_embargo_prevents_target_overlap_between_splits() -> None:
    """The core leakage guarantee: no shared target frame across splits."""
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    plan = assign_splits(_times(), spec)
    assert_no_leakage(plan)

    def span(name: str) -> tuple:
        members = [a for a in plan.assignments if a.split == name]
        return min(m.target_start for m in members), max(m.target_end for m in members)

    # Strict separation: the last train target precedes the first val target.
    assert span("train")[1] < span("val")[0]
    assert span("val")[1] < span("test")[0]
    assert plan.embargo_frames == HORIZON
    assert len(plan.embargoed) > 0


def test_embargo_shorter_than_the_horizon_is_rejected() -> None:
    """A too-narrow embargo would leak, so it must fail instead of splitting."""
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    with pytest.raises(ValueError, match="smaller than horizon"):
        assign_splits(_times(), spec, embargo_frames=HORIZON - 1)
    # The default (== horizon) is accepted.
    assert assign_splits(_times(), spec).embargo_frames == HORIZON
    # A wider embargo is also allowed.
    assert assign_splits(_times(), spec, embargo_frames=HORIZON + 4).embargo_frames == HORIZON + 4


def test_split_rejects_invalid_fractions() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    with pytest.raises(ValueError, match="three non-negative"):
        assign_splits(_times(), spec, fractions=(0.5, 0.5))
    with pytest.raises(ValueError, match="at least one split fraction"):
        assign_splits(_times(), spec, fractions=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="embargo_frames must be >= 0"):
        assign_splits(_times(), spec, embargo_frames=-1)


def test_split_handles_too_short_a_sequence() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    plan = assign_splits(_times(8), spec)  # 8 < 12 frames needed

    assert len(plan.assignments) == 0
    assert plan.coverage() == {"train": 0, "val": 0, "test": 0, "embargoed": 0}
    summary = describe_split(plan)
    assert summary["n_windows"] == 0
    assert summary["embargo_frames"] == HORIZON


def test_describe_split_reports_time_coverage() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    payload = describe_split(assign_splits(_times(), spec))

    assert payload["embargo_minutes"] == HORIZON * FRAME_MINUTES
    assert payload["train"]["n"] > 0
    assert payload["train"]["init_start"] < payload["train"]["init_end"]


def test_dataset_sample_shapes_and_dtypes() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    dataset = NowcastWindowDataset([_cube()], spec, terrain=np.zeros((4, 8, 8), dtype=np.float32))

    assert len(dataset) == spec.n_windows(N_FRAMES)
    sample = dataset[0]
    assert tuple(sample.x.shape) == (SEQ_LEN, 12, 8, 8)      # (T_in, C, H, W)
    assert tuple(sample.terrain.shape) == (4, 8, 8)
    for name in ("thunderstorm", "cloudburst", "flood", "flood_soft"):
        assert tuple(getattr(sample, name).shape) == (HORIZON, 8, 8)
        assert getattr(sample, name).dtype.is_floating_point
    assert tuple(sample.rain_class.shape) == (HORIZON, 8, 8)
    assert not sample.rain_class.dtype.is_floating_point  # .long() -> int64
    # Normalised inputs must be inside [0, 1].
    assert float(sample.x.min()) >= 0.0 and float(sample.x.max()) <= 1.0


def test_dataset_targets_match_the_cube_labels() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    cube = _cube()
    sample = NowcastWindowDataset([cube], spec)[0]
    future = spec.future_slice(0)

    assert np.allclose(sample.thunderstorm.numpy(), cube.labels["thunderstorm"][future])
    assert np.array_equal(sample.rain_class.numpy(), cube.labels["rain_class"][future].astype(np.int64))
    assert np.allclose(sample.flood.numpy(), cube.labels["flood"][future])


def test_dataset_target_values_are_valid() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    dataset = NowcastWindowDataset([_cube()], spec)
    for position in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[position]
        for name in ("thunderstorm", "cloudburst", "flood"):
            values = getattr(sample, name).numpy()
            assert set(np.unique(values)).issubset({0.0, 1.0}), name
        assert 0.0 <= float(sample.flood_soft.min()) <= float(sample.flood_soft.max()) <= 1.0
        assert 0 <= int(sample.rain_class.min())
        assert int(sample.rain_class.max()) <= 3


def test_dataset_preserves_timestamps_and_provenance() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    cube = _cube()
    sample = NowcastWindowDataset([cube], spec, event_ids=["evt-test"])[0]

    # init_time is the last *input* frame; targets are the following frames.
    assert sample.init_time == cube.times[SEQ_LEN - 1].isoformat()
    assert sample.target_times[0] == cube.times[SEQ_LEN].isoformat()
    assert len(sample.target_times) == HORIZON
    assert sample.provenance["event_id"] == "evt-test"
    assert sample.provenance["is_synthetic"] is True
    assert sample.provenance["frame_minutes"] == FRAME_MINUTES
    assert sample.grid["nx"] == 8 and sample.grid["crs"] == "EPSG:4326"


def test_dataset_provenance_summary_declares_synthetic_status() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    summary = NowcastWindowDataset([_cube()], spec).provenance_summary()

    assert summary["is_synthetic"] is True
    assert "SYNTHETIC DEMO" in summary["validation_status"]
    assert "not independent forecasting skill" in summary["validation_status"]
    assert summary["seq_len"] == SEQ_LEN and summary["horizon"] == HORIZON
    assert len(summary["channels"]) == 12
    assert summary["targets"]["rain_class"]["task"] == "categorical"
    assert summary["rain_classes"] == ["no_rain", "light", "heavy", "extreme"]


def test_dataset_rejects_mismatched_grids_and_missing_labels() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    with pytest.raises(ValueError, match="identical grid"):
        NowcastWindowDataset(
            [_cube(), _cube(seed=1, bbox=(77.0, 29.0, 78.0, 30.0))], spec
        )

    broken = _cube()
    broken.labels.pop("flood")
    with pytest.raises(ValueError, match="missing required label"):
        NowcastWindowDataset([broken], spec)


def test_dataset_rejects_bad_window_indices_and_terrain() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    dataset = NowcastWindowDataset([_cube()], spec, window_indices=[(0, 0), (0, 2)])
    assert len(dataset) == 2
    with pytest.raises(IndexError, match="window"):
        NowcastWindowDataset([_cube()], spec, window_indices=[(0, 999)])
    with pytest.raises(IndexError, match="cube index"):
        NowcastWindowDataset([_cube()], spec, window_indices=[(3, 0)])
    with pytest.raises(ValueError, match="terrain tensor shape"):
        NowcastWindowDataset([_cube()], spec, terrain=np.zeros((3, 8, 8), dtype=np.float32))


# --------------------------------------------------------------------------- #
# build_datasets / loaders
# --------------------------------------------------------------------------- #
def test_build_datasets_partitions_every_window() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    cubes = [_cube(seed=1), _cube(seed=2)]
    total_windows = spec.n_windows(N_FRAMES) * len(cubes)
    datasets, plans = build_datasets(
        cubes, spec, terrain=np.zeros((4, 8, 8), dtype=np.float32), event_ids=["a", "b"]
    )

    # Each cube keeps its own plan, so the embargo total is the sum over cubes.
    assigned = sum(len(datasets[name]) for name in ("train", "val", "test"))
    embargoed = total_windows - assigned
    assert embargoed > 0
    assert assigned + embargoed == total_windows
    pairs = [
        set(datasets[name].window_indices)
        for name in ("train", "val", "test")
        if len(datasets[name])
    ]
    for i, first in enumerate(pairs):
        for second in pairs[i + 1 :]:
            assert not (first & second)


def test_build_dataloaders_yields_batched_tensors() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    datasets, _plans = build_datasets(
        [_cube(seed=1), _cube(seed=2)], spec, terrain=np.zeros((4, 8, 8), dtype=np.float32)
    )
    loaders = build_dataloaders(datasets, batch_size=2, seed=1)
    assert "train" in loaders

    batch = next(iter(loaders["train"]))
    batch_size = min(2, len(datasets["train"]))
    assert tuple(batch["x"].shape) == (batch_size, SEQ_LEN, 12, 8, 8)
    assert tuple(batch["terrain"].shape) == (batch_size, 4, 8, 8)
    assert tuple(batch["rain_class"].shape) == (batch_size, HORIZON, 8, 8)
    assert len(batch["meta"]) == batch_size
    assert batch["meta"][0]["init_time"]


def test_collate_keeps_metadata_per_sample() -> None:
    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    dataset = NowcastWindowDataset([_cube()], spec, event_ids=["evt-x"])
    batch = collate_samples([dataset[0], dataset[1]])

    assert tuple(batch["x"].shape) == (2, SEQ_LEN, 12, 8, 8)
    assert batch["terrain"] is None
    assert batch["meta"][0]["provenance"]["event_id"] == "evt-x"
    assert batch["meta"][1]["init_time"] != batch["meta"][0]["init_time"]


def test_split_summary_is_serialisable() -> None:
    import json

    spec = WindowSpec(seq_len=SEQ_LEN, horizon=HORIZON)
    datasets, plans = build_datasets([_cube()], spec)
    payload = split_summary(datasets, plans)

    assert set(payload) == {"train", "val", "test"}
    assert json.loads(json.dumps(payload, default=str))["train"]["n_windows"] == payload["train"]["n_windows"]
