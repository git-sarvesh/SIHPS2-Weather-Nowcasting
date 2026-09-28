"""Windowed dataset and leakage-safe DataLoader construction.

The dataset is a thin, contract-preserving adapter over
:class:`~app.ingestion.base.ObservationCube`: it does not invent features, it
reuses ``cube.window()`` (which normalises channels with the physical bounds in
:mod:`app.physics`) and the multi-task labels the ingestion layer already
derives from the same physics that drives the imagery.

Each sample carries, alongside the tensors, the grid, the frame timestamps, the
channel/target units and the cube provenance, so a training run is auditable and
a synthetic dataset is always identifiable as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.grid import FRAME_MINUTES
from app.ingestion.base import EXPERIMENTAL_DISCLAIMER, ObservationCube, WindowSpec
from app.physics import CHANNELS, RAIN_CLASSES
from app.training.splits import SPLIT_NAMES, SplitPlan, assert_no_leakage, assign_splits

__all__ = [
    "REQUIRED_LABELS",
    "TARGET_SPECS",
    "NowcastWindowDataset",
    "TrainingSample",
    "build_dataloaders",
    "build_datasets",
    "collate_samples",
    "dataset_provenance",
    "split_summary",
]

#: Label contract of every sample: sample key -> (cube label key, unit, task kind).
TARGET_SPECS: dict[str, tuple[str, str, str]] = {
    "thunderstorm": ("thunderstorm", "binary probability", "binary"),
    "rain_class": ("rain_class", "class index 0-3", "categorical"),
    "cloudburst": ("cloudburst", "binary probability", "binary"),
    "flood": ("flood", "binary probability", "binary"),
    "flood_soft": ("flood_soft", "probability in [0, 1]", "regression"),
}

#: Label keys a cube must expose to be trainable.
REQUIRED_LABELS: tuple[str, ...] = tuple(key for key, *_ in TARGET_SPECS.values())


@dataclass(slots=True)
class TrainingSample:
    """One supervised window: model input, terrain, targets and provenance."""

    x: torch.Tensor                    # (T_in, C, H, W) normalised inputs
    terrain: torch.Tensor | None       # (4, H, W) terrain tensor
    thunderstorm: torch.Tensor         # (T_out, H, W) binary
    rain_class: torch.Tensor           # (T_out, H, W) int64
    cloudburst: torch.Tensor           # (T_out, H, W) binary
    flood: torch.Tensor                # (T_out, H, W) binary
    flood_soft: torch.Tensor           # (T_out, H, W) soft flood risk
    grid: dict[str, Any]
    init_time: str
    target_times: list[str]
    provenance: dict[str, Any]

    def as_batch(self) -> dict[str, Any]:
        """Add the batch axis, matching the network's ``(B, T, C, H, W)`` contract."""
        return {
            "x": self.x.unsqueeze(0),
            "terrain": None if self.terrain is None else self.terrain.unsqueeze(0),
            "thunderstorm": self.thunderstorm.unsqueeze(0),
            "rain_class": self.rain_class.unsqueeze(0),
            "cloudburst": self.cloudburst.unsqueeze(0),
            "flood": self.flood.unsqueeze(0),
            "flood_soft": self.flood_soft.unsqueeze(0),
            "meta": {
                "init_time": self.init_time,
                "grid": self.grid,
                "target_times": self.target_times,
            },
        }


class NowcastWindowDataset(Dataset):
    """Sliding-window samples from one or more :class:`ObservationCube` objects.

    Parameters
    ----------
    cubes:
        Sequences sharing a *single* :class:`~app.grid.GridSpec`.
    spec:
        Window geometry. The default is 6 history frames and 6 target frames.
    window_indices:
        ``(cube_index, window_index)`` pairs, as produced by
        :func:`~app.training.splits.assign_splits`. When ``None``, every window
        of every cube is used.
    terrain:
        ``(4, H, W)`` terrain tensor shared by the cubes, or ``None`` to run the
        flood head without terrain late fusion.
    is_synthetic:
        Recorded on every sample so downstream reports can label the dataset.
    """

    def __init__(
        self,
        cubes: Sequence[ObservationCube],
        spec: WindowSpec | None = None,
        *,
        window_indices: Sequence[tuple[int, int]] | None = None,
        terrain: np.ndarray | None = None,
        is_synthetic: bool = True,
        event_ids: Sequence[str] | None = None,
    ) -> None:
        if not cubes:
            raise ValueError("at least one ObservationCube is required")
        self.spec = spec or WindowSpec(seq_len=6, horizon=6)
        reference = cubes[0].grid
        reference_dict = reference.to_dict()
        for cube in cubes[1:]:
            if cube.grid.to_dict() != reference_dict:
                raise ValueError("all cubes must share an identical grid")
        missing = [key for key in REQUIRED_LABELS if key not in cubes[0].labels]
        if missing:
            raise ValueError(f"cubes are missing required label arrays: {missing}")

        self.cubes = list(cubes)
        self.event_ids = (
            list(event_ids) if event_ids else [f"cube-{i}" for i in range(len(self.cubes))]
        )
        if len(self.event_ids) != len(self.cubes):
            raise ValueError("event_ids must have one entry per cube")
        self.terrain = None
        if terrain is not None:
            tensor = np.asarray(terrain, dtype=np.float32)
            if tensor.shape != (4, *reference.shape):
                raise ValueError(f"terrain tensor shape {tensor.shape} != (4, {reference.shape})")
            self.terrain = torch.from_numpy(np.ascontiguousarray(tensor))
        self.is_synthetic = bool(is_synthetic)
        self.grid_dict = reference_dict
        self.channels = [
            {"name": s.name, "unit": s.unit, "vmin": s.vmin, "vmax": s.vmax} for s in CHANNELS
        ]
        self.rain_classes = list(RAIN_CLASSES)
        self.frame_minutes = FRAME_MINUTES
        self.window_indices = self._resolve_indices(window_indices)

    def _resolve_indices(
        self, window_indices: Sequence[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        """Resolve to ``(cube_index, window_index)`` pairs, validating the range."""
        if window_indices is None:
            pairs: list[tuple[int, int]] = []
            for cube_index, cube in enumerate(self.cubes):
                pairs.extend(
                    (cube_index, w) for w in range(self.spec.n_windows(cube.n_frames))
                )
            return pairs
        pairs = []
        for cube_index, window_index in window_indices:
            if not 0 <= cube_index < len(self.cubes):
                raise IndexError(f"cube index {cube_index} out of range")
            available = self.spec.n_windows(self.cubes[cube_index].n_frames)
            if not 0 <= window_index < available:
                raise IndexError(
                    f"window {window_index} out of range for cube {cube_index} "
                    f"({available} available)"
                )
            pairs.append((cube_index, window_index))
        return pairs

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, position: int) -> TrainingSample:
        cube_index, window_index = self.window_indices[position]
        cube = self.cubes[cube_index]
        x_norm, labels, _x_phys = cube.window(self.spec, window_index)
        history = self.spec.history_slice(window_index)
        future = self.spec.future_slice(window_index)
        first = cube.provenance[0] if cube.provenance else None
        return TrainingSample(
            x=torch.from_numpy(np.ascontiguousarray(x_norm.transpose(1, 0, 2, 3))),
            terrain=self.terrain,
            thunderstorm=torch.from_numpy(np.ascontiguousarray(labels["thunderstorm"])).float(),
            rain_class=torch.from_numpy(np.ascontiguousarray(labels["rain_class"])).long(),
            cloudburst=torch.from_numpy(np.ascontiguousarray(labels["cloudburst"])).float(),
            flood=torch.from_numpy(np.ascontiguousarray(labels["flood"])).float(),
            flood_soft=torch.from_numpy(np.ascontiguousarray(labels["flood_soft"])).float(),
            grid=self.grid_dict,
            init_time=cube.times[history.stop - 1].isoformat(),
            target_times=[t.isoformat() for t in cube.times[future]],
            provenance={
                "event_id": self.event_ids[cube_index],
                "is_synthetic": self.is_synthetic,
                "source": first.source if first else "unknown",
                "product": first.product if first else "unknown",
                "attribution": first.attribution if first else "",
                "disclaimer": EXPERIMENTAL_DISCLAIMER,
                "window_index": window_index,
                "frame_minutes": FRAME_MINUTES,
            },
        )

    # -------------------------------------------------------------- reporting
    def provenance_summary(self) -> dict[str, Any]:
        """Dataset-level provenance and label/unit contract (stored in checkpoints)."""
        return {
            **dataset_provenance(self.cubes, self.is_synthetic),
            "n_samples": len(self),
            "seq_len": self.spec.seq_len,
            "horizon": self.spec.horizon,
            "stride": self.spec.stride,
            "frame_minutes": FRAME_MINUTES,
            "grid": self.grid_dict,
            "channels": self.channels,
            "targets": {
                name: {"cube_label": key, "unit": unit, "task": task}
                for name, (key, unit, task) in TARGET_SPECS.items()
            },
            "rain_classes": self.rain_classes,
            "terrain_fused": self.terrain is not None,
            "disclaimer": EXPERIMENTAL_DISCLAIMER,
        }

    def label_statistics(self, *, max_samples: int = 8) -> dict[str, float]:
        """Positive fractions over a few samples (used to weight the losses)."""
        keys = ("thunderstorm", "cloudburst", "flood", "rain_class_extreme")
        n = min(max(max_samples, 0), len(self))
        if n == 0:
            return {key: 0.0 for key in keys}
        totals = {key: 0.0 for key in keys}
        for position in range(n):
            sample = self[position]
            totals["thunderstorm"] += float(sample.thunderstorm.mean())
            totals["cloudburst"] += float(sample.cloudburst.mean())
            totals["flood"] += float(sample.flood.mean())
            totals["rain_class_extreme"] += float((sample.rain_class == 3).float().mean())
        return {key: round(value / n, 6) for key, value in totals.items()}


def dataset_provenance(cubes: Sequence[ObservationCube], is_synthetic: bool) -> dict[str, Any]:
    """Aggregate provenance across the cubes making up a dataset."""
    sources = sorted({p.source for cube in cubes for p in cube.provenance})
    return {
        "is_synthetic": bool(is_synthetic),
        "sources": sources,
        "n_cubes": len(cubes),
        "disclaimer": EXPERIMENTAL_DISCLAIMER,
        "validation_status": (
            "SYNTHETIC DEMO - metrics describe agreement with the SIHPS synthetic "
            "generator, not independent forecasting skill."
            if is_synthetic
            else "Operational observations - independent validation is still required "
            "before any skill claim."
        ),
    }

def build_datasets(
    cubes: Sequence[ObservationCube],
    spec: WindowSpec,
    *,
    terrain: np.ndarray | None = None,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    embargo_frames: int | None = None,
    is_synthetic: bool = True,
    event_ids: Sequence[str] | None = None,
) -> tuple[dict[str, NowcastWindowDataset], dict[str, SplitPlan]]:
    """Split every cube chronologically, then build one dataset per split.

    Each event is split *independently in time* and the per-cube window indices
    are merged, so a given init time never appears in two splits. Synthetic
    events are separate multi-hour sequences, so windows of different events
    cannot share target frames either.

    Returns
    -------
    (datasets, plans)
        ``datasets`` maps ``"train" | "val" | "test"`` to a dataset; ``plans``
        maps the same keys to a representative per-cube split plan.
    """
    plans: dict[str, SplitPlan] = {}
    pairs: dict[str, list[tuple[int, int]]] = {name: [] for name in SPLIT_NAMES}
    for cube_index, cube in enumerate(cubes):
        for name in SPLIT_NAMES:
            plan = assign_splits(cube.times, spec, fractions=fractions, embargo_frames=embargo_frames)
            assert_no_leakage(plan)
            plans.setdefault(name, plan)
            pairs[name].extend((cube_index, w) for w in plan.indices_for(name))
    datasets = {
        name: NowcastWindowDataset(
            cubes,
            spec,
            window_indices=pairs[name],
            terrain=terrain,
            is_synthetic=is_synthetic,
            event_ids=event_ids,
        )
        for name in SPLIT_NAMES
    }
    return datasets, plans


def build_dataloaders(
    datasets: dict[str, NowcastWindowDataset],
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """DataLoaders with a deterministic, seed-driven shuffling order."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loaders: dict[str, DataLoader] = {}
    for name in SPLIT_NAMES:
        dataset = datasets.get(name)
        if dataset is None or len(dataset) == 0:
            continue
        loaders[name] = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=(name == "train"),
            num_workers=max(0, int(num_workers)),
            drop_last=False,
            collate_fn=collate_samples,
            generator=generator if name == "train" else None,
        )
    return loaders


def collate_samples(samples: Sequence[TrainingSample]) -> dict[str, Any]:
    """Collate :class:`TrainingSample` objects into batched tensors.

    The ``meta`` entry keeps the per-sample grid/timestamp provenance so a
    trained prediction can be traced back to its issue time and location.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    stacked = {
        name: torch.stack([getattr(sample, name) for sample in samples])
        for name in ("x", "thunderstorm", "rain_class", "cloudburst", "flood", "flood_soft")
    }
    terrain = samples[0].terrain
    if terrain is not None:
        stacked["terrain"] = torch.stack([sample.terrain for sample in samples if sample.terrain is not None])
    else:
        stacked["terrain"] = None
    stacked["meta"] = [
        {"init_time": s.init_time, "target_times": s.target_times, "provenance": s.provenance}
        for s in samples
    ]
    return stacked


def split_summary(
    datasets: dict[str, NowcastWindowDataset], plans: dict[str, SplitPlan]
) -> dict[str, Any]:
    """JSON-serialisable description of the split for the training report."""
    return {
        name: {
            "n_windows": len(datasets[name]) if name in datasets else 0,
            "label_statistics": (
                datasets[name].label_statistics() if name in datasets and len(datasets[name]) else {}
            ),
            "embargo_frames": plans[name].embargo_frames if name in plans else None,
            "fractions": list(plans[name].fractions) if name in plans else None,
        }
        for name in SPLIT_NAMES
    }
