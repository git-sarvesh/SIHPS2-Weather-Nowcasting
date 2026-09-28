"""Shared ingestion data structures and connector base class.

The contract between ingestion and modelling is :class:`ObservationCube`: a
``(T, C, H, W)`` array in *physical units* on the common :class:`~app.grid.GridSpec`,
plus per-frame quality masks (0 = missing, 1 = good) and provenance records.
Normalisation to ``[0, 1]`` happens lazily in :meth:`ObservationCube.normalised`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import FRAME_MINUTES, GridSpec
from app.logging_conf import get_logger
from app.physics import CHANNELS, N_CHANNELS, channel_index, normalise_channels

logger = get_logger("ingestion.base")

#: Mandatory disclaimer attached to every artefact leaving the system.
EXPERIMENTAL_DISCLAIMER = (
    "Experimental AI prediction for research and preparedness support only. "
    "This is NOT an official India Meteorological Department (IMD) warning. "
    "Always refer to IMD for authoritative alerts."
)


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(tz=timezone.utc)


def parse_time(value: str | datetime) -> datetime:
    """Parse an ISO-8601 string (or pass an aware datetime through) as UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file that contributed data, with its digest and size.

    Phase 8.4. A cube is often built from several files (IMDAA per-layer
    downloads, a DEM, a station export). Recording only a primary file would
    leave the others unverifiable, so every contributing file is listed here and
    verified individually.

    ``sha256`` is ``None`` when the bytes were never available. A missing digest
    is *unknown*, which every observational check treats as a failure - it is
    never reported as verified and never invented.
    """

    #: Path of the file, used to re-hash and to re-verify later.
    path: str
    #: SHA-256 of the file's bytes, or ``None`` when unknown.
    sha256: str | None = None
    #: Size in bytes, or ``None`` when unknown.
    size_bytes: int | None = None
    #: Organisation that provided this file.
    provider: str = ""
    #: What this file contributed, e.g. ``"insat_l1b"``, ``"dem"``, ``"imdaa_pressure"``.
    role: str = ""
    #: Free-form identity of the file at the source (e.g. an RDS job id).
    source_identity: str = ""
    #: True only when the bytes still exist and can be re-hashed.
    bytes_available: bool = False

    @property
    def has_verifiable_hash(self) -> bool:
        return bool(self.sha256) and bool(self.bytes_available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "provider": self.provider,
            "role": self.role,
            "source_identity": self.source_identity,
            "bytes_available": self.bytes_available,
            "has_verifiable_hash": self.has_verifiable_hash,
        }

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> SourceFile:
        """Rebuild from :meth:`to_dict`, ignoring unknown keys."""
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in item.items() if k in fields})


def source_file_from_disk(
    path: str | Path, *, provider: str = "", role: str = "", source_identity: str = ""
) -> SourceFile:
    """Build a :class:`SourceFile` by hashing a file that exists on disk.

    A file that cannot be read yields a record with ``sha256=None`` rather than
    a fabricated digest, so the gap is explicit and fails closed later.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    digest: str | None = None
    size: int | None = None
    try:
        from app.ingestion.realtime.provenance import file_sha256

        digest = file_sha256(p)
        size = p.stat().st_size
    except (OSError, ValueError):
        digest, size = None, None
    return SourceFile(
        path=str(p),
        sha256=digest,
        size_bytes=size,
        provider=provider,
        role=role,
        source_identity=source_identity,
        bytes_available=digest is not None,
    )


@dataclass(slots=True)
class Provenance:
    """Where a piece of data came from (auditability, Part 10).

    Phase 8.3 adds cryptographically verifiable fields. Every one is optional
    with a safe default, so existing constructors and stored records keep
    working unchanged.

    The hash rule is strict: :attr:`source_sha256` is the digest of the
    **original source bytes** and is ``None`` whenever those bytes are not
    available. A hash is never synthesised, inferred or copied from a filename.
    ``None`` means *unknown*, and every observational consumer treats unknown as
    a failure - see :func:`app.ingestion.realtime.provenance.verify_source_hash`.
    """

    source: str
    product: str
    valid_from: datetime
    valid_to: datetime | None = None
    fetched_at: datetime = field(default_factory=utcnow)
    path: str | None = None
    is_synthetic: bool = True
    attribution: str = ""
    # ------------------------------------------------------- phase 8.3
    #: SHA-256 hex digest of the original source file, or ``None`` if unknown.
    source_sha256: str | None = None
    #: The file ``source_sha256`` was computed from. Re-hashing this path must
    #: reproduce the digest; the two are never allowed to drift apart.
    source_path: str | None = None
    #: True only when the original bytes still exist and can be re-hashed.
    source_bytes_available: bool = False
    #: Organisation that provided the data (e.g. ``"NCMRWF"``, ``"IMD"``).
    provider: str = ""
    #: When this system acquired the data, distinct from when it was fetched.
    acquired_at: datetime | None = None
    #: ``channel name -> physical unit`` for the fields this record supplies.
    channel_units: dict[str, str] = field(default_factory=dict)
    #: Explicit spatial coverage, e.g. ``"77.5-80.5E 29.0-31.5N"``.
    coverage: str | None = None
    #: Every file that contributed, verified individually (Phase 8.4).
    source_files: list[SourceFile] = field(default_factory=list)

    @property
    def has_verifiable_hash(self) -> bool:
        """Whether a real digest of real bytes is on record.

        True when at least one source file carries one. An empty manifest is
        *unknown*, not verified.
        """
        if self.source_files:
            return any(f.has_verifiable_hash for f in self.source_files)
        return bool(self.source_sha256) and bool(self.source_bytes_available)

    def to_dict(self) -> dict[str, Any]:
        # Only dataclass fields are emitted, so ``ObservationCube.load_npz`` can
        # rebuild a Provenance from this mapping. Computed properties such as
        # ``has_verifiable_hash`` are deliberately excluded.
        return {
            "source": self.source,
            "product": self.product,
            "provider": self.provider,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
            "fetched_at": self.fetched_at.isoformat(),
            "path": self.path,
            "is_synthetic": self.is_synthetic,
            "attribution": self.attribution,
            "source_sha256": self.source_sha256,
            "source_path": self.source_path,
            "source_bytes_available": self.source_bytes_available,
            "channel_units": dict(self.channel_units),
            "coverage": self.coverage,
            "source_files": [f.to_dict() for f in self.source_files],
        }


@dataclass(slots=True)
class TerrainStack:
    """DEM-derived terrain layers on the model grid (Part 1.3)."""

    grid: GridSpec
    elevation_m: np.ndarray          # (H, W)
    slope_deg: np.ndarray            # (H, W)
    aspect_deg: np.ndarray           # (H, W)
    flow_accumulation: np.ndarray    # (H, W) upstream cell count
    drainage_basin: np.ndarray       # (H, W) integer basin id
    twi: np.ndarray                  # (H, W) topographic wetness index
    land_use: np.ndarray             # (H, W) integer land-cover class
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        expected = self.grid.shape
        for name in (
            "elevation_m",
            "slope_deg",
            "aspect_deg",
            "flow_accumulation",
            "drainage_basin",
            "twi",
            "land_use",
        ):
            arr = np.asarray(getattr(self, name))
            setattr(self, name, arr)
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} != grid shape {expected}")

    @property
    def normalised_flow(self) -> np.ndarray:
        """Log-scaled flow accumulation in ``[0, 1]``."""
        fa = np.log1p(np.clip(self.flow_accumulation, 0, None))
        hi = float(np.percentile(fa, 99.5)) or 1.0
        return np.clip(fa / hi, 0.0, 1.0)

    def to_feature_stack(self) -> np.ndarray:
        """``(4, H, W)`` terrain tensor for the flood head's late fusion path."""
        return np.stack(
            [
                np.clip(self.elevation_m / 8000.0, 0.0, 1.0),
                np.clip(self.slope_deg / 45.0, 0.0, 1.0),
                self.normalised_flow,
                np.clip(self.land_use / 20.0, 0.0, 1.0),
            ]
        ).astype(np.float32)

    def statistics(self) -> dict[str, float]:
        return {
            "elevation_m_min": float(self.elevation_m.min()),
            "elevation_m_max": float(self.elevation_m.max()),
            "elevation_m_mean": float(self.elevation_m.mean()),
            "slope_deg_mean": float(self.slope_deg.mean()),
            "slope_deg_p95": float(np.percentile(self.slope_deg, 95)),
            "n_basins": int(np.unique(self.drainage_basin).size),
            "twi_p95": float(np.percentile(self.twi, 95)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "statistics": self.statistics(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


def _provenance_from_dict(item: dict[str, Any]) -> Provenance:
    """Rebuild a :class:`Provenance` from :meth:`Provenance.to_dict` output.

    Forward compatible: keys a Phase 8.2 payload does not carry (the Phase 8.3
    integrity fields) take their defaults rather than raising, and any unknown
    key is dropped instead of failing the load. Integrity fields default to
    "unknown", which every verifier treats as a failure - never as a pass.
    """
    fields = {f.name for f in dataclasses.fields(Provenance)}
    kwargs: dict[str, Any] = {k: v for k, v in item.items() if k in fields}
    for key in ("valid_from", "valid_to", "fetched_at", "acquired_at"):
        value = kwargs.get(key)
        if isinstance(value, str):
            kwargs[key] = parse_time(value) if value else None
    # Phase 8.4: rebuild the manifest entries, not raw dicts, so a reloaded
    # record can be verified exactly like the original.
    manifest = kwargs.get("source_files") or []
    kwargs["source_files"] = [
        SourceFile.from_dict(e) if isinstance(e, dict) else e for e in manifest
    ]
    return Provenance(**kwargs)


@dataclass(slots=True)
class ObservationCube:
    """Aligned multi-channel observation/nowcast cube on the common grid."""

    grid: GridSpec
    times: list[datetime]
    channels: np.ndarray                    # (T, C, H, W) physical units
    quality: np.ndarray | None = None       # (T, H, W) in [0, 1]
    provenance: list[Provenance] = field(default_factory=list)
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.channels, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"channels must be (T, C, H, W), got {arr.shape}")
        if arr.shape[1] != N_CHANNELS:
            raise ValueError(f"expected {N_CHANNELS} channels, got {arr.shape[1]}")
        if arr.shape[2:] != self.grid.shape:
            raise ValueError(f"channel raster {arr.shape[2:]} != grid {self.grid.shape}")
        if len(self.times) != arr.shape[0]:
            raise ValueError("len(times) must equal the time dimension")
        self.channels = arr
        if self.quality is None:
            self.quality = np.ones((arr.shape[0], *self.grid.shape), dtype=np.float32)

    # ------------------------------------------------------------- accessors
    @property
    def n_frames(self) -> int:
        return int(self.channels.shape[0])

    def channel(self, name: str) -> np.ndarray:
        """Physical-unit ``(T, H, W)`` array for a named channel."""
        return self.channels[:, channel_index(name), :, :]

    def normalised(self) -> np.ndarray:
        """``(T, C, H, W)`` cube scaled to ``[0, 1]`` using physical bounds."""
        return normalise_channels(self.channels)

    def frame(self, index: int) -> ObservationCube:
        """A single-frame cube (used by snapshot endpoints)."""
        return ObservationCube(
            grid=self.grid,
            times=[self.times[index]],
            channels=self.channels[index : index + 1],
            quality=self.quality[index : index + 1] if self.quality is not None else None,
            provenance=self.provenance,
            labels={k: v[index : index + 1] for k, v in self.labels.items()},
            metadata=dict(self.metadata),
        )

    def subset(self, start: int, stop: int) -> ObservationCube:
        """Half-open temporal slice ``[start, stop)``."""
        if not (0 <= start < stop <= self.n_frames):
            raise ValueError(f"invalid slice [{start}, {stop}) for {self.n_frames} frames")
        return ObservationCube(
            grid=self.grid,
            times=self.times[start:stop],
            channels=self.channels[start:stop],
            quality=self.quality[start:stop] if self.quality is not None else None,
            provenance=self.provenance,
            labels={k: v[start:stop] for k, v in self.labels.items()},
            metadata=dict(self.metadata),
        )

    def window(self, spec: WindowSpec, index: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """Sliding-window slice for window ``index``.

        Returns
        -------
        x_norm:
            ``(C, T_in, H, W)`` normalised model inputs.
        y:
            Mapping label name -> ``(T_out, H, W)`` targets.
        x_phys:
            The same window in physical units, used by the What-If simulator and
            by the XAI physical-consistency checks.
        """
        history = spec.history_slice(index)
        future = spec.future_slice(index)
        x_phys = np.ascontiguousarray(self.channels[history].transpose(1, 0, 2, 3))
        x_norm = normalise_channels(x_phys.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)
        y = {name: np.asarray(value[future]) for name, value in self.labels.items()}
        return x_norm, y, x_phys

    def label_keys(self) -> list[str]:
        return sorted(self.labels)

    def newest_time(self) -> datetime:
        return max(self.times)

    def provenance_summary(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.provenance]

    # ------------------------------------------------------------------- io
    def save_npz(self, path: str | Path, *, include_labels: bool = True) -> Path:
        """Persist the cube to a compressed ``.npz`` (demo/test interchange)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "channels": self.channels.astype(np.float32),
            "quality": self.quality.astype(np.float32),
            "times": np.array([t.isoformat() for t in self.times]),
            "grid": np.array(json.dumps(self.grid.to_dict())),
            "metadata": np.array(json.dumps(self.metadata, default=str)),
            "provenance": np.array(json.dumps(self.provenance_summary(), default=str)),
        }
        if include_labels:
            for key, value in self.labels.items():
                payload[f"label__{key}"] = np.asarray(value)
        np.savez_compressed(path, **payload)
        logger.debug("saved observation cube", extra={"path": str(path), "frames": self.n_frames})
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> ObservationCube:
        """Load a cube written by :meth:`save_npz`."""
        with np.load(Path(path), allow_pickle=False) as data:
            grid = GridSpec.from_dict(json.loads(str(data["grid"])))
            times = [parse_time(str(t)) for t in data["times"]]
            labels = {k[len("label__") :]: data[k] for k in data.files if k.startswith("label__")}
            provenance = [
                _provenance_from_dict(item)
                for item in json.loads(str(data["provenance"]))
            ]
            quality = data["quality"] if "quality" in data.files else None
            return cls(
                grid=grid,
                times=times,
                channels=data["channels"],
                quality=quality,
                provenance=provenance,
                labels=labels,
                metadata=json.loads(str(data["metadata"])),
            )


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Sliding-window geometry: ``seq_len`` past frames -> ``horizon`` future frames."""

    seq_len: int = 6
    horizon: int = 12
    stride: int = 1
    frame_minutes: int = FRAME_MINUTES

    @property
    def total_frames(self) -> int:
        return self.seq_len + self.horizon

    @property
    def input_duration_h(self) -> float:
        return (self.seq_len - 1) * self.frame_minutes / 60.0

    @property
    def horizon_hours(self) -> list[float]:
        """Lead time [h] of each forecast step, relative to the last input frame."""
        return [step * self.frame_minutes / 60.0 for step in range(1, self.horizon + 1)]

    def n_windows(self, n_frames: int) -> int:
        """Number of usable sliding windows in an ``n_frames`` sequence."""
        if n_frames < self.total_frames:
            return 0
        return (n_frames - self.total_frames) // self.stride + 1

    def history_slice(self, index: int) -> slice:
        """Index slice of the ``seq_len`` input frames for window ``index``."""
        start = index * self.stride
        return slice(start, start + self.seq_len)

    def future_slice(self, index: int) -> slice:
        """Index slice of the ``horizon`` target frames for window ``index``."""
        start = index * self.stride + self.seq_len
        return slice(start, start + self.horizon)

    def split_times(self, times: Sequence[datetime]) -> tuple[list[datetime], list[datetime]]:
        """Split a frame axis into ``(input_times, target_times)``."""
        return list(times[: self.seq_len]), list(times[self.seq_len : self.total_frames])

    def target_times(self, init_time: datetime) -> list[datetime]:
        """Absolute valid times of the forecast steps issued at ``init_time``."""
        return [init_time + timedelta(minutes=self.frame_minutes * (i + 1)) for i in range(self.horizon)]

    def step_for_lead_hours(self, lead_hours: float) -> int:
        """Forecast-step index (0-based) closest to ``lead_hours``."""
        step = int(round(lead_hours * 60.0 / self.frame_minutes)) - 1
        return max(0, min(self.horizon - 1, step))


class DataConnector:
    """Base class for source connectors (satellite, reanalysis, terrain)."""

    #: Human readable source name, e.g. ``"INSAT-3D"``.
    source_name: str = "unknown"
    #: Whether the connector returns synthetic data (demo mode).
    is_synthetic: bool = True

    def __init__(
        self, grid: GridSpec, *, demo_mode: bool = True, cache_dir: str | Path | None = None
    ) -> None:
        self.grid = grid
        self.demo_mode = demo_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(self, name: str) -> Path | None:
        """Return an existing cache path for ``name`` (``None`` when absent)."""
        if not self.cache_dir:
            return None
        path = self.cache_dir / name
        return path if path.exists() else None

    def health(self) -> dict[str, Any]:
        """Connector health/freshness information for ``/api/v1/health``."""
        return {
            "source": self.source_name,
            "synthetic": self.is_synthetic,
            "demo_mode": self.demo_mode,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    def fetch(self, *args, **kwargs):  # pragma: no cover - abstract
        """Retrieve data for a time window; implemented by subclasses."""
        raise NotImplementedError

    def iter_frames(self, cube: ObservationCube) -> Iterator[ObservationCube]:
        """Yield single-frame cubes (streaming convenience)."""
        for index in range(cube.n_frames):
            yield cube.frame(index)


__all__ = [
    "CHANNELS",
    "EXPERIMENTAL_DISCLAIMER",
    "DataConnector",
    "ObservationCube",
    "Provenance",
    "TerrainStack",
    "WindowSpec",
    "parse_time",
    "utcnow",
]
