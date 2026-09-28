"""Provenance records for ingested datasets: checksums, QC, and de-duplication.

Every ingested product gets a :class:`DatasetProvenance` describing where it
came from, what it contains, how it was processed, and its SHA-256 checksum.
Two invariants matter most:

* **No credentials.** :meth:`DatasetProvenance.to_dict` runs every string field
  through a redaction pass, so a secret that accidentally reaches a provenance
  record is masked before it is written.
* **No silent re-ingestion.** :func:`ingest_key` derives a stable identity from
  ``(source, product, acquisition time, checksum)``; the repository's unique
  constraint on that key is what prevents the same product being stored twice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.grid import GridSpec
from app.ingestion.base import utcnow
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.provenance")

__all__ = [
    "DataClass",
    "DatasetProvenance",
    "HashVerification",
    "QualityReport",
    "ResamplingRecord",
    "file_sha256",
    "hash_file_or_none",
    "ingest_key",
    "verify_source_hash",
]


# --------------------------------------------------------------------------- #
# Phase 8.3 - cryptographic source verification
# --------------------------------------------------------------------------- #
HASH_OK = "ok"
HASH_MISMATCH = "mismatch"
HASH_MISSING = "missing"
HASH_UNAVAILABLE = "bytes_unavailable"
HASH_NOT_RECORDED = "not_recorded"
HASH_UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class HashVerification:
    """Outcome of checking one source file against its recorded digest.

    ``status`` is one of the ``HASH_*`` constants. Only ``HASH_OK`` means the
    bytes on disk are provably the bytes that were ingested. Every other value
    is a distinct, reportable failure - in particular ``not_recorded`` (no
    digest was ever stored) is never conflated with ``ok``.
    """

    status: str
    expected: str | None
    actual: str | None
    path: str | None
    detail: str

    @property
    def verified(self) -> bool:
        return self.status == HASH_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "path": self.path,
            "detail": self.detail,
            "verified": self.verified,
        }


def hash_file_or_none(path: str | Path | None) -> str | None:
    """SHA-256 of a file, or ``None`` when the bytes are not readable.

    Used at acquisition time so a missing or unreadable file records *no*
    digest. Inventing one would be indistinguishable from a real one later.
    """
    if path is None:
        return None
    try:
        return file_sha256(path)
    except (OSError, ValueError):
        return None


def verify_source_hash(
    expected_sha256: str | None,
    path: str | Path | None,
    *,
    require_bytes: bool = True,
) -> HashVerification:
    """Check a file's bytes against a recorded digest.

    Fails closed. A missing digest, a missing file, or a mismatch are all
    failures, and each is reported distinctly so an operator can tell "never
    hashed" from "changed since ingestion".

    Parameters
    ----------
    expected_sha256:
        The digest recorded at acquisition. ``None`` means unknown, which is a
        failure - never a pass.
    path:
        The file to re-hash. When ``None`` and ``require_bytes`` is set, the
        check fails: bytes that cannot be re-read cannot be verified.
    require_bytes:
        Set ``False`` only for synthetic demonstrations, where no source file is
        expected to exist. Observational paths must leave this at ``True``.
    """
    if not expected_sha256:
        return HashVerification(
            HASH_NOT_RECORDED, None, None, str(path) if path else None,
            "no source hash was recorded at acquisition; provenance cannot be "
            "verified and is never inferred from a filename",
        )
    if path is None:
        status = HASH_UNAVAILABLE if require_bytes else HASH_NOT_RECORDED
        return HashVerification(
            status, expected_sha256, None, None,
            "the source file path is not recorded, so its bytes cannot be verified",
        )
    if not Path(path).exists():
        return HashVerification(
            HASH_UNAVAILABLE, expected_sha256, None, str(path),
            "the source file is missing, so its bytes cannot be verified",
        )
    actual = hash_file_or_none(path)
    if actual is None:
        return HashVerification(
            HASH_UNREADABLE, expected_sha256, None, str(path),
            "the source file exists but could not be read for hashing",
        )
    if actual.lower() != expected_sha256.strip().lower():
        return HashVerification(
            HASH_MISMATCH, expected_sha256, actual, str(path),
            "the file on disk does not match the digest recorded at acquisition; "
            "it has been modified since ingestion",
        )
    return HashVerification(
        HASH_OK, expected_sha256, actual, str(path),
        "the file matches its recorded SHA-256 digest",
    )



class DataClass:
    """The honest label for a data stream.

    These are deliberately distinct strings: a reanalysis is not an
    observation, and a derived field is neither. The UI and API surface this
    verbatim, so a field can never be presented under a more impressive label
    than it earned.
    """

    OBSERVATION = "observation"
    REANALYSIS = "reanalysis"
    DERIVED = "derived"
    INTERPOLATED = "interpolated"
    SYNTHETIC = "synthetic"

    ALL = (OBSERVATION, REANALYSIS, DERIVED, INTERPOLATED, SYNTHETIC)

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls.ALL:
            raise ValueError(
                f"unknown data class {value!r}; expected one of {list(cls.ALL)}"
            )
        return value


#: Substrings that must never appear in a persisted provenance field.
_SECRET_HINTS = ("password", "passwd", "token", "secret", "cookie", "api_key", "apikey")


def _scrub(value: Any) -> Any:
    """Mask anything that looks like a credential before it is persisted."""
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if any(h in str(key).lower() for h in _SECRET_HINTS) else _scrub(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(hint in lowered for hint in _SECRET_HINTS) and "=" in value:
            return "<redacted>"
        return value
    return value


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so large NetCDFs need not be loaded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_key(
    source: str,
    product: str,
    acquired_at: datetime | str,
    checksum: str,
) -> str:
    """Stable de-duplication key for one ingested product.

    Two runs over the same file produce the same key, so a unique index on this
    column blocks duplicate ingestion at the database level.
    """
    if isinstance(acquired_at, datetime):
        stamp = acquired_at.isoformat()
    else:
        stamp = str(acquired_at)
    raw = "|".join([source.strip(), product.strip(), stamp, checksum.strip().lower()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class QualityReport:
    """Validation outcome for one ingested file or product.

    ``missing_fraction`` is measured, not assumed, and a dataset with too much
    missing data is reported as such rather than being silently accepted.
    """

    n_values: int = 0
    n_missing: int = 0
    #: Additional non-fatal problems (range violations, short time series, ...).
    warnings: list[str] = field(default_factory=list)
    #: Fatal problems; a dataset with any of these is rejected.
    errors: list[str] = field(default_factory=list)
    passed: bool = True

    @property
    def missing_fraction(self) -> float:
        if self.n_values == 0:
            return 0.0
        return self.n_missing / self.n_values

    def fail(self, message: str) -> QualityReport:
        self.errors.append(message)
        self.passed = False
        return self

    def warn(self, message: str) -> QualityReport:
        self.warnings.append(message)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_values": self.n_values,
            "n_missing": self.n_missing,
            "missing_fraction": round(self.missing_fraction, 6),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ResamplingRecord:
    """One documented spatial or temporal resampling step.

    Recorded rather than implied, so a reviewer can always answer "was this
    value interpolated, averaged, or taken verbatim?" - the question that
    decides whether a 30-minute frame is an observation or an estimate.
    """

    stage: str
    method: str
    #: Source resolution / cadence.
    source_resolution: str
    #: Target resolution / cadence.
    target_resolution: str
    #: What the source resolution actually was, in physical units.
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "method": self.method,
            "source_resolution": self.source_resolution,
            "target_resolution": self.target_resolution,
            "detail": self.detail,
        }


@dataclass(slots=True)
class DatasetProvenance:
    """A reproducible record of one ingested dataset.

    Populated by the pipeline and persisted to the ``dataset_provenance``
    table. Every field is either a fact about the data or a documented choice
    made while processing it; nothing is left implicit.
    """

    source: str
    product: str
    #: One of :class:`DataClass` - observation, reanalysis, derived, synthetic.
    data_class: str
    #: Observation / valid time of the data itself.
    acquired_at: datetime
    #: When this system read the file.
    ingested_at: datetime = field(default_factory=utcnow)
    path: str | None = None
    checksum: str = ""

    # -- spatial -----------------------------------------------------------
    min_lon: float | None = None
    min_lat: float | None = None
    max_lon: float | None = None
    max_lat: float | None = None
    crs: str = "EPSG:4326"
    source_resolution: str = ""
    processed_resolution: str = ""

    # -- temporal ----------------------------------------------------------
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_cadence: str = ""
    processed_cadence: str = ""

    # -- content -----------------------------------------------------------
    variables: list[dict[str, Any]] = field(default_factory=list)
    n_frames: int = 0

    # -- processing --------------------------------------------------------
    quality: QualityReport = field(default_factory=QualityReport)
    resampling: list[ResamplingRecord] = field(default_factory=list)
    processing_config: dict[str, Any] = field(default_factory=dict)
    processing_version: str = "phase5-1.0"
    pipeline: str = "app.ingestion.realtime.pipeline"

    #: Attribution / licensing text that must be reproduced downstream.
    attribution: str = ""
    license_note: str = ""

    # ---------------------------------------------- phase 8.3 (migration 0003)
    #: SHA-256 of the ORIGINAL source bytes, or ``None`` when they are not
    #: available. Never synthesised, never derived from a filename.
    source_sha256: str | None = None
    #: Path of the file :attr:`source_sha256` was computed from. Re-hashing this
    #: path must reproduce the digest; the two are never allowed to drift apart.
    source_path: str | None = None
    #: True only when those bytes still exist and can be re-hashed.
    source_bytes_available: bool = False
    #: Organisation that provided the data.
    provider: str = ""
    #: ``field name -> physical unit`` for what this dataset supplies.
    channel_units: dict[str, str] = field(default_factory=dict)
    #: Explicit spatial coverage declaration, e.g. "77.5-80.5E 29.0-31.5N".
    coverage: str | None = None
    #: Acquisition time of the data itself, distinct from ``ingested_at``.
    acquired_at_src: datetime | None = None

    def __post_init__(self) -> None:
        self.data_class = DataClass.validate(self.data_class)

    @property
    def key(self) -> str:
        """De-duplication key (see :func:`ingest_key`)."""
        return ingest_key(self.source, self.product, self.acquired_at, self.checksum)

    def bounds(self) -> tuple[float, float, float, float] | None:
        if None in (self.min_lon, self.min_lat, self.max_lon, self.max_lat):
            return None
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)  # type: ignore[return-value]

    def overlaps_grid(self, grid: GridSpec) -> bool:
        """Whether this dataset's spatial extent intersects the model AOI."""
        extent = self.bounds()
        if extent is None:
            return False
        return not (
            extent[2] < grid.min_lon
            or extent[0] > grid.max_lon
            or extent[3] < grid.min_lat
            or extent[1] > grid.max_lat
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ingest_key"] = self.key
        payload["acquired_at"] = self.acquired_at.isoformat()
        payload["ingested_at"] = self.ingested_at.isoformat()
        payload["valid_from"] = self.valid_from.isoformat() if self.valid_from else None
        payload["valid_to"] = self.valid_to.isoformat() if self.valid_to else None
        return _scrub(payload)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
