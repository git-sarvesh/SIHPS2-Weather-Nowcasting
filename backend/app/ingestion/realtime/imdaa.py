"""IMDAA reanalysis and MERA rainfall adapters (NCMRWF RDS portal).

Verified access facts
---------------------
``https://rds.ncmrwf.gov.in/api/datasets/catalog`` and ``/api/openapi.json``
are readable **without authentication** and were queried directly while writing
this module. The published facts encoded below are transcribed from that
response, not assumed:

===========================  ==========  =========  ==========  ==========
Dataset (slug)               Period      Cadence    Resolution Variables
===========================  ==========  =========  ==========  ==========
``hourly-pressure`` (IMDAA)  1979-2020   3-hourly   ~12 km      T, RH, u, v,
                                                            geopotential
                                                            height, 10-1000 hPa
``imdaa-daily``              1979-2020   daily      ~12 km      single level
``mera`` (rainfall)          2020-2025   hourly     4 km        rainfall
===========================  ==========  =========  ==========  ==========

**Coverage consequence.** IMDAA ends in 2020; MERA begins in 2020. The
project's required evaluation window (2024-2025) is therefore reachable only
through MERA, and only as rainfall - never as upper-air state.
:func:`assess_split_feasibility` computes this overlap from the published
ranges and reports the shortfall rather than quietly degrading the split.

The bulk ``/download/request`` endpoint is account-gated, so files are read from
a **local, operator-supplied NetCDF directory**. That keeps the project
offline-capable and makes each ingestion step reproducible from checksummed
files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import GridSpec
from app.ingestion.base import DataConnector
from app.ingestion.realtime.access import AccessStatus, Availability
from app.ingestion.realtime.imdaa_netcdf import IMDAA_PROVIDER
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.imdaa")

__all__ = [
    "CoverageResult",
    "IMDAAConnector",
    "IMDAAProduct",
    "PRODUCTS",
    "REQUIRED_SPLIT",
    "SplitFeasibility",
    "assess_split_feasibility",
    "catalog_entry",
    "coverage",
    "list_datasets",
    "parse_times",
]

#: Public, credential-free catalog endpoints (verified reachable).
CATALOG_URL = "https://rds.ncmrwf.gov.in/api/datasets/catalog"
DATASET_URL = "https://rds.ncmrwf.gov.in/api/datasets/{slug}"
OPENAPI_URL = "https://rds.ncmrwf.gov.in/api/openapi.json"



@dataclass(frozen=True, slots=True)
class IMDAAProduct:
    """One documented IMDAA/MERA product, as published by the portal."""

    slug: str
    title: str
    start_year: int
    end_year: int
    cadence_hours: int
    resolution_km: float
    product_type: str
    #: NetCDF variable name -> model channel it feeds, where a mapping exists.
    channel_map: dict[str, str] = field(default_factory=dict)

    @property
    def years(self) -> tuple[int, int]:
        return (self.start_year, self.end_year)

    def covers(self, year: int) -> bool:
        return self.start_year <= year <= self.end_year

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "period": f"{self.start_year}-{self.end_year}",
            "cadence_hours": self.cadence_hours,
            "resolution_km": self.resolution_km,
            "product_type": self.product_type,
            "channel_map": dict(self.channel_map),
        }


#: Published products, transcribed from the portal catalog.
PRODUCTS: dict[str, IMDAAProduct] = {
    "hourly-pressure": IMDAAProduct(
        slug="hourly-pressure",
        title="IMDAA 3-hourly data on pressure levels",
        start_year=1979,
        end_year=2020,
        cadence_hours=3,
        resolution_km=12.0,
        product_type="IMDAA",
        channel_map={"Relative_humidity": "iwv"},
    ),
    "imdaa-daily": IMDAAProduct(
        slug="imdaa-daily",
        title="IMDAA daily data on single level",
        start_year=1979,
        end_year=2020,
        cadence_hours=24,
        resolution_km=12.0,
        product_type="IMDAA",
    ),
    "mera": IMDAAProduct(
        slug="mera",
        title="MERA hourly rainfall analysis",
        start_year=2020,
        end_year=2025,
        cadence_hours=1,
        resolution_km=4.0,
        product_type="Rainfall",
        channel_map={"rainfall": "rain_class"},
    ),
}

#: The split the project requirements ask for.
REQUIRED_SPLIT: dict[str, tuple[int, int]] = {
    "train": (2015, 2022),
    "val": (2023, 2023),
    "test": (2024, 2025),
}


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """How much of the model AOI a supplied product set actually covers."""

    product: str
    cells_expected: int
    cells_covered: int
    coverage_fraction: float
    n_files: int = 0

    @property
    def complete(self) -> bool:
        return self.cells_expected > 0 and self.coverage_fraction >= 0.999

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "cells_expected": self.cells_expected,
            "cells_covered": self.cells_covered,
            "coverage_fraction": self.coverage_fraction,
            "n_files": self.n_files,
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class SplitFeasibility:
    """Whether a requested train/val/test year split can be built from real data."""

    requested: dict[str, tuple[int, int]]
    available: dict[str, tuple[int, int]]
    feasible: bool
    unsatisfiable: list[str]
    proposed: dict[str, tuple[int, int]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": {k: list(v) for k, v in self.requested.items()},
            "available": {k: list(v) for k, v in self.available.items()},
            "feasible": self.feasible,
            "unsatisfiable": self.unsatisfiable,
            "proposed": {k: list(v) for k, v in self.proposed.items()},
            "notes": list(self.notes),
        }

#: Environment variables for the account-gated bulk download service.
RDS_USER_ENV = "SIHPS_RDS_USER"
RDS_PASSWORD_ENV = "SIHPS_RDS_PASSWORD"

IMDAA_ATTRIBUTION = (
    "IMDAA reanalysis: NCMRWF/IMD/Met Office under the National Monsoon Mission; "
    "Met Office Unified Model with an incremental 4D-Var data assimilation system. "
    "Cite the IMDAA dataset and the NCMRWF RDS portal."
)
MERA_ATTRIBUTION = (
    "MERA hourly rainfall analysis: NCMRWF, multi-source satellite + radar blended "
    "analysis. See Amarjyothi et al. (2025), Meteorology and Atmospheric Physics 137, "
    "doi:10.1007/s00703-025-01098-4."
)



def assess_split_feasibility(
    requested: dict[str, tuple[int, int]] | None = None,
    products: Iterable[IMDAAProduct] | None = None,
) -> SplitFeasibility:
    """Compare a requested year split against real published coverage.

    Example
    -------
    >>> report = assess_split_feasibility()
    >>> report.feasible
    False
    >>> "test" in report.unsatisfiable
    True
    """
    requested = dict(requested or REQUIRED_SPLIT)
    catalogue = list(products if products is not None else PRODUCTS.values())
    available = {p.slug: p.years for p in catalogue}

    # A split needs a product covering its whole range. We also record *which*
    # products qualify, because "covered by rainfall only" is not equivalent to
    # "covered by upper-air state" for a nowcasting model.
    unsatisfiable: list[str] = []
    rainfall_only: list[str] = []
    qualifying: dict[str, list[str]] = {}
    for name, (lo, hi) in requested.items():
        matches = [p for p in catalogue if p.start_year <= lo and p.end_year >= hi]
        qualifying[name] = [p.slug for p in matches]
        if not matches:
            unsatisfiable.append(name)
        elif all(p.product_type == "Rainfall" for p in matches):
            rainfall_only.append(name)
    feasible = not unsatisfiable

    notes: list[str] = []
    imdaa = PRODUCTS["hourly-pressure"]
    proposed: dict[str, tuple[int, int]] = {}
    if imdaa.covers(2020) and imdaa.covers(2008):
        # IMDAA (1979-2020) supports an upper-air split entirely inside the
        # published record; MERA (2020-2025) supports rainfall-only evaluation.
        proposed = {"train": (2008, 2014), "val": (2015, 2017), "test": (2018, 2020)}
        notes.append(
            "Proposed alternative (upper-air, IMDAA only): train 2008-2014, "
            "val 2015-2017, test 2018-2020 - all inside the published 1979-2020 "
            "IMDAA record."
        )
    notes.append(
        "The required 2024-2025 test window is covered ONLY by MERA (hourly "
        "rainfall, 4 km, 2020-2025), which contains no upper-air state, so a "
        "joint satellite/reanalysis/rainfall evaluation over 2024-2025 is not "
        "possible with the currently accessible datasets."
    )
    if rainfall_only:
        notes.append(
            "Splits covered by a rainfall-only product (no upper-air state): "
            + ", ".join(rainfall_only)
            + "."
        )
    if unsatisfiable:
        notes.append(
            "Unsatisfiable as specified: "
            + ", ".join(f"{n} {requested[n][0]}-{requested[n][1]}" for n in unsatisfiable)
            + ". No missing observation has been synthesised to fill these years."
        )
    return SplitFeasibility(
        requested=requested,
        available=available,
        feasible=feasible,
        unsatisfiable=unsatisfiable,
        proposed=proposed,
        notes=notes,
    )

def coverage(
    product: IMDAAProduct, grid: GridSpec, files: Iterable[Path] = ()
) -> CoverageResult:
    """Fraction of the model AOI covered by the supplied files for a product.

    Only meaningful for files the operator has actually placed on disk, so an
    empty ``files`` argument yields 0.0 coverage and ``n_files=0``. There is no
    synthetic fallback: an absent product reads as absent.
    """
    paths = [Path(p) for p in files]
    if not paths:
        return CoverageResult(
            product=product.slug,
            cells_expected=int(grid.size),
            cells_covered=0,
            coverage_fraction=0.0,
            n_files=0,
        )
    covered = np.zeros(grid.shape, dtype=bool)
    n_read = 0
    for path in paths:
        if not path.exists():
            continue
        extent = _read_extent(path)
        if extent is None:
            continue
        min_lon, min_lat, max_lon, max_lat = extent
        lon = np.asarray(grid.lon_centers())[None, :]
        lat = np.asarray(grid.lat_centers())[:, None]
        covered |= (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)
        n_read += 1
    cells_covered = int(covered.sum())
    return CoverageResult(
        product=product.slug,
        cells_expected=int(grid.size),
        cells_covered=cells_covered,
        coverage_fraction=round(cells_covered / float(grid.size), 4) if grid.size else 0.0,
        n_files=n_read,
    )


def _read_extent(path: Path) -> tuple[float, float, float, float] | None:
    """Best-effort spatial extent of a NetCDF file, or ``None`` when unreadable."""
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            lons = _coord(ds, ("lon", "longitude", "x"))
            lats = _coord(ds, ("lat", "latitude", "y"))
            if lons is not None and lats is not None:
                return (
                    float(np.nanmin(lons)),
                    float(np.nanmin(lats)),
                    float(np.nanmax(lons)),
                    float(np.nanmax(lats)),
                )
    except Exception as exc:  # noqa: BLE001 - a malformed file is a data fact
        logger.warning(
            "could not read NetCDF extent",
            extra={"path": str(path), "error": type(exc).__name__},
        )
    return None


def _coord(ds, names: tuple[str, ...]):
    for name in names:
        if name in ds.coords:
            return np.asarray(ds.coords[name].values)
    return None


def _coverage_string_from_files(files: Iterable[Path]) -> str | None:
    """Union of the *real* file extents, or ``None`` when none declares one.

    Measured from the files themselves, never from a requested bounding box: a
    request is an intention, an extent read from the data is evidence.
    """
    extents = [extent for extent in (_read_extent(Path(p)) for p in files) if extent]
    if not extents:
        return None
    min_lon = min(e[0] for e in extents)
    min_lat = min(e[1] for e in extents)
    max_lon = max(e[2] for e in extents)
    max_lat = max(e[3] for e in extents)
    return f"{min_lon}-{max_lon}E {min_lat}-{max_lat}N"


def list_datasets(timeout: float = 30.0) -> list[dict[str, Any]]:
    """Fetch the public dataset catalog; ``[]`` when unreachable.

    Callers must read an empty result as "unknown", never as "no data exists".
    """
    if timeout <= 0:
        return []
    try:
        import httpx

        response = httpx.get(CATALOG_URL, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("IMDAA catalog unavailable", extra={"error": type(exc).__name__})
        return []
    return list(payload) if isinstance(payload, list) else []


def catalog_entry(slug: str, timeout: float = 30.0) -> dict[str, Any] | None:
    """Fetch one dataset's published metadata, or ``None`` when unavailable."""
    if timeout <= 0:
        return None
    try:
        import httpx

        response = httpx.get(
            DATASET_URL.format(slug=slug), timeout=timeout, follow_redirects=True
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "IMDAA dataset metadata unavailable",
            extra={"slug": slug, "error": type(exc).__name__},
        )
        return None


class IMDAAConnector(DataConnector):
    """Reads operator-supplied IMDAA / MERA NetCDF files from a local directory.

    Local files rather than direct download: the RDS portal's bulk
    ``/download/request`` endpoint is account-gated. Reading a checksummed local
    directory keeps the project offline-capable and makes each training sample
    reproducible from a file the operator can verify.

    This connector never fabricates a field. If the directory is empty or a file
    is malformed, :meth:`availability` and :meth:`fetch` say so.
    """

    source_name = "IMDAA/MERA (NCMRWF RDS)"
    #: A reanalysis is not synthetic - and is also not an observation. That
    #: distinction is carried separately as ``data_class``.
    is_synthetic = False
    data_class = "reanalysis"

    def __init__(
        self,
        grid: GridSpec,
        *,
        data_dir: str | Path | None = None,
        product: str = "hourly-pressure",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(grid, demo_mode=False, cache_dir=cache_dir)
        self.product_slug = product
        self.product = PRODUCTS.get(product)
        if self.product is None:
            raise ValueError(
                f"unknown IMDAA product {product!r}; known: {sorted(PRODUCTS)}"
            )
        self.data_dir = Path(data_dir) if data_dir else None

    def local_files(self) -> list[Path]:
        """NetCDF files the operator has placed in the configured directory."""
        if self.data_dir is None or not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("*.nc")) + sorted(self.data_dir.glob("*.nc4"))

    def availability(self, *, timeout: float = 30.0) -> AccessStatus:
        """Report what this connector can do right now.

        ``timeout <= 0`` skips the optional catalog read so ``/health`` stays
        fast and cannot fail on a network hiccup.
        """
        files = self.local_files()
        catalogue = list_datasets(timeout=timeout)
        details: dict[str, Any] = {
            "product": self.product.to_dict(),
            "local_dir": str(self.data_dir) if self.data_dir else None,
            "n_local_files": len(files),
            "catalog_entries_seen": len(catalogue),
            "bulk_download": "account-gated (POST /download/request on rds.ncmrwf.gov.in)",
        }
        if files:
            details["coverage"] = coverage(self.product, self.grid, files).to_dict()
            return AccessStatus(
                source=self.source_name,
                availability=Availability.AVAILABLE,
                reason=f"{len(files)} local IMDAA/MERA file(s) present and readable",
                details=details,
                is_synthetic=False,
            )
        return AccessStatus(
            source=self.source_name,
            availability=Availability.NEEDS_MANUAL_DOWNLOAD,
            reason=(
                "portal metadata is readable but no local NetCDF files were supplied; "
                "bulk download requires a registered account"
                if catalogue
                else "no local files supplied and the public catalog was not probed"
            ),
            details=details,
            manual_instructions=(
                (f"Create {self.data_dir} and place IMDAA/MERA NetCDF files there, then "
                 if self.data_dir
                 else "Set SIHPS_IMDAA_DIR to a directory, place IMDAA/MERA NetCDF "
                 "files in it, then ")
                + "re-run ingestion. Files come from registering at "
                "https://rds.ncmrwf.gov.in/ and submitting a download request for the "
                f"'{self.product_slug}' product. The pipeline then reads them offline."
            ),
            is_synthetic=False,
        )

    def fetch(self, *args, **kwargs):
        """Assemble an :class:`ObservationCube` from the local IMDAA/MERA files.

        Real assembly only: each file is read, its pressure-level temperature and
        humidity are turned into CAPE and IWV, and those are aligned to the
        common :class:`~app.grid.GridSpec`. Only the channels IMDAA can genuinely
        supply are filled; the other ten stay NaN, so a consumer cannot mistake
        this for a complete model input.

        Raises when no local file is present or none can yield a derived channel.
        Nothing is fabricated to keep the call from failing.

        Phase 8.3/8.4: every contributing file is hashed from **its own bytes** and
        recorded in a per-file manifest, so the chain can be re-verified later. A
        file whose bytes cannot be read yields no digest - the field stays ``None``
        rather than being invented.
        """
        from app.ingestion.base import ObservationCube, Provenance, source_file_from_disk
        from app.ingestion.realtime.imdaa_netcdf import (
            IMDAA_DERIVED_CHANNELS,
            build_observation_cube,
            validate_imdaa_file,
        )
        from app.ingestion.realtime.provenance import ResamplingRecord
        from app.physics import CHANNELS, N_CHANNELS, channel_index, channel_units

        files = self.local_files()
        if not files:
            raise FileNotFoundError(
                f"no IMDAA/MERA NetCDF files in {self.data_dir}. Download them from "
                "https://rds.ncmrwf.gov.in/ (registered account required) and place "
                "them there; no data is generated in their absence."
            )

        frames: list[Any] = []
        records: list[ResamplingRecord] = []
        notes: list[str] = []
        used: list[Any] = []
        for path in files:
            report = validate_imdaa_file(path)
            if not report.ok:
                notes.append(f"{path.name}: skipped ({'; '.join(report.quality.errors)})")
                continue
            try:
                cube = build_observation_cube(path, self.grid, report=report)
            except ValueError as exc:
                notes.append(f"{path.name}: no derivable channel ({exc})")
                continue
            frames.append(cube)
            records.extend(ResamplingRecord(**r) for r in cube.metadata["resampling"])
            used.append((path, report, cube))

        if not frames:
            raise ValueError(
                f"none of the {len(files)} local file(s) could supply a derived "
                f"channel: {'; '.join(notes) or 'unknown reason'}"
            )

        times: list[Any] = []
        stacked: list[Any] = []
        for cube in frames:
            times.extend(cube.times)
            stacked.append(cube.channels)
        channels = np.concatenate(stacked, axis=0)
        available = [
            name for name in IMDAA_DERIVED_CHANNELS
            if np.isfinite(channels[:, channel_index(name)]).any()
        ]
        missing = [
            spec.name for spec in CHANNELS if spec.name not in available
        ]
        fraction = np.mean(np.isfinite(channels), axis=(1, 2, 3)).astype(np.float32)
        quality = np.broadcast_to(fraction[:, None, None], (len(times), *self.grid.shape)).copy()

        # -- Phase 8.3/8.4: one manifest entry per contributing file ----------
        # The digest is computed from each file's own bytes here; a file that
        # cannot be re-read records NO digest rather than a placeholder.
        declared_units = {name: channel_units()[name] for name in available}
        manifest = [
            source_file_from_disk(
                path,
                provider=IMDAA_PROVIDER,
                role="imdaa_pressure_levels" if report.pressure_levels_hpa else "imdaa_single_level",
                source_identity=self.product_slug,
            )
            for path, report, _cube in used
        ]
        return ObservationCube(
            grid=self.grid,
            times=times,
            channels=channels,
            quality=quality,
            provenance=[
                Provenance(
                    source=self.source_name,
                    product=f"{self.product_slug}:{path.name}",
                    valid_from=min(cube.times),
                    valid_to=max(cube.times),
                    path=str(path),
                    is_synthetic=False,
                    attribution=(
                        MERA_ATTRIBUTION if self.product_slug == "mera"
                        else IMDAA_ATTRIBUTION
                    ),
                    source_sha256=entry.sha256,
                    source_path=str(path),
                    source_bytes_available=entry.bytes_available,
                    provider=IMDAA_PROVIDER,
                    acquired_at=datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ),
                    channel_units=declared_units,
                    coverage=_coverage_string_from_files([p for p, _r, _c in used]),
                    source_files=[entry],
                )
                for (path, _report, cube), entry in zip(used, manifest, strict=False)
            ],
            metadata={
                "files": [str(path) for path, _report, _cube in used],
                "spatial_coverage": _coverage_string_from_files(
                    [p for p, _r, _c in used]
                ),
                "channel_units": declared_units,
                "channels_available": available,
                "channels_missing": missing,
                "provider": IMDAA_PROVIDER,
                "acquired_at_source": (
                    "local file modification time (the earliest verifiable instant "
                    "associated with the bytes)"
                ),
                "source_sha256": [entry.sha256 for entry in manifest],
                "resampling": [r.to_dict() for r in records],
                "notes": notes,
                "trainable": False,
                "trainable_reason": (
                    f"IMDAA supplied {len(available)} of {N_CHANNELS} channels "
                    f"({available}). The remaining {len(missing)} need INSAT-3D "
                    "and/or DEM inputs."
                ),
            },
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["attribution"] = (
            MERA_ATTRIBUTION if self.product_slug == "mera" else IMDAA_ATTRIBUTION
        )
        payload["data_class"] = self.data_class
        payload["availability"] = self.availability(timeout=0.0).to_dict()
        return payload


def parse_times(ds) -> list[datetime]:
    """Decode a NetCDF time axis into timezone-aware UTC datetimes."""
    if "time" not in ds.coords:
        return []
    decoded = ds.time.dt.strftime("%Y-%m-%dT%H:%M:%S").values
    return [datetime.fromisoformat(str(v)) for v in decoded]

