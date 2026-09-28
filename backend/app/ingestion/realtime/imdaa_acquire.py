"""Entry point for authenticated IMDAA acquisition (Phase 7).

Two honest modes, chosen by what is actually available:

**With credentials** (``SIHPS_RDS_EMAIL`` / ``SIHPS_RDS_PASSWORD`` set): submits
a small, explicitly configured sample request for the Uttarakhand AOI, waits,
downloads, validates the file, derives CAPE only if temperature *and* specific
humidity are present, aligns to the canonical grid, and records provenance.

**Without credentials**: reports the precise external action required and exits
non-zero *without* pretending any data was acquired. The public catalog is
still queried, so the operator sees exactly which product would be requested.

Run::

    python -m app.ingestion.realtime.imdaa_acquire --check
    python -m app.ingestion.realtime.imdaa_acquire --year 2019 --month 07 --day 01
    python -m app.ingestion.realtime.imdaa_acquire --split-coverage
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.db.repository import record_dataset_provenance
from app.db.session import get_session_factory
from app.grid import GridSpec
from app.ingestion.base import source_file_from_disk
from app.ingestion.realtime.imdaa_netcdf import (
    IMDAA_PROVIDER,
    build_observation_cube,
    validate_imdaa_file,
)
from app.ingestion.realtime.provenance import (
    DataClass,
    DatasetProvenance,
    ResamplingRecord,
    file_sha256,
)
from app.ingestion.realtime.rds_client import (
    RDS_API_URL,
    RDSClient,
    RDSCredentials,
    RDSDownloadError,
    build_sample_request,
)
from app.logging_conf import get_logger, setup_logging

logger = get_logger("ingestion.realtime.imdaa_acquire")

#: Provider recorded on every IMDAA acquisition, defined next to the reader so
#: the connector and this entry point cannot disagree. Re-exported here.
__all__ = [
    "IMDAA_PROVIDER",
    "SplitCoverage",
    "acquire_sample",
    "assess_split_coverage",
    "main",
]

IMDAA_ATTRIBUTION = (
    "IMDAA reanalysis: NCMRWF/IMD/Met Office under the National Monsoon Mission; "
    "Met Office Unified Model with an incremental 4D-Var data assimilation system. "
    "Retrieved from the NCMRWF RDS portal (https://rds.ncmrwf.gov.in/)."
)

#: The split Phase 5 proposed for the IMDAA record.
TARGET_SPLIT: dict[str, tuple[int, int]] = {
    "train": (2008, 2014),
    "val": (2015, 2017),
    "test": (2018, 2020),
}

#: Published IMDAA coverage, used only to report the boundary of feasibility.
IMDAA_COVERAGE = (1979, 2020)


@dataclass(slots=True)
class SplitCoverage:
    """Whether the target year split can actually be filled from real files.

    Computed from the years that are *actually present on disk*, never from the
    product's advertised range. A year with no file counts as absent, so the
    report cannot be flattered by assuming coverage that was never downloaded.
    """

    years_requested: dict[str, tuple[int, int]]
    years_available: list[int]
    satisfied: dict[str, bool]
    missing_years: dict[str, list[int]]
    verdict: str
    notes: list[str]

    @property
    def complete(self) -> bool:
        return all(self.satisfied.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "years_requested": {k: list(v) for k, v in self.years_requested.items()},
            "years_available": list(self.years_available),
            "satisfied": dict(self.satisfied),
            "missing_years": {k: list(v) for k, v in self.missing_years.items()},
            "complete": self.complete,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def _years_from_time_axis(dataset) -> list[int]:
    """Read years from a time axis, tolerating a string-typed axis.

    ``xarray`` only exposes ``.dt`` when the axis was decoded to datetimes; a
    file whose time coordinate is stored as text has no ``.dt``. Both cases are
    handled rather than assuming a decode.
    """
    if "time" not in dataset.coords:
        return []
    values = np.asarray(dataset["time"].values).ravel()
    years: set[int] = set()
    for value in values:
        text = str(value)
        match = re.match(r"(\d{4})", text)
        if match:
            years.add(int(match.group(1)))
    return sorted(years)


def years_in_files(paths: Sequence[Path]) -> list[int]:
    """Extract the distinct years present in a set of IMDAA NetCDF files.

    The time axis is authoritative; the filename is only a fallback. A file
    whose axis says 2016 but whose name says 2008 is therefore counted as 2016,
    so a mislabelled file is visible rather than silently placed in the wrong
    split.
    """
    years: set[int] = set()
    for path in paths:
        if not path.exists():
            continue
        from_file: set[int] = set()
        try:
            import xarray as xr

            with xr.open_dataset(path) as dataset:
                from_file.update(_years_from_time_axis(dataset))
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            logger.warning(
                "could not read the time axis",
                extra={"path": path.name, "error": type(exc).__name__},
            )
        if from_file:
            years.update(from_file)
            continue
        # Fallback: a 4-digit year anywhere in the file name.
        for match in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.stem):
            years.add(int(match))
    return sorted(years)


def assess_split_coverage(
    data_dir: str | Path | None = None, *, requested: dict[str, tuple[int, int]] | None = None
) -> SplitCoverage:
    """Report whether the target split is satisfiable from files actually present.

    An empty directory is reported as ``no_data`` rather than as a failure to
    parse, because "nothing downloaded yet" is a different fact from "the data
    is broken".
    """
    requested = dict(requested or TARGET_SPLIT)
    directory = Path(data_dir) if data_dir else Path(get_settings().imdaa_dir or "")
    files: list[Path] = []
    if directory and directory.exists():
        files = sorted(directory.glob("*.nc")) + sorted(directory.glob("*.nc4"))

    available = years_in_files(files)
    satisfied: dict[str, bool] = {}
    missing: dict[str, list[int]] = {}
    for name, (lo, hi) in requested.items():
        wanted = list(range(lo, hi + 1))
        absent = [year for year in wanted if year not in available]
        missing[name] = absent
        satisfied[name] = not absent

    notes: list[str] = []
    if not files:
        verdict = "no_data"
        notes.append(
            "no IMDAA NetCDF files are present, so no year can be verified. "
            "Acquire a sample first (see the manual_instructions field)."
        )
    elif self_out_of_range := [
        year for year in available if not (IMDAA_COVERAGE[0] <= year <= IMDAA_COVERAGE[1])
    ]:
        verdict = "out_of_published_range"
        notes.append(
            f"file(s) carry years {self_out_of_range} outside the published IMDAA "
            f"record {IMDAA_COVERAGE[0]}-{IMDAA_COVERAGE[1]}; they are excluded "
            "from the split until the discrepancy is explained."
        )
    elif all(satisfied.values()):
        verdict = "complete"
        notes.append(
            "every year in the target split is present in the acquired files."
        )
    else:
        verdict = "incomplete"
        for name, absent in missing.items():
            if absent:
                notes.append(
                    f"split {name!r} is missing {len(absent)} year(s): {absent[:8]}"
                    f"{' ...' if len(absent) > 8 else ''}"
                )
        notes.append(
            "missing years are reported, never filled with a substitute period."
        )
    return SplitCoverage(
        years_requested=requested,
        years_available=available,
        satisfied=satisfied,
        missing_years=missing,
        verdict=verdict,
        notes=notes,
    )



def acquire_sample(
    *,
    dataset_type: str = "imdaa-daily",
    year: str = "2019",
    month: Sequence[str] = ("07",),
    day: Sequence[str] = ("01",),
    time: Sequence[str] = ("00",),
    variables: Sequence[str] = ("2t", "dpt", "r", "u", "v", "msl", "tcc"),
    frequency: str | None = None,
    pressure_level: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    out_dir: str | Path | None = None,
    persist: bool = True,
    poll_seconds: float = 10.0,
    max_polls: int = 120,
) -> dict[str, Any]:
    """Acquire, validate and record one explicitly configured IMDAA sample.

    Raises :class:`~app.ingestion.realtime.rds_client.RDSDownloadError` when the
    portal cannot be used. It never returns a result implying a download that
    did not happen.
    """
    settings = get_settings()
    grid: GridSpec = settings.grid
    target_bbox = bbox or (grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat)
    root = Path(out_dir) if out_dir else Path(settings.imdaa_dir or settings.data_dir / "imdaa")
    raw_dir = root

    credentials = RDSCredentials(
        email=settings.rds_email, password=settings.rds_password
    )
    client = RDSClient(
        base_url=settings.rds_api_url or RDS_API_URL,
        credentials=credentials,
    )
    # Fails fast, with the exact external action, when no account is configured.
    client.credentials.require()

    request = build_sample_request(
        dataset_type=dataset_type,
        year=year,
        month=month,
        day=day,
        time=time,
        variables=variables,
        min_lon=target_bbox[0],
        min_lat=target_bbox[1],
        max_lon=target_bbox[2],
        max_lat=target_bbox[3],
        frequency=frequency,
        pressure_level=pressure_level,
    )
    job_submitted_at = datetime.now(tz=timezone.utc)
    job, path = client.run_sample(
        request, raw_dir, poll_seconds=poll_seconds, max_polls=max_polls
    )
    report = validate_imdaa_file(path)
    # The file's own latitude/longitude extent is the real coverage; the
    # requested bbox is only a fallback for a file that declares no extent.
    coverage_string = _coverage_from_report(report, target_bbox)

    resampling: list[ResamplingRecord] = []
    derived: dict[str, Any] = {}
    cube_summary: dict[str, Any] = {"built": False}
    if report.ok and report.pressure_levels_hpa:
        # Real feature extraction: derive CAPE/IWV, align to the canonical grid
        # and assemble the cube. A file that cannot support either channel is
        # reported, not padded with a plausible number.
        try:
            cube = build_observation_cube(path, grid, report=report)
        except ValueError as exc:
            derived["channels"] = {
                "attempted": True,
                "failed": True,
                "reason": str(exc),
            }
        else:
            resampling = [
                ResamplingRecord(**r) for r in cube.metadata["resampling"]
            ]
            derived["channels"] = {
                "attempted": True,
                "failed": False,
                "available": cube.metadata["channels_available"],
                "missing": cube.metadata["channels_missing"],
                "coverage": cube.metadata["channel_coverage"],
                "trainable": cube.metadata["trainable"],
                "trainable_reason": cube.metadata["trainable_reason"],
            }
            derived["cape"] = {
                "attempted": True,
                "method": "pseudo-adiabatic parcel ascent (app.physics.parcel_ascent)",
                "finite_cells": int(
                    np.count_nonzero(np.isfinite(cube.channel("cape")))
                ),
            }
            derived["iwv"] = {
                "attempted": True,
                "method": "(1/g) * integral(q dp) by the trapezoidal rule",
                "finite_cells": int(
                    np.count_nonzero(np.isfinite(cube.channel("iwv")))
                ),
            }
            derived["notes"] = cube.metadata["notes"]
            cube_summary = {
                "built": True,
                "n_frames": cube.n_frames,
                "shape": list(cube.channels.shape),
                "grid": cube.grid.to_dict(),
                "channels_available": cube.metadata["channels_available"],
            }
    elif report.ok:
        derived["channels"] = {
            "attempted": False,
            "reason": (
                "this file has no pressure axis, so it cannot supply the CAPE or "
                "IWV channels"
            ),
        }

    quality = report.quality
    quality.warn(
        "IMDAA is a reanalysis, not an observation: it is model output that "
        "assimilates observations"
    )
    # ------------------------------------------- phase 8.3/8.4 integrity fields
    # The *downloaded* file is the source of record. `checksum` above is the same
    # file's digest, but it is recorded here explicitly as the source digest so the
    # manifest and the aggregate cannot be confused. No digest is invented when
    # the file cannot be read.
    source_file = source_file_from_disk(
        path,
        provider=IMDAA_PROVIDER,
        role="imdaa_pressure_levels" if report.pressure_levels_hpa else "imdaa_single_level",
        source_identity=str(job.job_id),
    )
    provenance = DatasetProvenance(
        source="NCMRWF RDS (IMDAA)",
        product=dataset_type,
        data_class=DataClass.REANALYSIS,
        acquired_at=datetime.now(tz=timezone.utc),
        path=str(path),
        checksum=file_sha256(path),
        min_lon=report.lon_range[0] if report.lon_range else target_bbox[0],
        min_lat=report.lat_range[0] if report.lat_range else target_bbox[1],
        max_lon=report.lon_range[1] if report.lon_range else target_bbox[2],
        max_lat=report.lat_range[1] if report.lat_range else target_bbox[3],
        crs=report.crs,
        source_resolution="~12 km IMDAA native grid",
        processed_resolution=f"not yet resampled (native {report.crs})",
        source_cadence=report.times[0] if report.times else "unknown",
        processed_cadence="none (single sample)",
        n_frames=len(report.times),
        quality=quality,
        resampling=resampling,
        processing_config={
            "request": request,
            "job": job.to_dict(),
            "detected_variables": report.detected,
            "units": report.units,
            "pressure_levels_hpa": report.pressure_levels_hpa,
            "derived": derived,
        },
        attribution=IMDAA_ATTRIBUTION,
        license_note="Cite NCMRWF/IMD and the IMDAA dataset when redistributing.",
        # ------------------------------------------- phase 8.3/8.4 integrity
        source_sha256=source_file.sha256,
        source_path=source_file.path,
        source_bytes_available=source_file.bytes_available,
        source_files=[source_file],
        provider=IMDAA_PROVIDER,
        acquired_at_src=job_submitted_at,
        channel_units=imdaa_channel_units(report),
        coverage=coverage_string,
    )
    if persist and report.ok:
        with get_session_factory().scope() as session:
            _row, _created = record_dataset_provenance(session, provenance)

    return {
        "acquired": True,
        "job": job.to_dict(),
        "file": str(path),
        "sha256": provenance.checksum,
        "validation": report.to_dict(),
        "derived": derived,
        "cube": cube_summary,
        "provenance": provenance.to_dict(),
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sihps-imdaa",
        description=(
            "Acquire a small IMDAA sample from the authenticated NCMRWF RDS "
            "portal, or report exactly what is missing."
        ),
    )
    parser.add_argument("--dataset-type", default="imdaa-daily")
    parser.add_argument("--year", default="2019")
    parser.add_argument("--month", default="07")
    parser.add_argument("--day", default="01")
    parser.add_argument("--time", default="00")
    parser.add_argument("--frequency", default=None, help="e.g. 3H for pressure levels")
    parser.add_argument(
        "--variables",
        default="2t,dpt,r,u,v,msl,tcc",
        help="Comma-separated variable codes, exactly as the portal lists them.",
    )
    parser.add_argument(
        "--pressure-levels",
        default=None,
        help="Comma-separated pressure levels, e.g. 1000,925,850,700,500.",
    )
    parser.add_argument("--bbox", default=None, help="min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-polls", type=int, default=120)
    parser.add_argument(
        "--check", action="store_true", help="Report access status and exit."
    )
    parser.add_argument(
        "--split-coverage", action="store_true", help="Report split coverage and exit."
    )
    parser.add_argument(
        "--acquisition-readiness",
        action="store_true",
        help=(
            "Report the full chain: source access, files acquired, data processed, "
            "and observational readiness. Exits non-zero unless every stage passes."
        ),
    )
    parser.add_argument(
        "--readiness-dir", default=None,
        help="Directory searched for acquired source files (readiness check).",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Report 'not_run' for CAPE/IWV validation instead of executing it.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _coverage_from_report(report, fallback_bbox) -> str:
    """Coverage string from the file's own extent, falling back to the request.

    The file's lat/lon range is a measurement; the requested bbox is an
    intention. The measurement wins whenever the file declares one.
    """
    if report.lon_range and report.lat_range:
        (min_lon, max_lon), (min_lat, max_lat) = report.lon_range, report.lat_range
        return f"{min_lon}-{max_lon}E {min_lat}-{max_lat}N"
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in fallback_bbox)
    return f"{min_lon}-{max_lon}E {min_lat}-{max_lat}N"


def imdaa_channel_units(report) -> dict[str, str]:
    """Physical units of the model channels this IMDAA file can supply.

    Derived from the file's own declared NetCDF units, falling back to the
    documented unit for the field. Only channels IMDAA genuinely provides are
    listed: the INSAT radiances and DEM elevation stay absent, so the readiness
    gate continues to block on them rather than borrowing a station field's unit.
    """
    from app.ingestion.realtime.imdaa_netcdf import (
        IMDAA_CHANNEL_UNITS as _FALLBACK_UNITS,
    )

    units: dict[str, str] = {}
    declared = dict(report.units or {})
    for netcdf_field, channel in _IMDAA_FIELD_TO_CHANNEL.items():
        if netcdf_field not in report.detected:
            continue
        unit = declared.get(netcdf_field) or _FALLBACK_UNITS.get(netcdf_field)
        if not unit or unit.lower() in ("unknown", ""):
            continue
        units[channel] = unit
    return units


#: NetCDF field -> model channel, for the fields IMDAA can actually supply.
_IMDAA_FIELD_TO_CHANNEL: dict[str, str] = {
    "temperature": "cape",          # CAPE is derived from this column
    "specific_humidity": "iwv",     # IWV is derived from this column
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = get_settings()

    if args.acquisition_readiness:
        from app.ingestion.realtime.acquisition_readiness import (
            assess_acquisition_readiness,
        )

        report = assess_acquisition_readiness(
            data_dir=args.readiness_dir or settings.imdaa_dir or None,
            run_validation=not args.skip_validation,
        )
        payload = report.to_dict()
        print("=" * 74)
        print("ACQUISITION READINESS")
        print("=" * 74)
        for stage in payload["stages"]:
            print(f"[{'ok  ' if stage['passed'] else 'FAIL'}] "
                  f"{stage['name']:24s} {stage['detail']}")
        print("-" * 74)
        print(f"channels    : {len(payload['channels_available'])}/12 available, "
              f"{len(payload['channels_missing'])} missing")
        if payload["channels_missing"]:
            print("  missing   : " + ", ".join(payload["channels_missing"]))
        print(f"validation  : cape={payload['validation_status']['cape']} "
              f"iwv={payload['validation_status']['iwv']}")
        print("-" * 74)
        print("next actions:")
        for action in payload["next_actions"]:
            print(f"  - {action}")
        print("-" * 74)
        print("OBSERVATIONAL READINESS:", payload["observational_readiness"])
        return 0 if payload["observational_readiness"] else 1

    if args.split_coverage:
        coverage = assess_split_coverage(args.out_dir or settings.imdaa_dir or None)
        print("=" * 74)
        print("IMDAA SPLIT COVERAGE")
        print("=" * 74)
        print("verdict :", coverage.verdict)
        print("available years:", coverage.years_available or "none on disk")
        for name, absent in coverage.missing_years.items():
            span = coverage.years_requested[name]
            print(f"  {name:5s} {span[0]}-{span[1]}: " + ("complete" if not absent else f"missing {len(absent)}"))
        for note in coverage.notes:
            print("  -", note)
        return 0 if coverage.complete else 1

    client = RDSClient(
        base_url=settings.rds_api_url or RDS_API_URL,
        credentials=RDSCredentials(email=settings.rds_email, password=settings.rds_password),
    )
    status = client.availability(probe=args.check)

    if args.check:
        print("=" * 74)
        print("NCMRWF RDS ACCESS")
        print("=" * 74)
        print("availability:", status.availability.value)
        print("reason      :", status.reason)
        if status.manual_instructions:
            print("required    :", status.manual_instructions)
        print("credentials :", json.dumps(status.details["credentials"]))
        print(
            "note        : the public catalog (GET /datasets/catalog) is open, "
            "but /jobs requires a bearer token."
        )
        return 0 if status.availability.value == "available" else 1

    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    levels = (
        [p.strip() for p in args.pressure_levels.split(",") if p.strip()]
        if args.pressure_levels
        else None
    )
    bbox = None
    if args.bbox:
        parts = [float(p) for p in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
        bbox = (parts[0], parts[1], parts[2], parts[3])

    try:
        result = acquire_sample(
            dataset_type=args.dataset_type,
            year=args.year,
            month=[m.strip() for m in args.month.split(",") if m.strip()],
            day=[d.strip() for d in args.day.split(",") if d.strip()],
            time=[t.strip() for t in args.time.split(",") if t.strip()],
            variables=variables,
            frequency=args.frequency,
            pressure_level=levels,
            bbox=bbox,
            out_dir=args.out_dir,
            persist=not args.no_persist,
            poll_seconds=args.poll_seconds,
            max_polls=args.max_polls,
        )
    except RDSDownloadError as exc:
        print("IMDAA acquisition did not complete:", exc, file=__import__("sys").stderr)
        print(
            "No data was acquired and no provenance was written.",
            file=__import__("sys").stderr,
        )
        return 2

    print("=" * 74)
    print("IMDAA SAMPLE ACQUIRED")
    print("=" * 74)
    print("file      :", result["file"])
    print("sha256    :", result["sha256"][:32], "...")
    print("job       :", result["job"]["job_id"], "->", result["job"]["status"])
    validation = result["validation"]
    print("valid     :", validation["ok"])
    print("variables :", validation["detected"])
    print("levels    :", validation["n_pressure_levels"])
    print("crs       :", validation["crs"])
    cube = result["cube"]
    print("cube      :", "built" if cube["built"] else "NOT built")
    if cube["built"]:
        print("  frames  :", cube["n_frames"], "shape", cube["shape"])
        print("  channels:", cube["channels_available"], "(of 12)")
    channels = result["derived"].get("channels", {})
    if channels.get("failed"):
        print("  reason  :", channels["reason"])
    for name in ("cape", "iwv"):
        entry = result["derived"].get(name)
        if entry and entry.get("attempted"):
            print(f"  {name:7s}: {entry['method']} -> {entry['finite_cells']} cells")
    if channels.get("trainable_reason"):
        print("trainable :", channels["trainable_reason"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

