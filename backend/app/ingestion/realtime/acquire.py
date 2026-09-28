"""Acquire a real IMD station-observation dataset and record its provenance.

This is the Phase 6 milestone script. It performs a genuine, keyless download
from IMD's public OGC service over the Uttarakhand AOI, stores the raw bytes
separately from the processed artifact, checksums both, and writes a
``DatasetProvenance`` row plus a dataset-readiness verdict.

Run with::

    python -m app.ingestion.realtime.acquire --bbox 77.5,29.0,80.5,31.5

It does not train anything and does not fabricate any value.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.repository import record_dataset_provenance
from app.db.session import get_session_factory
from app.ingestion.realtime.imd_ogc import (
    IMD_OGC_ATTRIBUTION,
    LAYER_PARSERS,
    IMDOGCConnector,
    OGCFeatureSet,
)
from app.ingestion.realtime.provenance import (
    DataClass,
    DatasetProvenance,
    QualityReport,
    ResamplingRecord,
    file_sha256,
)
from app.ingestion.realtime.readiness import assess_readiness
from app.logging_conf import get_logger, setup_logging

logger = get_logger("ingestion.realtime.acquire")

__all__ = ["AcquisitionOutcome", "acquire_dataset", "main"]

#: Layers acquired by default: the two with verified, physically meaningful
#: observation schemas.
DEFAULT_LAYERS: tuple[str, ...] = ("imd:synop_data_layer", "imd:aws_data_layer")

#: Physical units of the fields this source supplies. Declared once, from the
#: parsers that produce them, so the provenance record cannot drift from the
#: data. Phase 8.3 requires units to be recorded rather than inferred.
IMD_OGC_CHANNEL_UNITS: dict[str, str] = {
    "air_temperature": "degC",
    "dewpoint_temperature": "degC",
    "relative_humidity": "%",
    "pressure_msl": "hPa",
    "wind_speed": "m/s",
    "wind_direction": "degree",
    "rainfall": "mm",
    "rainfall_rate": "mm/h",
    "cloud_cover": "okta",
}


def _coverage_string(bbox: Sequence[float]) -> str:
    """``(min_lon, min_lat, max_lon, max_lat)`` -> ``"minLon-maxLonE minLat-maxLatN"``.

    Uses repr-style float formatting deliberately: ``:g`` renders ``29.0`` as
    ``"29"``, producing a string :func:`app.training.gate.parse_coverage_string`
    rejects. That would make the dataset silently unverifiable at its own gate,
    so the value is always written in a form the parser accepts.
    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    return f"{min_lon}-{max_lon}E {min_lat}-{max_lat}N"


def _raw_digests(raw_files: Sequence[Path]) -> dict[str, str]:
    """``filename -> SHA-256`` for each readable raw download.

    Recorded per file so the byte-set is fully described. A file that cannot be
    read is simply absent from the mapping - no placeholder digest is emitted,
    because an invented hash is indistinguishable from a real one later.
    """
    digests: dict[str, str] = {}
    for path in sorted(raw_files):
        try:
            digests[Path(path).name] = file_sha256(path)
        except OSError:
            continue
    return digests


@dataclass(slots=True)
class AcquisitionOutcome:
    """Everything a reviewer needs to judge one acquisition run."""

    provenance: DatasetProvenance
    readiness: Any
    observations: list[dict[str, Any]]
    raw_files: list[Path]
    processed_path: Path
    per_layer: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "readiness": self.readiness.to_dict(),
            "per_layer": self.per_layer,
            "raw_files": [str(p) for p in self.raw_files],
            "processed": {
                "path": str(self.processed_path),
                "sha256": file_sha256(self.processed_path),
                "n_records": len(self.observations),
            },
        }


def acquire_dataset(
    *,
    bbox: tuple[float, float, float, float] | None = None,
    layers: Sequence[str] = DEFAULT_LAYERS,
    out_dir: str | Path | None = None,
    persist: bool = True,
    timeout: float = 90.0,
) -> AcquisitionOutcome:
    """Download real IMD observations, process them, and record provenance.

    Raw responses are written to ``<out_dir>/raw`` and the processed artifact to
    ``<out_dir>/processed``; the two are never mixed, so a processed file can
    always be traced back to the exact bytes it came from.
    """
    settings = get_settings()
    grid = settings.grid
    target_bbox = tuple(bbox) if bbox else (
        grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat
    )
    root = Path(out_dir) if out_dir else Path(settings.data_dir) / "realtime"
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    connector = IMDOGCConnector(grid, bbox=target_bbox, raw_dir=raw_dir, timeout=timeout)

    observations: list[dict[str, Any]] = []
    per_layer: dict[str, dict[str, Any]] = {}
    raw_files: list[Path] = []
    resampling: list[ResamplingRecord] = []
    retrieved: list[datetime] = []

    for layer in layers:
        fetched: OGCFeatureSet = connector.fetch_layer(layer)
        records = LAYER_PARSERS[layer](fetched)
        payload = json.dumps(fetched.raw.decode("utf-8", "replace"), default=str)
        # Recompute the raw artefact path deterministically for the record.
        from app.ingestion.realtime.imd_ogc import _persist_raw

        raw_path = _persist_raw(fetched, raw_dir)
        raw_files.append(raw_path)
        retrieved.append(fetched.retrieved_at)
        per_layer[layer] = {
            **fetched.summary(),
            "n_parsed_records": len(records),
            "raw_path": str(raw_path),
            "raw_sha256": file_sha256(raw_path),
            "n_bytes": len(payload.encode("utf-8")),
        }
        observations.extend(record.to_dict() for record in records)
        if layer == "imd:synop_data_layer":
            resampling.append(
                ResamplingRecord(
                    stage="accumulation_to_rate",
                    method="amount_mm / window_hours -> mm/h (WMO 3/6/12/24 h window)",
                    source_resolution="3/6/12/24 h accumulation",
                    target_resolution="instantaneous rate mm/h",
                    detail=(
                        "SYNOP publishes accumulated rainfall, not a rate. The "
                        "shortest published window is used per record and recorded "
                        "in rain_window_h; no value is invented when absent."
                    ),
                )
            )
        else:
            resampling.append(
                ResamplingRecord(
                    stage="accumulation_to_rate",
                    method="amount_mm / (rain_sel minutes / 60) -> mm/h",
                    source_resolution="per-station accumulation window (0.5-10 min)",
                    target_resolution="instantaneous rate mm/h",
                    detail=(
                        "rain_sel is NULL for most AWS rows, so the rate is left "
                        "missing for those records rather than assuming a window."
                    ),
                )
            )

    # -- processed artifact -------------------------------------------------
    stamp = (min(retrieved) if retrieved else datetime.now(tz=timezone.utc)).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    processed_path = processed_dir / f"imd_ogc_stations_{stamp}.json"
    processed_path.write_text(
        json.dumps(
            {
                "dataset": "IMD GeoServer OGC station observations",
                "data_class": DataClass.OBSERVATION,
                "is_synthetic": False,
                "attribution": IMD_OGC_ATTRIBUTION,
                "bbox": list(target_bbox),
                "layers": list(layers),
                "n_records": len(observations),
                "records": observations,
            },
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )

    # -- readiness ----------------------------------------------------------
    # Surface observations map to none of the 12 satellite/upper-air channels
    # directly; `elevation` is DEM-derived and comes from the terrain stack.
    available_channels: list[str] = []
    readiness = assess_readiness(
        dataset_name="IMD GeoServer OGC station observations (Uttarakhand)",
        is_synthetic=False,
        n_records=len(observations),
        n_stations=len({o["station_id"] for o in observations}),
        available_channels=available_channels,
        distinct_times=len({o["valid_time"] for o in observations}),
    )

    # -- provenance ---------------------------------------------------------
    quality = QualityReport(
        n_values=len(observations),
        n_missing=sum(1 for o in observations if o["rainfall_mm_per_h"] is None),
    )
    quality.warn(
        "station snapshot: not a contiguous time series, so it cannot supply the "
        "6-frame history the model consumes"
    )
    # ------------------------------------------- phase 8.3 integrity fields
    # The *raw* downloaded bytes are the source of record; the processed JSON
    # is derived from them. `source_path` and `source_sha256` always describe
    # the same file, so re-hashing the path reproduces the digest exactly.
    raw_digests = _raw_digests(raw_files)
    primary_raw = next((p for p in raw_files if Path(p).name in raw_digests), None)
    source_digest = raw_digests.get(Path(primary_raw).name) if primary_raw else None
    coverage_string = _coverage_string(target_bbox)
    provenance = DatasetProvenance(
        source="IMD GeoServer OGC",
        product="+".join(layers),
        data_class=DataClass.OBSERVATION,
        acquired_at=min(retrieved) if retrieved else datetime.now(tz=timezone.utc),
        valid_from=min(retrieved) if retrieved else datetime.now(tz=timezone.utc),
        valid_to=max(retrieved) if retrieved else None,
        path=str(processed_path),
        checksum=file_sha256(processed_path),
        min_lon=target_bbox[0],
        min_lat=target_bbox[1],
        max_lon=target_bbox[2],
        max_lat=target_bbox[3],
        crs="EPSG:4326",
        source_resolution="point observations (AWS/SYNOP stations)",
        processed_resolution=f"point; canonical grid is {settings.grid.res_km:g} km",
        source_cadence="AWS: day resolution; SYNOP: 3-hourly (WMO)",
        processed_cadence="none (point data, not gridded)",
        n_frames=len({o["valid_time"] for o in observations}),
        quality=quality,
        resampling=resampling,
        processing_config={
            "bbox": list(target_bbox),
            "layers": {k: v for k, v in per_layer.items()},
            "readiness_verdict": readiness.verdict,
        },
        attribution=IMD_OGC_ATTRIBUTION,
        license_note=(
            "IMD data. Redistribution should cite IMD; station identifiers are "
            "public in the source service."
        ),
        # ------------------------------------------- phase 8.3 integrity fields
        # The *raw* downloaded bytes are the source of record; the processed
        # JSON is derived from them. Both are hashed from the files on disk.
        source_sha256=source_digest,
        source_path=str(primary_raw) if primary_raw else None,
        source_bytes_available=source_digest is not None,
        provider="India Meteorological Department (IMD)",
        channel_units=IMD_OGC_CHANNEL_UNITS,
        coverage=coverage_string,
    )

    if persist:
        with get_session_factory().scope() as session:
            _row, was_created = record_dataset_provenance(session, provenance)
            # `created` collides with a LogRecord attribute, so rename the key.
            logger.info(
                "provenance recorded",
                extra={"new_row": was_created, "origin": provenance.source},
            )

    return AcquisitionOutcome(
        provenance=provenance,
        readiness=readiness,
        observations=observations,
        raw_files=raw_files,
        processed_path=processed_path,
        per_layer=per_layer,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sihps-acquire",
        description=(
            "Download genuine IMD station observations from the public OGC service, "
            "store raw and processed artifacts separately, and record provenance."
        ),
    )
    parser.add_argument(
        "--bbox",
        default=None,
        help="min_lon,min_lat,max_lon,max_lat (default: the configured canonical grid).",
    )
    parser.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS),
        help="Comma-separated OGC layer names to acquire.",
    )
    parser.add_argument("--out-dir", default=None, help="Artifact root directory.")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Download and process but do not write a provenance row.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    bbox = None
    if args.bbox:
        parts = [float(p) for p in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
        bbox = (parts[0], parts[1], parts[2], parts[3])
    layers = [name.strip() for name in args.layers.split(",") if name.strip()]

    outcome = acquire_dataset(
        bbox=bbox,
        layers=layers,
        out_dir=args.out_dir,
        persist=not args.no_persist,
        timeout=args.timeout,
    )
    print("=" * 74)
    print("SIHPS REAL-DATA ACQUISITION")
    print("=" * 74)
    print("source          :", outcome.provenance.source)
    print("product         :", outcome.provenance.product)
    print("data class      :", outcome.provenance.data_class)
    print("records         :", len(outcome.observations))
    print("acquired at     :", outcome.provenance.acquired_at.isoformat())
    print("processed sha256:", outcome.provenance.checksum[:16], "...")
    print("readiness       :", outcome.readiness.verdict)
    print("can train model :", outcome.readiness.can_train)
    for note in outcome.readiness.notes:
        print("  -", note)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
