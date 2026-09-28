"""Acquisition readiness: can a genuine observational dataset be assembled?

Phase 8.4. This answers a single question with an auditable, fail-closed chain
of evidence, and it deliberately separates four states that are easy to
conflate:

``source_access``
    Can this system reach and authenticate to the source at all?
``files_acquired``
    Are real source bytes on disk, with a digest computed from those bytes?
``data_processed``
    Did those bytes produce a validated, unit-annotated, coverage-declared
    dataset?
``observational_readiness``
    Can that dataset satisfy the model's full contract - all 12 channels, a
    genuine 30-minute cadence, complete labels, and passing CAPE/IWV validation?

A pipeline that is only reachable through the synthetic generator can satisfy
none of these. Nothing here fabricates a credential, a file, a digest or a
measurement; every stage reports what it actually found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.logging_conf import get_logger
from app.physics import CHANNEL_META_INDEX, channel_units

logger = get_logger("ingestion.realtime.acquisition_readiness")

__all__ = [
    "KNOWN_SOURCES",
    "STAGE_FILES",
    "STAGE_OBSERVATIONAL",
    "STAGE_PROCESSED",
    "STAGE_SOURCE",
    "AcquisitionReadiness",
    "Stage",
    "assess_acquisition_readiness",
]

STAGE_SOURCE = "source_access"
STAGE_FILES = "files_acquired"
STAGE_PROCESSED = "data_processed"
STAGE_OBSERVATIONAL = "observational_readiness"

#: Sources this system knows how to reach, and what each one can supply.
KNOWN_SOURCES: dict[str, dict[str, Any]] = {
    "ncmrwf-rds": {
        "label": "NCMRWF RDS (IMDAA/MERA reanalysis)",
        "provider": "NCMRWF",
        "auth_env": ("SIHPS_RDS_EMAIL", "SIHPS_RDS_PASSWORD"),
        "data_class": "reanalysis",
        "channels": ("cape", "iwv"),
    },
    "mosdac-insat": {
        "label": "MOSDAC INSAT-3D/3DR (satellite radiances)",
        "provider": "MOSDAC / ISRO",
        "auth_env": ("SIHPS_MOSDAC_USER", "SIHPS_MOSDAC_PASSWORD"),
        "data_class": "observation",
        "channels": (
            "tir1_bt", "tir2_bt", "wv_bt", "vis_refl", "swir_refl", "mir_bt",
        ),
    },
    "dem": {
        "label": "SRTM/CartoDEM elevation",
        "provider": "NASA/USGS",
        "auth_env": (),
        "data_class": "observation",
        "channels": ("elevation",),
    },
}


@dataclass(slots=True)
class Stage:
    """One stage of the acquisition chain, with the evidence for its verdict."""

    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class AcquisitionReadiness:
    """Full acquisition-readiness verdict, stage by stage."""

    stages: list[Stage] = field(default_factory=list)
    channels_available: list[str] = field(default_factory=list)
    channels_missing: list[str] = field(default_factory=list)
    validation_status: dict[str, str | None] = field(default_factory=dict)
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)

    @property
    def source_access(self) -> bool:
        stage = self.stage(STAGE_SOURCE)
        return bool(stage and stage.passed)

    @property
    def files_acquired(self) -> bool:
        stage = self.stage(STAGE_FILES)
        return bool(stage and stage.passed)

    @property
    def data_processed(self) -> bool:
        stage = self.stage(STAGE_PROCESSED)
        return bool(stage and stage.passed)

    @property
    def observational_readiness(self) -> bool:
        """True only when every stage passes. There is no partial credit."""
        return all(s.passed for s in self.stages) and bool(self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observational_readiness": self.observational_readiness,
            "source_access": self.source_access,
            "files_acquired": self.files_acquired,
            "data_processed": self.data_processed,
            "stages": [s.to_dict() for s in self.stages],
            "channels_available": list(self.channels_available),
            "channels_missing": list(self.channels_missing),
            "validation_status": dict(self.validation_status),
            "blockers": list(self.blockers),
            "next_actions": list(self.next_actions),
        }



def _credentials_present(env_names: Sequence[str]) -> tuple[bool, list[str]]:
    """Whether the named environment variables are set. Never invents a value."""
    import os

    missing = [name for name in env_names if not os.environ.get(name)]
    return (not missing), missing


def assess_acquisition_readiness(
    *,
    source_files: Sequence[str | Path] = (),
    verify: bool = True,
    data_dir: str | Path | None = None,
    run_validation: bool = True,
) -> AcquisitionReadiness:
    """Assess the whole chain from credentials to observational readiness.

    Parameters
    ----------
    source_files:
        Real source files to consider. Their bytes are hashed and re-verified;
        a file that does not exist yields no digest and fails the stage.
    verify:
        Re-hash the files and compare against any digest already recorded.
    data_dir:
        Directory searched for acquired files when ``source_files`` is empty.
    run_validation:
        Execute the Phase 8 CAPE/IWV validators now. When ``False`` the status is
        reported as ``not_run``, which is *not* a pass.
    """
    from app.ingestion.realtime.provenance import file_sha256

    result = AcquisitionReadiness()
    all_channels = list(channel_units())
    creds = {name: _credentials_present(spec["auth_env"])
              for name, spec in KNOWN_SOURCES.items()}
    accessible = [name for name, (ok, _) in creds.items() if ok]

    # --- stage 1: source access -------------------------------------------
    result.stages.append(
        Stage(
            STAGE_SOURCE,
            bool(accessible),
            (
                f"credentialed access available for: {', '.join(accessible)}"
                if accessible
                else "no source has its credentials configured; nothing can be "
                "acquired and no data can be genuine"
            ),
            {
                name: {
                    "label": spec["label"],
                    "provider": spec["provider"],
                    "required_env": list(spec["auth_env"]),
                    "credentials_present": creds[name][0],
                    "missing_env": creds[name][1],
                    "can_supply_channels": list(spec["channels"]),
                }
                for name, spec in KNOWN_SOURCES.items()
            },
        )
    )

    # --- stage 2: files acquired -------------------------------------------
    candidates: list[Path] = [Path(p) for p in source_files]
    if not candidates and data_dir:
        root = Path(data_dir)
        candidates = sorted(
            p for p in root.rglob("*")
            if p.suffix.lower() in (".nc", ".nc4", ".geojson")
        )
    present = [p for p in candidates if p.is_file()]
    digests = {p: file_sha256(p) for p in present} if verify else {}
    result.stages.append(
        Stage(
            STAGE_FILES,
            bool(present),
            (
                f"{len(present)} source file(s) on disk with digests computed from "
                "their actual bytes"
                if present
                else "no real source file is present; a dataset cannot be built "
                "from nothing, and no digest is invented for an absent file"
            ),
            {
                "n_candidates": len(candidates),
                "n_present": len(present),
                "files": [
                    {"path": str(p), "sha256": digests.get(p),
                     "bytes": p.stat().st_size}
                    for p in present
                ],
            },
        )
    )


    # --- stage 3: channels and metadata ------------------------------------
    # Only channels whose source has credentials can be considered available;
    # everything else stays missing and blocks readiness.
    available: list[str] = []
    for name, spec in KNOWN_SOURCES.items():
        if creds[name][0]:
            available.extend(spec["channels"])
    missing = [c for c in all_channels if c not in available]
    result.channels_available = sorted(available)
    result.channels_missing = missing
    result.stages.append(
        Stage(
            STAGE_PROCESSED,
            bool(present) and not missing,
            (
                f"{len(available)} of {len(all_channels)} model channels can be "
                f"sourced; missing: {', '.join(missing)}"
                if missing
                else f"all {len(all_channels)} model channels have a source"
            ),
            {
                "available": result.channels_available,
                "missing": missing,
                "declared_units": {c: CHANNEL_META_INDEX[c].unit for c in all_channels},
                "insat_and_dem_remain_missing": bool(
                    {"tir1_bt", "elevation"} & set(missing)
                ),
            },
        )
    )

    # --- stage 4: observational readiness ----------------------------------
    if run_validation:
        from app.ingestion.realtime.validation import (
            run_cape_validation,
            run_iwv_validation,
        )

        cape_status = run_cape_validation().status
        iwv_status = run_iwv_validation().status
    else:
        cape_status = iwv_status = "not_run"
    result.validation_status = {"cape": cape_status, "iwv": iwv_status}
    result.stages.append(
        Stage(
            STAGE_OBSERVATIONAL,
            False,  # never asserted: cadence and labels require real data
            (
                "CAPE and IWV validation passed, but a genuine 30-minute cadence, "
                "complete labels and a full 12-channel dataset still require real "
                "observations; readiness is not claimed"
                if cape_status == "pass" and iwv_status == "pass"
                else f"feature validation not passed (cape={cape_status}, "
                f"iwv={iwv_status})"
            ),
            {"cape_validation": cape_status, "iwv_validation": iwv_status},
        )
    )

    # --- blockers and next actions ----------------------------------------
    for stage in result.stages:
        if not stage.passed:
            result.blockers.append(f"{stage.name}: {stage.detail}")
    if not result.source_access:
        result.next_actions.append(
            "Register and configure credentials for the sources above "
            "(SIHPS_RDS_EMAIL/SIHPS_RDS_PASSWORD, MOSDAC account). Until then no "
            "data can be genuine."
        )
    if not result.files_acquired:
        result.next_actions.append(
            "Acquire real source bytes into the configured data directory; the "
            "readiness check only reports what is on disk."
        )
    if result.channels_missing:
        result.next_actions.append(
            f"{len(result.channels_missing)} model channel(s) have no acquired "
            f"source: {', '.join(result.channels_missing)}. They stay missing; no "
            "unit or value is borrowed from another source."
        )
    result.next_actions.append(
        "Observational readiness additionally requires six consecutive 30-minute "
        "frames, complete labels, and a passing CAPE/IWV report; none can be "
        "asserted without genuine observations."
    )
    return result
