"""Training data gate: the Phase 8 policy enforced at the training boundary.

Why this module exists
----------------------
:func:`app.training.train.load_training_cubes` has exactly one behaviour: it
produces SIHPS-generated cubes (either from a demo directory or from the
synthetic connector). There is no real-data branch. Meanwhile ``--observational``
flips ``is_synthetic`` to ``False`` on that same synthetic data, which would
label generated fields as observations. This module closes that gap.

The contract
------------
:func:`describe_dataset` reads a dataset's identity from the artefacts
themselves - each cube's :class:`~app.ingestion.base.Provenance` records, its
``metadata`` and its actual valid times. It never infers authenticity from a
filename, a directory name, or a tensor shape. A cube whose provenance says
``is_synthetic=True`` is synthetic, whatever it is called.

:func:`enforce_training_gate` then applies the Phase 8 policy to the exact dataset
and configuration a run will consume, **before** any model, optimiser or
checkpoint exists.

Two modes, kept strictly apart
------------------------------
``production``
    Real data only. Requires real provenance, every required channel, complete
    labels, a declared spatial extent, a genuine six-frame 30-minute cadence, and
    a passing CAPE/IWV validation report.
``synthetic_demo``
    The existing generator, explicitly requested, loudly labelled, and
    **forbidden from writing a production checkpoint**.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.grid import FRAME_MINUTES
from app.ingestion.base import ObservationCube, SourceFile
from app.ingestion.realtime.provenance import verify_source_hash
from app.ingestion.realtime.readiness import (
    REQUIRED_CHANNELS,
    REQUIRED_LABELS,
    assess_readiness,
)
from app.logging_conf import get_logger
from app.physics import CHANNELS as _CHANNELS

logger = get_logger("training.gate")

__all__ = [
    "PRODUCTION",
    "SYNTHETIC_DEMO",
    "SYNTHETIC_EVAL",
    "OBSERVATIONAL",
    "DEMO_DIR_MARKER",
    "EVAL_DEMO_DIR_MARKER",
    "DatasetDescriptor",
    "EvaluationBlocked",
    "TrainingBlocked",
    "actual_spatial_coverage",
    "coverage_gap",
    "describe_dataset",
    "enforce_evaluation_gate",
    "enforce_training_gate",
    "parse_coverage_string",
    "synthetic_demo_output_dir",
    "verify_provenance",
]

PRODUCTION = "production"
SYNTHETIC_DEMO = "synthetic_demo"

#: Directory component that marks an artefact as a non-production demonstration.
#: A synthetic-demo run may only write beneath a path containing this, so a
#: generated checkpoint can never land on the production model path.
DEMO_DIR_MARKER = "synthetic-demo"



class TrainingBlocked(RuntimeError):
    """Raised when a run may not proceed.

    Carries the full report so a caller can present every failed criterion, the
    dataset identity and the required next action, rather than one opaque
    message.
    """

    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report

    def render(self) -> str:
        """Human-readable, actionable failure text."""
        report = self.report
        lines = [
            "=" * 74,
            "TRAINING BLOCKED - dataset does not satisfy the Phase 8 policy",
            "=" * 74,
            f"mode           : {report.get('mode')}",
            f"dataset        : {report.get('dataset_name')}",
            f"data class     : {report.get('data_class')}",
            f"provenance     : {report.get('provenance_summary')}",
        ]
        cadence = report.get("temporal") or {}
        if cadence:
            lines += [
                f"cadence        : {cadence.get('native_cadence_minutes')} min "
                f"(model requires {cadence.get('required_cadence_minutes')})",
                f"max 6-frame run: {cadence.get('max_contiguous_run')}",
            ]
        lines += [
            f"spatial extent : {report.get('spatial_coverage')}",
            f"channels       : {len(report.get('channels_present', []))}"
            f"/{len(REQUIRED_CHANNELS)} present",
            f"labels         : {len(report.get('labels_present', []))}"
            f"/{len(REQUIRED_LABELS)} present",
            f"CAPE validation: {report.get('cape_validation')}",
            f"IWV validation : {report.get('iwv_validation')}",
        ]
        if report.get("channels_missing"):
            lines.append("missing channels: " + ", ".join(report["channels_missing"]))
        if report.get("labels_missing"):
            lines.append("missing labels   : " + ", ".join(report["labels_missing"]))
        lines.append("-" * 74)
        lines.append("failed criteria:")
        for blocker in report.get("scientific_blockers") or ["<none reported>"]:
            lines.append(f"  - {blocker}")
        lines.append("-" * 74)
        lines.append("next action:")
        for action in report.get("next_actions") or ["<none reported>"]:
            lines.append(f"  - {action}")
        lines.append(
            "No model, optimiser or checkpoint was created. No data was "
            "interpolated, repeated or fabricated."
        )
        return "\n".join(lines)


def synthetic_demo_output_dir(base: str) -> str:
    """Return a demo-scoped output directory that cannot be a production path."""
    from pathlib import Path

    return str(Path(base) / DEMO_DIR_MARKER)



@dataclass(slots=True)
class DatasetDescriptor:
    """Explicit, typed identity of the dataset a run will consume.

    Every field is read from the artefacts themselves. Nothing is inferred from
    a path, a filename, or a tensor shape.
    """

    #: Identifier built from the provenance records, not from a filename.
    dataset_name: str
    #: Declared data class, taken from the cubes' provenance.
    data_class: str
    #: True when *every* cube's provenance declares it synthetic.
    is_synthetic: bool
    #: Channels with at least one finite value across the cubes.
    channels_present: list[str] = field(default_factory=list)
    #: Label arrays actually carried by the cubes.
    labels_present: list[str] = field(default_factory=list)
    #: Genuine valid times across all cubes, sorted and de-duplicated.
    valid_times: list[Any] = field(default_factory=list)
    #: Spatial extent string, when the cubes declare one. ``None`` when the
    #: cubes carry no explicit declaration - which blocks production training.
    spatial_coverage: str | None = None
    #: The cubes' actual raster extent, for comparison against what was requested.
    actual_coverage: dict[str, float] | None = None
    #: Uncovered part of the requested area [deg], or ``None`` when fully covered.
    coverage_gap: dict[str, float] | None = None
    #: True when an explicit coverage declaration is present and non-empty.
    coverage_declared: bool = False
    #: One line per distinct provenance record, for the audit trail.
    provenance_summary: list[str] = field(default_factory=list)
    n_cubes: int = 0
    n_records: int = 0
    n_stations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "data_class": self.data_class,
            "is_synthetic": self.is_synthetic,
            "channels_present": list(self.channels_present),
            "labels_present": list(self.labels_present),
            "n_valid_times": len(self.valid_times),
            "spatial_coverage": self.spatial_coverage,
            "actual_coverage": self.actual_coverage,
            "coverage_gap": self.coverage_gap,
            "coverage_declared": self.coverage_declared,
            "provenance_summary": list(self.provenance_summary),
            "n_cubes": self.n_cubes,
        }


def _declared_spatial_coverage(cube: ObservationCube) -> str | None:
    """The spatial extent the cube *declares*, or ``None``.

    Deliberately does **not** fall back to the cube's :class:`GridSpec`. A
    ``GridSpec`` says where a cube was placed on the model's raster; it is not
    evidence that the underlying source data covered that area. Phase 8.2
    requires an explicit coverage declaration, so the two cannot be conflated.
    """
    for key in ("spatial_coverage", "coverage", "extent"):
        value = cube.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def actual_spatial_coverage(cube: ObservationCube) -> dict[str, float]:
    """The cube's raster extent as numbers, for a real coverage comparison."""
    return {
        "min_lon": float(cube.grid.min_lon),
        "min_lat": float(cube.grid.min_lat),
        "max_lon": float(cube.grid.max_lon),
        "max_lat": float(cube.grid.max_lat),
    }


def coverage_gap(
    actual: dict[str, float], requested: dict[str, float]
) -> dict[str, float] | None:
    """Uncovered area in degrees, or ``None`` when the request is fully covered.

    A positive gap in any direction means the requested area is not fully
    supported by the data, and the shortfall is reported in degrees rather than
    rounded away.
    """
    # An uncovered strip exists where the requested box reaches *beyond* the data:
    # west/south shortfall when the data starts inside the request, east/north
    # when the request extends past the data's far edge.
    gap = {
        "west": max(0.0, actual["min_lon"] - requested["min_lon"]),
        "south": max(0.0, actual["min_lat"] - requested["min_lat"]),
        "east": max(0.0, requested["max_lon"] - actual["max_lon"]),
        "north": max(0.0, requested["max_lat"] - actual["max_lat"]),
    }
    return gap if any(value > 1e-9 for value in gap.values()) else None


def parse_coverage_string(value: str) -> dict[str, float] | None:
    """Parse ``"minLon-minLonE minLat-maxLatN"`` into a numeric box.

    Returns ``None`` when the string cannot be parsed, so an unparseable
    declaration is reported rather than silently accepted or silently dropped.
    """
    import re

    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*E"
        r"\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*N\s*$",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    min_lon, max_lon, min_lat, max_lat = (float(g) for g in match.groups())
    if min_lon > max_lon or min_lat > max_lat:
        return None
    return {
        "min_lon": min_lon, "min_lat": min_lat,
        "max_lon": max_lon, "max_lat": max_lat,
    }


def describe_dataset(cubes: Sequence[ObservationCube]) -> DatasetDescriptor:
    """Identify a dataset purely from its cubes' contents and provenance.

    Authenticity comes from ``Provenance.is_synthetic`` on every cube. A caller
    cannot make generated data look real by renaming a directory, and a real
    dataset cannot be dismissed because its filename looks unfamiliar.
    """
    import numpy as np

    from app.physics import CHANNELS

    if not cubes:
        return DatasetDescriptor(
            dataset_name="<empty>", data_class="unknown", is_synthetic=True
        )

    synthetic_flags = [
        bool(record.is_synthetic)
        for cube in cubes
        for record in (cube.provenance or [None])
        if record is not None
    ]
    # ``any``, not ``all``: a single generated cube taints the set, because one
    # fabricated frame is enough to make the run untrustworthy. No provenance at
    # all is likewise treated as synthetic - absence of evidence of a real
    # observation must never be read as evidence of one.
    is_synthetic = not synthetic_flags or any(synthetic_flags)

    channels: list[str] = []
    if cubes:
        finite = np.isfinite(np.concatenate([c.channels for c in cubes], axis=0)).any(axis=(0, 2, 3))
        channels = [spec.name for spec, ok in zip(CHANNELS, finite, strict=False) if ok]

    labels = sorted({key for cube in cubes for key in cube.labels})

    times: set[Any] = set()
    for cube in cubes:
        times.update(cube.times)
    ordered_times = sorted(times)

    summary: list[str] = []
    for cube in cubes:
        for record in cube.provenance or []:
            line = f"{record.source} / {record.product} (synthetic={record.is_synthetic})"
            if line not in summary:
                summary.append(line)

    sources = {r.source for c in cubes for r in (c.provenance or [])}
    data_class = "synthetic" if is_synthetic else "real"

    # Declared coverage comes from metadata; actual coverage comes from the
    # rasters. They are compared, never conflated.
    declared = _declared_spatial_coverage(cubes[0])
    actual = {
        "min_lon": min(actual_spatial_coverage(c)["min_lon"] for c in cubes),
        "min_lat": min(actual_spatial_coverage(c)["min_lat"] for c in cubes),
        "max_lon": max(actual_spatial_coverage(c)["max_lon"] for c in cubes),
        "max_lat": max(actual_spatial_coverage(c)["max_lat"] for c in cubes),
    }
    gap = None
    if declared is not None:
        parsed = parse_coverage_string(declared)
        if parsed is not None:
            gap = coverage_gap(actual, parsed)

    return DatasetDescriptor(
        dataset_name="+".join(sorted(sources)) or "unnamed",
        data_class=data_class,
        is_synthetic=is_synthetic,
        channels_present=channels,
        labels_present=labels,
        valid_times=ordered_times,
        spatial_coverage=declared,
        actual_coverage=actual,
        coverage_gap=gap,
        coverage_declared=declared is not None,
        provenance_summary=summary,
        n_cubes=len(cubes),
        n_records=sum(c.n_frames for c in cubes),
    )



def _next_actions(
    readiness: Any, descriptor: DatasetDescriptor, mode: str
) -> list[str]:
    """Concrete, ordered steps to unblock the run."""
    actions: list[str] = []
    if descriptor.is_synthetic:
        actions.append(
            "This dataset is synthetic (every Provenance record says so). Production "
            "training requires real data: acquire it from a registered source and "
            "carry its Provenance records through ingestion."
        )
        actions.append(
            "To exercise the pipeline without real data, run in synthetic-demo mode "
            f"(--mode {SYNTHETIC_DEMO}), which cannot write a production checkpoint."
        )
        return actions
    missing = readiness.channels_missing
    if missing:
        actions.append(
            f"Supply the {len(missing)} missing model channel(s): "
            + ", ".join(missing[:6])
            + ("..." if len(missing) > 6 else "")
            + ". They are never zero-filled; the sources that provide them are "
            "listed in readiness.blockers."
        )
    missing_labels = readiness.labels_missing
    if missing_labels:
        actions.append(
            f"Provide the missing target(s): {', '.join(missing_labels)}. Without "
            "labels there is no supervised signal."
        )
    if readiness.temporal is None:
        actions.append(
            "Pass the dataset's genuine valid times so the 30-minute cadence "
            "contract can be measured."
        )
    elif not readiness.temporally_feasible:
        cadence = readiness.temporal.get("native_cadence_minutes")
        actions.append(
            f"The data are {cadence:g}-minute, but the model needs six consecutive "
            "30-minute frames. Obtain a native 30-minute source (INSAT imagery is "
            "the realistic candidate); do not repeat or interpolate the existing "
            "fields, which would fabricate observations."
        )
    if readiness.spatial_coverage is None:
        actions.append(
            "Declare the dataset's spatial coverage explicitly (metadata key "
            "'spatial_coverage', e.g. '77.5-80.5E 29.0-31.5N'). A GridSpec alone is "
            "not accepted, because it records placement rather than data extent."
        )
    elif descriptor.coverage_gap:
        gap = descriptor.coverage_gap
        actions.append(
            "Re-request the dataset for the declared area: it is currently "
            "uncovered by "
            + ", ".join(f"{s} {v:g} deg" for s, v in gap.items() if v > 0)
            + "."
        )
    elif readiness.spatial_coverage:
        declared_box = parse_coverage_string(readiness.spatial_coverage)
        if declared_box is None:
            actions.append(
                f"The spatial declaration {readiness.spatial_coverage!r} could not be "
                "parsed as 'minLon-maxLonE minLat-maxLatN', so actual versus requested "
                "coverage could not be compared."
            )
    if readiness.cape_validation is None or readiness.iwv_validation is None:
        actions.append(
            "Run the CAPE and IWV validation (app.ingestion.realtime.validation) "
            "and supply the resulting status; unvalidated derived features block "
            "training."
        )
    for name, status in (("CAPE", readiness.cape_validation), ("IWV", readiness.iwv_validation)):
        if status is not None and status != "pass":
            detail = (
                "no independent reference implementation was available"
                if status == "no_reference_available"
                else "at least one check failed against its reference"
            )
            actions.append(f"Resolve the {name} validation failure ({detail}).")
    if not actions:
        actions.append("No action required; the dataset satisfies the policy.")
    return actions



def enforce_training_gate(
    cubes: Sequence[ObservationCube],
    *,
    mode: str,
    output_dir: str,
    seq_len: int = 6,
    frame_minutes: int = FRAME_MINUTES,
    cape_validation: str | None = None,
    iwv_validation: str | None = None,
) -> dict[str, Any]:
    """Apply the Phase 8 policy to the dataset a run will consume.

    Returns the readiness report on success. Raises :class:`TrainingBlocked` -
    carrying the full report - on any failure. Must be called **before** a
    model, optimiser or checkpoint is created, so a blocked run leaves no
    artefacts behind.

    Parameters
    ----------
    mode:
        ``production`` or ``synthetic_demo``. There is no third option, and no
        flag that lets synthetic data into production.
    output_dir:
        Where the run intends to write. Checked so a synthetic-demo run cannot
        target a production path.
    frame_minutes:
        The cadence the model requires. Kept as a parameter so the gate and the
        model's window geometry cannot silently diverge; readiness measures
        against the project's canonical cadence, and this value is checked
        against it.
    """
    if mode not in (PRODUCTION, SYNTHETIC_DEMO):
        raise ValueError(f"unknown training mode {mode!r}; expected {PRODUCTION} or {SYNTHETIC_DEMO}")
    if frame_minutes != FRAME_MINUTES:
        # A run claiming a cadence the grid module does not define would let the
        # gate pass against a contract the model does not actually use.
        raise ValueError(
            f"frame_minutes={frame_minutes} does not match the project's canonical "
            f"cadence of {FRAME_MINUTES} minutes (app.grid.FRAME_MINUTES)"
        )

    descriptor = describe_dataset(cubes)
    readiness = assess_readiness(
        dataset_name=descriptor.dataset_name,
        is_synthetic=descriptor.is_synthetic,
        n_records=descriptor.n_records,
        n_stations=descriptor.n_stations,
        available_channels=descriptor.channels_present,
        valid_times=descriptor.valid_times,
        data_class="observation" if not descriptor.is_synthetic else "synthetic",
        spatial_coverage=descriptor.spatial_coverage,
        required_labels=REQUIRED_LABELS,
        available_labels=descriptor.labels_present,
        cape_validation=cape_validation,
        iwv_validation=iwv_validation,
        required_frames=seq_len,
    )
    report = {
        "mode": mode,
        "dataset_name": descriptor.dataset_name,
        "data_class": descriptor.data_class,
        "provenance_summary": descriptor.provenance_summary,
        "descriptor": descriptor.to_dict(),
        "temporal": readiness.temporal,
        "spatial_coverage": readiness.spatial_coverage,
        "actual_coverage": descriptor.actual_coverage,
        "coverage_gap": descriptor.coverage_gap,
        "coverage_declared": descriptor.coverage_declared,
        "channels_present": readiness.channels_present,
        "channels_missing": readiness.channels_missing,
        "labels_present": readiness.labels_present,
        "labels_missing": readiness.labels_missing,
        "cape_validation": readiness.cape_validation,
        "iwv_validation": readiness.iwv_validation,
        "technically_permitted": readiness.technically_permitted,
        "scientifically_permitted": readiness.scientifically_permitted,
        "scientific_blockers": readiness.scientific_blockers,
    }

    # Coverage is a gate-level concern: readiness only sees the declared string,
    # while actual-versus-requested comparison needs the descriptor's rasters.
    if not descriptor.coverage_declared:
        report["scientific_blockers"] = list(report["scientific_blockers"]) + [
            "the dataset carries no explicit spatial coverage declaration; a "
            "GridSpec says where a cube was placed on the model raster, not that "
            "the source data covered that area"
        ]
    elif descriptor.coverage_gap:
        gap = descriptor.coverage_gap
        report["scientific_blockers"] = list(report["scientific_blockers"]) + [
            "the declared coverage exceeds the actual data extent: uncovered by "
            + ", ".join(
                f"{side} {value:g} deg" for side, value in gap.items() if value > 0
            )
        ]
    elif parse_coverage_string(descriptor.spatial_coverage or "") is None:
        report["scientific_blockers"] = list(report["scientific_blockers"]) + [
            f"the spatial declaration {descriptor.spatial_coverage!r} could not be "
            "parsed, so actual versus requested coverage is unverified"
        ]

    if mode == SYNTHETIC_DEMO:
        # A demo run is allowed to proceed on generated data, but only when the
        # caller asked for it explicitly and it cannot touch a production path.
        if not descriptor.is_synthetic:
            raise TrainingBlocked(
                "synthetic-demo mode was requested but the dataset is real; use "
                "production mode so it is validated and recorded as operational",
                report={**report, "next_actions": [
                    "Re-run with --mode production; real data must be validated, "
                    "not demonstrated."
                ]},
            )
        if DEMO_DIR_MARKER not in output_dir.replace("\\", "/").split("/"):
            raise TrainingBlocked(
                f"synthetic-demo mode may not write to {output_dir!r}: the path must "
                f"contain a {DEMO_DIR_MARKER!r} component so a generated checkpoint can "
                "never overwrite the production model",
                report={**report, "next_actions": [
                    f"Use a demo-scoped output path, e.g. "
                    f"{synthetic_demo_output_dir(output_dir)}"
                ]},
            )
        report["next_actions"] = [
            "Proceeding as a clearly labelled synthetic demonstration. No skill "
            "claim is made and the production checkpoint is untouched."
        ]
        return report

    # Production mode.
    report["next_actions"] = _next_actions(readiness, descriptor, mode)
    if descriptor.is_synthetic:
        report["next_actions"] = _next_actions(readiness, descriptor, mode)
    if not descriptor.coverage_declared:
        report["next_actions"] = [
            "Declare the dataset's spatial coverage explicitly (metadata key "
            "'spatial_coverage', e.g. '77.5-80.5E 29.0-31.5N'). A GridSpec alone is "
            "not accepted, because it records placement rather than data extent.",
            *report["next_actions"],
        ]
    elif descriptor.coverage_gap:
        gap = descriptor.coverage_gap
        report["next_actions"] = [
            "Re-request the dataset for the declared area: it is currently "
            "uncovered by "
            + ", ".join(f"{s} {v:g} deg" for s, v in gap.items() if v > 0)
            + ".",
            *report["next_actions"],
        ]
    if not readiness.training_permitted or report["scientific_blockers"]:
        raise TrainingBlocked(
            "the dataset does not satisfy the Phase 8 training policy; "
            f"{len(report['scientific_blockers'])} criterion/criteria failed",
            report=report,
        )
    return report



# --------------------------------------------------------------------------- #
# Phase 8.2 - evaluation integrity and provenance enforcement
# --------------------------------------------------------------------------- #
#: Evaluation modes. ``observational`` requires verified real provenance; the
#: other is a labelled demonstration that cannot update skill summaries.
OBSERVATIONAL = "observational"
SYNTHETIC_EVAL = "synthetic_demo"

EVAL_DEMO_DIR_MARKER = "synthetic-demo"


class EvaluationBlocked(RuntimeError):
    """Raised when an evaluation may not be reported as observational.

    Carries the verification report so the failure is inspectable rather than a
    bare message.
    """

    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report

    def render(self) -> str:
        report = self.report
        lines = [
            "=" * 74,
            "EVALUATION BLOCKED - dataset fails the observational integrity policy",
            "=" * 74,
            f"requested mode : {report.get('requested_mode')}",
            f"data class     : {report.get('data_class')}",
            f"dataset        : {report.get('dataset_name')}",
        ]
        lines.append("provenance checks:")
        for check in report.get("checks", []):
            mark = "ok  " if check["passed"] else "FAIL"
            lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
        lines.append("-" * 74)
        lines.append("next action:")
        for action in report.get("next_actions", []) or ["<none reported>"]:
            lines.append(f"  - {action}")
        lines.append("No metrics were produced or written.")
        return "\n".join(lines)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_provenance(cubes: Sequence[ObservationCube]) -> list[dict[str, Any]]:
    """Verify each cube's provenance chain. Fails closed on anything missing.

    Every check is about *evidence*, not about a caller's assertion. Missing
    provenance is a failure, not a neutral result, because an unverifiable
    dataset cannot support an observational claim.
    """
    checks: list[dict[str, Any]] = []
    if not cubes:
        return [_check("cubes_present", False, "no cubes were supplied")]

    missing_provenance = [i for i, c in enumerate(cubes) if not c.provenance]
    checks.append(
        _check(
            "provenance_present",
            not missing_provenance,
            "every cube carries Provenance records"
            if not missing_provenance
            else f"{len(missing_provenance)} cube(s) carry no Provenance record; "
            "provenance is required and is never inferred",
        )
    )
    records = [r for c in cubes for r in (c.provenance or [])]
    checks.append(
        _check(
            "source_identity",
            all(r.source.strip() and r.product.strip() for r in records) if records else False,
            "every record names both a source and a product"
            if records
            else "no provenance records to identify",
        )
    )
    checks.append(
        _check(
            "attribution",
            all(r.attribution.strip() for r in records) if records else False,
            "every record carries attribution text"
            if records
            else "no provenance records to attribute",
        )
    )
    checks.append(
        _check(
            "observation_timestamps",
            all(r.valid_from is not None for r in records) if records else False,
            "every record declares a valid_from observation time"
            if records
            else "no provenance records",
        )
    )

    # Contradiction check: a record claiming real data while carrying synthetic
    # attribution, or vice versa.
    contradictions = [
        f"{r.source}: is_synthetic={r.is_synthetic} with attribution "
        f"{r.attribution[:40]!r}"
        for r in records
        if "synthetic" in r.attribution.lower() and not r.is_synthetic
    ]
    checks.append(
        _check(
            "synthetic_status_consistent",
            not contradictions,
            "synthetic status agrees with attribution on every record"
            if not contradictions
            else "contradictory records: " + "; ".join(contradictions),
        )
    )

    # Channel units: the cube must declare the physical units it claims, or the
    # channel cannot be interpreted at all.
    undeclared_units = [
        name
        for name, spec in _CHANNEL_UNITS
        if name not in _declared_channel_names(cubes)
    ]
    checks.append(
        _check(
            "channel_units_declared",
            not undeclared_units,
            "channel units are declared for every present channel"
            if not undeclared_units
            else f"{len(undeclared_units)} channel(s) have no declared unit metadata",
        )
    )

    spatial = _declared_spatial_coverage(cubes[0]) if cubes else None
    checks.append(
        _check(
            "spatial_extent_declared",
            spatial is not None,
            f"spatial coverage declared: {spatial}"
            if spatial
            else "no explicit spatial coverage declaration; a GridSpec is not one",
        )
    )

    # Phase 8.3/8.4: cryptographic verification of the original source bytes.
    # Every file in the manifest is checked, not just the primary one, so a
    # tampered secondary file (a DEM, a second INSAT granule) cannot slip through.
    # `require_bytes` stays True for observational runs; a synthetic demo has no
    # source file to hash and reports "not applicable" rather than failing.
    require_bytes = all(not r.is_synthetic for c in cubes for r in (c.provenance or []))
    hash_results = []
    for cube in cubes:
        for record in cube.provenance or []:
            # Phase 8.4: prefer the full manifest; fall back to the single
            # primary file for a pre-8.4 record, which reads as unknown.
            files = record.source_files or []
            if not files and (record.source_sha256 or record.source_path):
                files = [
                    SourceFile(
                        path=record.source_path or "",
                        sha256=record.source_sha256,
                        provider=record.provider,
                        bytes_available=record.source_bytes_available,
                    )
                ]
            for entry in files:
                result = verify_source_hash(
                    entry.sha256, entry.path, require_bytes=require_bytes
                )
                hash_results.append((entry, result))
    failed_hashes = [r for _, r in hash_results if not r.verified and require_bytes]
    if require_bytes:
        if not hash_results:
            checks.append(
                _check(
                    "source_hash_verified",
                    False,
                    "no source file is recorded, so no bytes can be verified; a "
                    "dataset with no manifest is unknown, not verified",
                )
            )
        else:
            checks.append(
                _check(
                    "source_hash_verified",
                    not failed_hashes,
                    f"all {len(hash_results)} contributing source file(s) match "
                    "their recorded SHA-256"
                    if not failed_hashes
                    else "; ".join(f"{r.status}: {r.detail}" for r in failed_hashes[:3]),
                )
            )
    else:
        checks.append(
            _check(
                "source_hash_verified",
                True,
                "not applicable: the data are synthetic and have no source file "
                "to hash (a synthetic run is never observational evidence)",
            )
        )
    return checks



#: ``(channel name, unit)`` for every model channel, used by provenance checks.
_CHANNEL_UNITS: tuple[tuple[str, str], ...] = tuple(
    (spec.name, spec.unit) for spec in _CHANNELS
)


def _declared_channel_names(cubes: Sequence[ObservationCube]) -> set[str]:
    """Channel names whose unit is declared somewhere in the cube metadata."""
    declared: set[str] = set()
    for cube in cubes:
        units = cube.metadata.get("channel_units")
        if isinstance(units, (dict, list, tuple)):
            declared.update(str(name) for name in units)
    return declared


def enforce_evaluation_gate(
    cubes: Sequence[ObservationCube],
    *,
    requested_mode: str,
    claimed_is_synthetic: bool | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Verify that a dataset may be reported as observational evidence.

    The caller's ``is_synthetic`` claim is recorded but **never trusted**: the
    verdict comes from the cubes' own provenance. A claim that contradicts the
    evidence fails closed.

    Parameters
    ----------
    requested_mode:
        ``observational`` or ``synthetic_demo``.
    claimed_is_synthetic:
        What the caller asserted. Used only to detect contradiction.
    output:
        Where a report would be written, when the caller supplied one.
    """
    if requested_mode not in (OBSERVATIONAL, SYNTHETIC_EVAL):
        raise ValueError(
            f"unknown evaluation mode {requested_mode!r}; expected "
            f"{OBSERVATIONAL} or {SYNTHETIC_EVAL}"
        )

    descriptor = describe_dataset(cubes)
    checks = verify_provenance(cubes)
    failed = [c for c in checks if not c["passed"]]

    report: dict[str, Any] = {
        "requested_mode": requested_mode,
        "claimed_is_synthetic": claimed_is_synthetic,
        "data_class": descriptor.data_class,
        "verified_is_synthetic": descriptor.is_synthetic,
        "dataset_name": descriptor.dataset_name,
        "provenance_summary": descriptor.provenance_summary,
        "spatial_coverage": descriptor.spatial_coverage,
        "actual_coverage": descriptor.actual_coverage,
        "coverage_gap": descriptor.coverage_gap,
        "checks": checks,
        "n_failed_checks": len(failed),
    }

    # Contradiction between the caller's claim and the evidence.
    contradiction = (
        claimed_is_synthetic is not None
        and bool(claimed_is_synthetic) != descriptor.is_synthetic
    )
    report["claim_contradicts_provenance"] = contradiction

    if requested_mode == SYNTHETIC_EVAL:
        if not descriptor.is_synthetic:
            report["next_actions"] = [
                "This dataset is real, so it must be evaluated in observational "
                "mode where its provenance is verified and recorded."
            ]
            raise EvaluationBlocked(
                "synthetic-demo evaluation was requested for a real dataset; real "
                "data must go through observational mode",
                report=report,
            )
        if output is not None and EVAL_DEMO_DIR_MARKER not in output.replace("\\", "/").split("/"):
            report["next_actions"] = [
                f"Write synthetic-demo reports beneath a '{EVAL_DEMO_DIR_MARKER}' "
                f"directory, e.g. {output}/{EVAL_DEMO_DIR_MARKER}"
            ]
            raise EvaluationBlocked(
                f"synthetic-demo evaluation may not write to {output!r}: the path "
                f"must contain a {EVAL_DEMO_DIR_MARKER!r} component so a synthetic "
                "report can never overwrite an observational one",
                report=report,
            )
        report["next_actions"] = [
            "Proceeding as a labelled synthetic demonstration. These scores may "
            "not update official skill summaries or model selection."
        ]
        return report

    # Observational mode: every check must pass and the evidence must be real.
    if descriptor.is_synthetic:
        report["next_actions"] = [
            "The cubes' own Provenance records declare them synthetic, so no "
            "observational evaluation is possible regardless of the caller's flag. "
            "Acquire real data and carry its provenance through ingestion.",
            *(
                [
                    f"The caller claimed is_synthetic={claimed_is_synthetic!r}; the "
                    f"evidence says {descriptor.is_synthetic!r} and the evidence wins."
                ]
                if contradiction
                else []
            ),
        ]
    elif contradiction:
        report["next_actions"] = [
            f"The caller claimed is_synthetic={claimed_is_synthetic!r} but the "
            f"provenance says {descriptor.is_synthetic!r}. The evidence wins; fix "
            "the claim or the data."
        ]
    elif failed:
        report["next_actions"] = [
            f"{len(failed)} provenance check(s) failed; see the report above."
        ]
    else:
        report["next_actions"] = [
            "Provenance verified; the evaluation may be recorded as observational."
        ]

    if descriptor.is_synthetic or contradiction or failed:
        raise EvaluationBlocked(
            "the dataset cannot be reported as observational evidence",
            report=report,
        )
    return report
