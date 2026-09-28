"""Capability report: what can and cannot be evaluated right now.

This exists because Phase 5 requires an explicit account of evaluations that
*could not* be performed, rather than a set of quietly-absent numbers. It
combines three facts:

* what real data this deployment can actually read
  (:func:`~app.ingestion.realtime.sources.describe_real_sources`),
* whether the required year split can be built from that coverage, and
* whether any real dataset has actually been ingested and checksummed.

The result is intentionally blunt: with no ingested real data, *no* independent
meteorological validation has been performed, and this report says so instead
of letting a reader assume the metrics are observational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ingestion.realtime.provenance import DataClass
from app.logging_conf import get_logger

logger = get_logger("training.validation_report")

__all__ = [
    "ValidationCapabilityReport",
    "EVALUATIONS_REQUIRING_REAL_DATA",
    "build_capability_report",
]

#: Evaluations that require genuine observations. Each entry names why it is
#: currently impossible, so the report is specific rather than a blanket denial.
EVALUATIONS_REQUIRING_REAL_DATA: tuple[tuple[str, str], ...] = (
    (
        "CSI / POD / FAR against observed rain events",
        "requires an independent observed rainfall reference (IMD stations or "
        "MERA) aligned to the forecast grid; none has been ingested",
    ),
    (
        "Brier Skill Score against climatology",
        "requires a real multi-year reference forecast sample to define the "
        "climatological frequency; only synthetic labels exist",
    ),
    (
        "Reliability / calibration curves from real forecasts",
        "requires a trained checkpoint evaluated on real held-out observations",
    ),
    (
        "CRPS of a calibrated ensemble",
        "requires real target fields and a trained probabilistic checkpoint",
    ),
    (
        "Skill by storm event across the 2024-2025 test window",
        "IMDAA reanalysis ends in 2020; only MERA (rainfall, 2020-2025) covers "
        "that window, and it contains no upper-air state",
    ),
    (
        "Lead-time skill decay in the Indian monsoon regime",
        "needs a multi-season real dataset spanning train/val/test; the "
        "accessible products do not overlap enough to form one",
    ),
)


@dataclass(slots=True)
class ValidationCapabilityReport:
    """Whether independent validation is possible, and precisely why not."""

    any_real_data_available: bool
    real_datasets_ingested: int
    split_feasible: bool
    unsatisfiable_splits: list[str]
    #: ``(evaluation, blocking reason)`` pairs.
    blocked_evaluations: list[tuple[str, str]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def validation_performed(self) -> bool:
        """Independent observational validation requires ingested real data."""
        return bool(self.real_datasets_ingested) and self.split_feasible

    @property
    def status(self) -> str:
        if self.validation_performed:
            return "validation_possible"
        if self.any_real_data_available:
            return "data_accessible_but_nothing_ingested"
        return "no_real_data_accessible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "validation_performed": self.validation_performed,
            "any_real_data_available": self.any_real_data_available,
            "real_datasets_ingested": self.real_datasets_ingested,
            "split_feasible": self.split_feasible,
            "unsatisfiable_splits": list(self.unsatisfiable_splits),
            "blocked_evaluations": [
                {"evaluation": name, "blocked_because": reason}
                for name, reason in self.blocked_evaluations
            ],
            "sources": self.sources,
            "notes": list(self.notes),
            "disclaimer": (
                "No independently validated skill has been demonstrated against "
                "observations in this repository. The synthetic demo scores are "
                "agreement with a generator, not forecasting skill."
            ),
        }


def build_capability_report(
    *, real_datasets_ingested: int = 0, grid: Any | None = None
) -> ValidationCapabilityReport:
    """Assemble the capability report from live access state and ingested counts.

    ``real_datasets_ingested`` should come from counting persisted
    ``dataset_provenance`` rows whose ``data_class`` is an observation,
    reanalysis, derived or interpolated (synthetic rows do not count).
    """
    from app.ingestion.realtime.sources import describe_real_sources

    try:
        source_report = describe_real_sources(grid)
    except Exception as exc:  # noqa: BLE001 - the report must always render
        logger.warning("capability report could not probe sources", extra={"error": type(exc).__name__})
        source_report = {"any_real_available": False, "sources": [], "split_feasibility": {}}

    feasibility = source_report.get("split_feasibility") or {}
    return ValidationCapabilityReport(
        any_real_data_available=bool(source_report.get("any_real_available")),
        real_datasets_ingested=int(real_datasets_ingested),
        split_feasible=bool(feasibility.get("feasible")),
        unsatisfiable_splits=list(feasibility.get("unsatisfiable") or []),
        blocked_evaluations=list(EVALUATIONS_REQUIRING_REAL_DATA),
        sources=list(source_report.get("sources") or []),
        notes=list(feasibility.get("notes") or []),
    )

