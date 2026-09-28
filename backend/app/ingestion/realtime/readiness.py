"""Dataset readiness: can an acquired dataset actually train the model?

This is the gate Phase 6 requires. Rather than forcing an incomplete real
dataset through a multivariate nowcasting network, this module compares what
was actually acquired against what
:class:`~app.models.network.MultiTaskNowcastNet` genuinely needs, and returns a
verdict plus a specific list of what is missing.

The requirements come from the real contracts, not from assumptions:

* input: ``(B, T_in, C=12, H, W)`` with ``T_in = 6`` and a 30-minute cadence,
  channels as defined in :data:`app.physics.CHANNELS`;
* targets: thunderstorm, rain_class, cloudburst, flood, flood_soft over
  ``T_out`` frames;
* frames must be a *contiguous time series*, not a per-station snapshot.

Verdicts
--------
``ready``
    Every required channel and target is present over a contiguous series.
``partial``
    Real data exists but cannot fill the full contract; the reason is listed.
``blocked``
    Nothing acquired can train the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.grid import FRAME_MINUTES
from app.ingestion.realtime.temporal import assess_temporal_feasibility
from app.physics import CHANNELS, N_CHANNELS

__all__ = [
    "READINESS_VERDICTS",
    "DatasetReadiness",
    "assess_readiness",
    "REQUIRED_CHANNELS",
    "REQUIRED_LABELS",
]

#: The model's required model channels, in the order the network expects.
REQUIRED_CHANNELS: tuple[str, ...] = tuple(spec.name for spec in CHANNELS)

#: The multi-task heads the network is trained against. A dataset lacking any of
#: these cannot produce a legitimate supervised target (Phase 8).
REQUIRED_LABELS: tuple[str, ...] = (
    "thunderstorm",
    "rain_class",
    "cloudburst",
    "flood",
    "flood_soft",
)

READINESS_VERDICTS = ("ready", "partial", "blocked")

#: Which required channels are satellite-only and therefore unobtainable from


@dataclass(slots=True)
class DatasetReadiness:
    """Whether an acquired dataset can train the model, and why not."""

    verdict: str
    dataset_name: str
    is_synthetic: bool
    n_records: int
    n_stations: int
    #: Required channels that are present with usable data.
    channels_present: list[str] = field(default_factory=list)
    channels_missing: list[str] = field(default_factory=list)
    #: Channel -> why it is unavailable, for the report.
    blockers: dict[str, str] = field(default_factory=dict)
    #: Contiguous frames actually available (0 when the data is a snapshot).
    n_contiguous_frames: int = 0
    required_frames: int = 6
    notes: list[str] = field(default_factory=list)
    # ---------------------------------------------------------------- phase 8
    #: Full temporal-feasibility report (cadence, gaps, contiguity). Phase 8.
    temporal: dict[str, Any] | None = None
    #: Human-readable spatial extent, when the source declares one.
    spatial_coverage: str | None = None
    #: Required training targets, and which of them are actually present.
    required_labels: list[str] = field(default_factory=list)
    labels_present: list[str] = field(default_factory=list)
    labels_missing: list[str] = field(default_factory=list)
    #: Validation status of the derived features (``pass`` / ``fail`` /
    #: ``no_reference_available``), or ``None`` when validation was not run.
    cape_validation: str | None = None
    iwv_validation: str | None = None
    #: Blockers specific to scientific and temporal permission.
    scientific_blockers: list[str] = field(default_factory=list)

    @property
    def can_train(self) -> bool:
        """Shape-level verdict from Phase 6. Kept for backward compatibility.

        This is *not* the training gate - use :attr:`training_permitted`, which
        additionally requires a feasible cadence and a passing validation report.
        """
        return self.verdict == "ready"

    @property
    def validation_reported(self) -> bool:
        """Whether feature validation results were supplied at all."""
        return self.cape_validation is not None or self.iwv_validation is not None

    @property
    def validation_passed(self) -> bool:
        """True only when every reported feature validated *and* all were reported."""
        reported = [s for s in (self.cape_validation, self.iwv_validation) if s is not None]
        return len(reported) == 2 and all(s == "pass" for s in reported)

    @property
    def temporally_feasible(self) -> bool:
        """Whether the dataset can supply the six-frame input window."""
        if self.temporal is None:
            return False
        return bool(self.temporal.get("feasible", False))

    @property
    def labels_complete(self) -> bool:
        return not self.required_labels or not self.labels_missing

    @property
    def spatial_declared(self) -> bool:
        """Whether a spatial extent was declared, so alignment is verifiable."""
        return bool(self.spatial_coverage and self.spatial_coverage.strip())

    @property
    def channels_complete(self) -> bool:
        """Every required model channel is present with usable data."""
        return not self.channels_missing

    @property
    def technically_permitted(self) -> bool:
        """Data-shape gate: channels, labels, a feasible cadence and frame count.

        Measured from the *real* temporal report when one exists. It deliberately
        does not reuse the legacy :attr:`can_train` heuristic, which only counts
        distinct times and therefore cannot tell a 30-minute series from a
        3-hourly one.
        """
        if not self.channels_complete or self.n_records <= 0:
            return False
        if not self.labels_complete:
            return False
        if self.temporal is None:
            # No genuine times were supplied, so the contract is unverifiable.
            return False
        return self.temporally_feasible

    @property
    def scientifically_permitted(self) -> bool:
        """True only for genuine data whose features have been validated.

        This is the gate that stops a dataset merely *shaped* like a valid input
        from being trained on: it requires real (not synthetic) data, complete
        labels, a feasible cadence, and a passing validation report.
        """
        if self.is_synthetic:
            return False
        if not self.technically_permitted:
            return False
        if not self.validation_passed:
            return False
        if not self.spatial_declared:
            return False
        return self.labels_complete

    @property
    def training_permitted(self) -> bool:
        """Whether real-data training may proceed on this dataset."""
        return self.scientifically_permitted

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "can_train": self.can_train,
            "dataset": self.dataset_name,
            "is_synthetic": self.is_synthetic,
            "n_records": self.n_records,
            "n_stations": self.n_stations,
            "channels_required": list(REQUIRED_CHANNELS),
            "n_channels_required": N_CHANNELS,
            "channels_present": list(self.channels_present),
            "channels_missing": list(self.channels_missing),
            "blockers": dict(self.blockers),
            "n_contiguous_frames": self.n_contiguous_frames,
            "required_frames": self.required_frames,
            "frame_minutes": FRAME_MINUTES,
            "notes": list(self.notes),
            # phase 8
            "temporal": self.temporal,
            "spatial_coverage": self.spatial_coverage,
            "required_labels": list(self.required_labels),
            "labels_present": list(self.labels_present),
            "labels_missing": list(self.labels_missing),
            "cape_validation": self.cape_validation,
            "iwv_validation": self.iwv_validation,
            "validation_reported": self.validation_reported,
            "validation_passed": self.validation_passed,
            "temporally_feasible": self.temporally_feasible,
            "labels_complete": self.labels_complete,
            "spatial_declared": self.spatial_declared,
            "channels_complete": self.channels_complete,
            "technically_permitted": self.technically_permitted,
            "scientifically_permitted": self.scientifically_permitted,
            "training_permitted": self.training_permitted,
            "scientific_blockers": list(self.scientific_blockers),
        }


def assess_readiness(
    *,
    dataset_name: str,
    is_synthetic: bool,
    n_records: int,
    n_stations: int,
    available_channels: Sequence[str],
    #: Distinct timestamps in the acquired data (used to test contiguity).
    distinct_times: int = 0,
    distinct_time_values: Sequence[str] = (),
    required_frames: int = 6,
    # ------------------------------------------------------------- phase 8
    #: Genuine valid times, used for the real temporal-feasibility measurement.
    #: When supplied this replaces the distinct-time-count heuristic entirely.
    valid_times: Sequence[str] = (),
    data_class: str = "observation",
    spatial_coverage: str | None = None,
    required_labels: Sequence[str] = (),
    available_labels: Sequence[str] = (),
    cape_validation: str | None = None,
    iwv_validation: str | None = None,
) -> DatasetReadiness:
    """Compare acquired data against the model's actual input contract.

    ``available_channels`` names channels with *usable* data; anything absent is
    reported with the reason it cannot be substituted. The function never
    suggests filling a gap with zeros or synthetic values.
    """
    present = [name for name in REQUIRED_CHANNELS if name in set(available_channels)]
    missing = [name for name in REQUIRED_CHANNELS if name not in set(present)]
    blockers = {name: _SURFACE_UNAVAILABLE.get(name, "not present in this dataset")
                for name in missing}

    # Contiguity: the model consumes sliding windows of `required_frames`
    # consecutive 30-minute frames, so a per-station snapshot cannot supply it.
    n_contiguous = 0
    if distinct_times:
        n_contiguous = int(min(distinct_times, required_frames))
        if distinct_times < required_frames:
            blockers["__temporal__"] = (
                f"only {distinct_times} distinct observation time(s) are available; "
                f"the model requires {required_frames} contiguous 30-minute frames. "
                "This dataset is a station snapshot, not a time series."
            )

    notes: list[str] = []
    if missing:
        notes.append(
            f"{len(missing)} of {N_CHANNELS} required model channels are absent. "
            "They are reported as missing and are never zero-filled or synthesised."
        )
    if is_synthetic:
        notes.append(
            "This dataset is synthetic; any metric computed on it describes "
            "agreement with a generator, not forecasting skill."
        )
    else:
        notes.append(
            "This dataset is genuine observational data from IMD, but surface "
            "station observations cannot supply satellite channel radiances or "
            "upper-air profiles."
        )

    can_train = not missing and bool(n_contiguous >= required_frames) and n_records > 0
    if can_train:
        verdict = "ready"
    elif n_records > 0:
        verdict = "partial"
    else:
        verdict = "blocked"

    # ------------------------------------------------------------- phase 8
    # Real temporal feasibility, measured from the actual valid times when they
    # are supplied. The distinct-time heuristic above cannot distinguish a
    # 30-minute series from a 3-hourly one, so it is reported but not relied on.
    temporal: dict[str, Any] | None = None
    if valid_times or distinct_time_values:
        temporal = assess_temporal_feasibility(
            list(valid_times) or list(distinct_time_values),
            source=dataset_name,
            data_class=data_class,
            channels_available=present,
            channels_missing=missing,
            required_frames=required_frames,
            spatial_coverage=spatial_coverage,
        ).to_dict()

    labels_present = [n for n in required_labels if n in set(available_labels)]
    labels_missing = [n for n in required_labels if n not in set(labels_present)]

    scientific_blockers: list[str] = []
    if is_synthetic:
        scientific_blockers.append(
            "synthetic data: any metric would describe agreement with a generator, "
            "not forecasting skill"
        )
    if temporal is None:
        scientific_blockers.append(
            "no genuine valid times were supplied, so the 30-minute cadence "
            "contract could not be measured"
        )
    elif not temporal["feasible"]:
        detail = (
            "the dataset cannot supply "
            f"{required_frames} consecutive 30-minute frames at its own cadence"
        )
        if temporal["cadence_matches_contract"] and temporal["max_contiguous_run"] >= required_frames:
            # The cadence and run length are fine, so the blocker must say so
            # rather than imply a timing problem that does not exist.
            detail = (
                f"the data are {temporal['native_cadence_minutes']:g}-minute and long "
                "enough, but they are not an instrument observation, so they cannot "
                "satisfy an observational training requirement"
            )
        scientific_blockers.append(f"temporal: {detail}")
    if labels_missing:
        scientific_blockers.append(f"missing required targets: {', '.join(labels_missing)}")
    if cape_validation is None or iwv_validation is None:
        scientific_blockers.append(
            "CAPE/IWV validation has not been reported; derived features must be "
            "validated against an independent reference before training"
        )
    else:
        for name, status in (("CAPE", cape_validation), ("IWV", iwv_validation)):
            if status != "pass":
                scientific_blockers.append(f"{name} validation status is {status!r}, not 'pass'")
    if spatial_coverage is None:
        # The gate layer emits the authoritative coverage message, including the
        # actual-versus-requested comparison that readiness cannot compute. Not
        # repeating it here keeps the report free of duplicate criteria.
        notes.append("spatial coverage was not declared (reported by the training gate)")

    return DatasetReadiness(
        verdict=verdict,
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        n_records=n_records,
        n_stations=n_stations,
        channels_present=present,
        channels_missing=missing,
        blockers=blockers,
        n_contiguous_frames=n_contiguous,
        required_frames=required_frames,
        notes=notes,
        temporal=temporal,
        spatial_coverage=spatial_coverage,
        required_labels=list(required_labels),
        labels_present=labels_present,
        labels_missing=labels_missing,
        cape_validation=cape_validation,
        iwv_validation=iwv_validation,
        scientific_blockers=scientific_blockers,
    )

#: surface observations. Used to explain *why* a station dataset is insufficient.
_SURFACE_UNAVAILABLE = {
    "tir1_bt": "infrared brightness temperature (10.8 um) - requires INSAT L1B/L2",
    "tir2_bt": "infrared brightness temperature (12.0 um) - requires INSAT L1B/L2",
    "wv_bt": "water-vapour brightness temperature (6.8 um) - requires INSAT",
    "vis_refl": "visible reflectance (0.65 um) - requires INSAT",
    "swir_refl": "shortwave infrared reflectance (1.6 um) - requires INSAT",
    "mir_bt": "middle infrared brightness temperature (3.9 um) - requires INSAT",
    "ctt": "cloud-top temperature - derived from INSAT infrared",
    "ctt_cooling_rate": "cloud-top cooling rate - requires a satellite time series",
    "wv_bt_anomaly": "water-vapour anomaly - requires INSAT water-vapour imagery",
    "iwv": "integrated water vapour - requires upper-air reanalysis or sounding",
    "cape": "convective available potential energy - requires upper-air profiles",
    "elevation": "DEM elevation - available from SRTM/CartoDEM, not from observations",
}
