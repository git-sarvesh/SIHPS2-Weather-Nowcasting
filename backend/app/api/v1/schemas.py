"""Request/response schemas for the v1 API.

The schemas are explicit (no free-form ``dict`` bodies) so that OpenAPI
documents the contract and FastAPI rejects malformed input before any inference
work happens. Response models stay permissive about the nested numerical detail
the risk/XAI layers emit, but every response carries the synthetic-demo
provenance block.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskField = Literal["overall", "thunderstorm", "cloudburst", "flood_risk", "compound_storm_cloudburst"]
Hazard = Literal["thunderstorm", "cloudburst", "flood"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForecastRequest(_Base):
    """Body of ``POST /api/v1/forecast``."""

    event_id: str | None = Field(
        default=None,
        description="Synthetic demo event id; omit to use the default cloudburst case.",
        examples=["evt-cloudburst-001"],
    )
    lead_hours: float | None = Field(
        default=None, gt=0.0, le=24.0, description="Lead time in hours (must match a model step)."
    )
    include_uncertainty: bool = Field(
        default=False, description="Run the MC-dropout ensemble and report spread / 90% interval."
    )
    mc_samples: int | None = Field(default=None, ge=2, le=100)


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    event_kind: str
    model_version: str
    init_time: str
    grid: dict[str, Any]
    lead_times_h: list[float]
    selected: dict[str, Any]
    fields: dict[str, dict[str, float]]
    per_step: list[dict[str, Any]]
    risk: dict[str, Any]
    uncertainty: dict[str, Any] | None = None
    demo_mode: bool
    is_synthetic: bool
    data_source: str
    attribution: str
    disclaimer: str
    accuracy_claim: str


class PointRiskRequest(_Base):
    """Body of ``POST /api/v1/risk/point``."""

    lat: float = Field(ge=-90.0, le=90.0, examples=[30.5])
    lon: float = Field(ge=-180.0, le=180.0, examples=[78.5])
    event_id: str | None = None
    lead_hours: float | None = Field(default=None, gt=0.0, le=24.0)

    @field_validator("lat", "lon")
    @classmethod
    def _reject_nonfinite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
            raise ValueError("coordinate must be a finite number")
        return value


class PointRiskResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    lat: float
    lon: float
    row: int | None
    col: int | None
    lead_hours: list[float]
    init_time: str
    hazards: dict[str, float]
    terrain_exposure: float
    flood_risk: float
    compound_storm_cloudburst: float
    overall_risk: float
    risk_category: str
    model_version: str
    demo_mode: bool
    is_synthetic: bool
    data_source: str
    attribution: str
    disclaimer: str
    accuracy_claim: str


class RiskGeoJSONRequest(_Base):
    """Body of ``POST /api/v1/risk/geojson``."""

    event_id: str | None = None
    lead_hours: float | None = Field(default=None, gt=0.0, le=24.0)
    min_category: int = Field(default=1, ge=0, le=3, description="LOW=0 .. EXTREME=3.")
    risk_field: RiskField = "overall"
    max_features: int = Field(default=2000, ge=1, le=20000)


class RiskGeoJSONResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]]
    metadata: dict[str, Any]


# --------------------------------------------------------------------------- #
# persistence / history (Phase 3)
# --------------------------------------------------------------------------- #
class ForecastRunSummary(BaseModel):
    """One entry of the paginated forecast history."""

    model_config = ConfigDict(extra="allow")

    id: int
    kind: str
    status: str
    init_time: str
    valid_time: str | None
    lead_hours: float
    n_lead_times: int
    model_version: str
    model_backend: str | None
    trained_checkpoint_loaded: bool
    is_synthetic: bool
    data_source: str | None
    risk_cells: int = 0


class ForecastHistoryResponse(BaseModel):
    """A page of forecast runs with pagination metadata."""

    total: int
    limit: int
    offset: int
    count: int = Field(description="Number of runs returned in this page.")
    has_more: bool
    runs: list[ForecastRunSummary]
    filters: dict[str, Any] = Field(default_factory=dict)


class RiskCellOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    row: int
    col: int
    lat: float
    lon: float
    thunderstorm: float
    cloudburst: float
    flood_probability: float
    flood_risk: float
    compound_storm_cloudburst: float
    terrain_exposure: float
    overall_risk: float
    risk_category: str
    uncertainty: dict[str, Any] | None = None


class ForecastDetailResponse(BaseModel):
    """A single persisted run, its provenance and (optionally) its risk cells."""

    model_config = ConfigDict(extra="allow")

    run: ForecastRunSummary
    created_at: str
    updated_at: str
    duration_seconds: float | None
    checkpoint: str | None
    checkpoint_sha256: str | None
    grid: dict[str, Any] | None
    provenance: dict[str, Any] | None
    risk_summary: dict[str, Any] | None
    error: str | None
    risk_cell_count: int
    risk_cells: list[RiskCellOut] = Field(default_factory=list)
    notice: str = (
        "Persisted record. When is_synthetic is true this is a SYNTHETIC DEMONSTRATION, "
        "not an observationally validated forecast and not an official IMD warning."
    )


class PersistForecastRequest(_Base):
    """Body of ``POST /api/v1/forecast/persist``."""

    event_id: str | None = None
    lead_hours: float | None = Field(default=None, gt=0.0, le=24.0)
    persist_risk: bool = Field(
        default=True, description="Store the per-cell risk raster alongside the run."
    )
    max_cells: int = Field(
        default=20000, ge=0, le=200000, description="Cap on stored risk cells (0 = metadata only)."
    )


class PersistForecastResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    forecast_run_id: int
    idempotency_key: str
    event_id: str | None
    init_time: str
    valid_time: str
    lead_hours: float
    model_version: str
    trained_checkpoint_loaded: bool
    is_synthetic: bool
    risk_cells: int
    notice: str


class CheckpointInfoResponse(BaseModel):
    """Active checkpoint identity, training provenance and calibration status."""

    model_config = ConfigDict(extra="allow")

    model_version: str
    backend: str
    checkpoint_path: str | None
    checkpoint_present: bool
    checkpoint_sha256: str | None
    trained_checkpoint_loaded: bool
    training_provenance: dict[str, Any] | None
    training_is_synthetic: bool | None
    training_validation_status: str | None
    calibration: dict[str, Any] | None
    calibration_present: bool
    is_synthetic: bool
    observational_validation: bool = Field(
        description="Always false: no independent observational validation exists."
    )
    validation_status: str


class ExplainRequest(_Base):
    """Body of ``POST /api/v1/explain``."""

    hazard: Hazard = "cloudburst"
    # Upper bound is enforced by the service against the actual step count.
    step: int | None = Field(default=None, ge=0)
    event_id: str | None = None
    include_consistency: bool = True
    include_what_if: bool = False
    perturbations: dict[str, float] | None = Field(
        default=None, description="Normalised channel deltas, e.g. {'iwv': 0.1, 'ctt': -0.1}."
    )


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    hazard: str
    step: int
    lead_hours: float
    attribution_result: dict[str, Any]
    gradcam_map: dict[str, Any]
    model_version: str
    physical_consistency: dict[str, Any] | None = None
    what_if: dict[str, Any] | None = None
    demo_mode: bool
    is_synthetic: bool
    data_source: str
    attribution: str
    disclaimer: str
    accuracy_claim: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ok", "degraded"]
    app_name: str
    env: str
    runtime_loaded: bool
    components: dict[str, Any]
    settings: dict[str, Any]
    demo_mode: bool
    is_synthetic: bool
    data_source: str
    attribution: str
    disclaimer: str
    accuracy_claim: str
    error: str | None = None


class ModelDescribeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_version: str
    backend: str
    describe: dict[str, Any]
    grid: dict[str, Any]
    input_shape: list[int]
    terrain_shape: list[int]
    lead_times_h: list[float]
    n_forecast_steps: int
    hazards: list[str]
    synthetic_event: dict[str, Any]
    provenance: list[dict[str, Any]]
    demo_mode: bool
    is_synthetic: bool
    data_source: str
    attribution: str
    disclaimer: str
    accuracy_claim: str


class ErrorResponse(BaseModel):
    """Uniform error body for 400/404/422/503 responses."""

    detail: str
    status_code: int
    demo_mode: bool | None = None


__all__ = [
    "CheckpointInfoResponse",
    "ErrorResponse",
    "ExplainRequest",
    "ExplainResponse",
    "ForecastDetailResponse",
    "ForecastHistoryResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastRunSummary",
    "HealthResponse",
    "ModelDescribeResponse",
    "PersistForecastRequest",
    "PersistForecastResponse",
    "PointRiskRequest",
    "PointRiskResponse",
    "RiskCellOut",
    "RiskGeoJSONRequest",
    "RiskGeoJSONResponse",
]
