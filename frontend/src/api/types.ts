/**
 * Typed response types for the SIHPS FastAPI backend.
 *
 * These mirror the backend's OpenAPI contract exactly (captured from the running
 * application). Fields the backend marks `extra="allow"` are typed loosely so a
 * newly added backend field never breaks the build, but every field the
 * dashboard actually consumes is required and typed.
 *
 * Source of truth: backend/app/api/v1/schemas.py
 */

/** Risk category names emitted by the backend risk engine. */
export type RiskCategory = 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME'

/** Hazards the model predicts. */
export type Hazard = 'thunderstorm' | 'cloudburst' | 'flood'

/** Selectable risk field for the GeoJSON layer endpoint. */
export type RiskField =
  | 'overall'
  | 'thunderstorm'
  | 'cloudburst'
  | 'flood_risk'
  | 'compound_storm_cloudburst'

/**
 * Provenance block attached to nearly every backend response.
 *
 * The dashboard surfaces these rather than hiding them: the system currently
 * runs on synthetic data and must never be mistaken for an official forecast.
 */
export interface Provenance {
  demo_mode: boolean
  is_synthetic: boolean
  data_source: string
  attribution: string
  disclaimer: string
  accuracy_claim: string
}

/**
 * Mandatory experimental-prediction disclaimer.
 *
 * Kept identical to the backend's `app.ingestion.base.EXPERIMENTAL_DISCLAIMER`
 * and asserted equal in `src/test/format.test.tsx`, so the dashboard can never
 * quietly drop or soften it.
 */
export const EXPERIMENTAL_DISCLAIMER =
  'Experimental AI prediction for research and preparedness support only. ' +
  'This is NOT an official India Meteorological Department (IMD) warning. ' +
  'Always refer to IMD for authoritative alerts.'

/** How the backend labels a data stream. Mirrors `DataClass` in app/physics. */
export type DataClass =
  | 'observation'
  | 'reanalysis'
  | 'derived'
  | 'interpolated'
  | 'synthetic'

/** Access state of one real-world data source, from `/health`. */
export interface RealSourceStatus {
  source: string
  availability:
    | 'available'
    | 'needs_credentials'
    | 'needs_manual_download'
    | 'unreachable'
    | 'metadata_only'
    | 'not_implemented'
    | 'not_probed'
  available: boolean
  reason: string
  is_synthetic: boolean
  details?: Record<string, unknown>
  manual_instructions?: string
}

/** Whether the required train/val/test year split can be built from real data. */
export interface SplitFeasibility {
  requested: Record<string, [number, number]>
  available: Record<string, [number, number]>
  feasible: boolean
  unsatisfiable: string[]
  proposed: Record<string, [number, number]>
  notes: string[]
}

/** `data_sources` block of `/health`. */
export interface DataSourcesHealth {
  any_real_available: boolean
  sources: RealSourceStatus[]
  split_feasibility: SplitFeasibility
  notes?: string[]
  error?: string
}

export interface GridInfo {
  min_lon: number
  min_lat: number
  max_lon: number
  max_lat: number
  nx: number
  ny: number
  res_km: number
  effective_res_km: { x: number; y: number }
  crs: string
  frame_minutes: number
}

export interface ComponentStatus {
  status: string
  loaded?: boolean
  [key: string]: unknown
}

export interface HealthSettings {
  env: string
  demo_mode: boolean
  model_backend: string
  model_version: string
  grid: GridInfo
  sequence_length: number
  forecast_steps: number
  lead_times_h: number[]
  mc_samples: number
  risk_weights: number[]
  risk_thresholds: number[]
}

export interface HealthResponse extends Provenance {
  status: 'ok' | 'degraded'
  app_name: string
  env: string
  runtime_loaded: boolean
  components: {
    grid: ComponentStatus & GridInfo
    terrain: ComponentStatus
    model: ComponentStatus & {
      requested_backend: string
      resolved_backend: string | null
      model_version: string
      trained_checkpoint_loaded: boolean
      input_shape?: number[]
    }
    risk_engine: ComponentStatus & { weights?: number[]; thresholds?: number[] }
    explainability: ComponentStatus
  }
  settings: HealthSettings
  error?: string | null
  demo_event?: Record<string, unknown> | null
  /** Present since Phase 3. */
  database?: { status: string; dialect?: string; current_revision?: string; error?: string }
  schedule?: {
    schedule_enabled: boolean
    schedule_requested?: boolean
    demo_mode?: boolean
    ingest_interval_minutes?: number
    batch_inference_interval_minutes?: number
    beat_schedule?: string[]
    note?: string
  }
  connectors?: {
    demo_mode: boolean
    any_live_available: boolean
    live: { name: string; available: boolean; is_synthetic: boolean; reason: string }[]
    synthetic: { name: string; available: boolean; is_synthetic: boolean; reason: string }[]
  }
  /** Phase 5: real-world source access state and historical coverage verdict. */
  data_sources?: DataSourcesHealth
}

export interface CheckpointResponse {
  model_version: string
  backend: string
  checkpoint_path: string | null
  checkpoint_present: boolean
  checkpoint_sha256: string | null
  trained_checkpoint_loaded: boolean
  training_provenance: Record<string, unknown> | null
  training_is_synthetic: boolean | null
  training_validation_status: string | null
  calibration: Record<string, unknown> | null
  calibration_present: boolean
  is_synthetic: boolean
  /** Always false: no independent observational validation exists. */
  observational_validation: boolean
  validation_status: string
}

export interface ModelConfig {
  model_version: string
  terrain_channels: number
  head_hidden: number
  use_cha: boolean
  rain_classes: string[]
  backbone: {
    in_channels: number
    encoder_width: number
    n_residual_blocks: number
    convlstm_channels: number[]
    kernel_size: number
    latent_dim: number
    dropout: number
    norm: string
    variational: boolean
    predict_steps: number
  }
}

export interface ModelDescribeResponse extends Provenance {
  model_version: string
  backend: string
  /** Architecture summary produced by `MultiTaskNowcastNet.describe()`. */
  describe: {
    model_version: string
    n_parameters: number
    config: ModelConfig
    [key: string]: unknown
  }
  grid: GridInfo
  input_shape: number[]
  terrain_shape: number[]
  lead_times_h: number[]
  n_forecast_steps: number
  hazards: string[]
  synthetic_event: Record<string, unknown>
  [key: string]: unknown
}

/** Per-hazard max / mean / p95 for one forecast step. */
export interface FieldStats {
  max: number
  mean: number
  p95: number
}

export interface ForecastStepEntry {
  step: number
  lead_hours: number
  valid_time: string
  thunderstorm: FieldStats
  cloudburst: FieldStats
  flood: FieldStats
}

export interface Hotspot {
  lat: number
  lon: number
  row: number
  col: number
  overall_risk: number
  category: RiskCategory
  thunderstorm: number
  cloudburst: number
  flood: number
  exposure: number
}

export interface ForecastResponse extends Provenance {
  event_id: string
  event_kind: string
  model_version: string
  init_time: string
  grid: GridInfo
  lead_times_h: number[]
  selected: { step: number; lead_hours: number; valid_time: string }
  fields: Record<Hazard, FieldStats>
  per_step: ForecastStepEntry[]
  risk: { lead_hours: number[]; summary: RiskSummary; thresholds: number[]; weights: number[] }
  /** Null unless `include_uncertainty` was requested. */
  uncertainty: UncertaintyResponse | null
}

export interface PointRiskResponse extends Provenance {
  lat: number
  lon: number
  row: number | null
  col: number | null
  lead_hours: number[]
  init_time: string
  hazards: Record<string, number>
  terrain_exposure: number
  flood_risk: number
  compound_storm_cloudburst: number
  overall_risk: number
  risk_category: RiskCategory
  model_version: string
  event_id: string
  grid: GridInfo
}

export interface GeoJsonFeature {
  type: 'Feature'
  geometry: { type: string; coordinates: unknown }
  properties: {
    event_type: string
    hazard: string
    risk_category: RiskCategory
    risk_category_code: number
    risk_max: number
    risk_mean: number
    area_km2: number
    n_cells: number
    lead_hours: number[]
    valid_time: string
    model_version: string
    demo_mode: boolean
    is_synthetic: boolean
    data_source: string
    attribution: string
    [key: string]: unknown
  }
}

export interface GeoJsonMetadata {
  grid: GridInfo
  n_features: number
  min_category: string
  risk_field: string
  model_version: string
  disclaimer: string
  summary: RiskSummary
  event_id: string | null
  selected_step: number
  lead_hours: number[]
  demo_mode: boolean
  is_synthetic: boolean
  data_source: string
  attribution: string
  accuracy_claim: string
}

export interface GeoJsonResponse {
  type: 'FeatureCollection'
  features: GeoJsonFeature[]
  metadata: GeoJsonMetadata
}

export interface PhysicalCheck {
  passed: boolean
  conforming_fraction?: number
  top_channels?: string[]
  [key: string]: unknown
}

export interface PhysicalConsistency {
  consistent: boolean
  score: number
  checks: Record<string, PhysicalCheck>
  violations: string[]
}

export interface WhatIfResult {
  baseline: Record<string, number>
  counterfactual: Record<string, number>
  delta: Record<string, number>
  perturbations_applied: Record<string, number>
}

export interface ExplainResponse extends Provenance {
  event_id: string
  hazard: string
  step: number
  lead_hours: number
  attribution_result: {
    hazard: string
    /** Raw Grad-CAM raster, `[height][width]`. */
    gradcam_map: number[][]
    channel_attributions: Record<string, number>
    top_channels: [string, number][]
    step: number
    model_version: string
  }
  /** Summary only - the full raster lives in `attribution_result.gradcam_map`. */
  gradcam_map: { shape: number[]; min: number; max: number; mean: number }
  model_version: string
  physical_consistency: PhysicalConsistency | null
  what_if: WhatIfResult | null
}

export interface ForecastRunSummary {
  id: number
  kind: string
  status: string
  init_time: string
  valid_time: string | null
  lead_hours: number
  n_lead_times: number
  model_version: string
  model_backend: string | null
  trained_checkpoint_loaded: boolean
  is_synthetic: boolean
  data_source: string | null
  risk_cells: number
}

export interface ForecastHistoryResponse {
  total: number
  limit: number
  offset: number
  count: number
  has_more: boolean
  runs: ForecastRunSummary[]
  filters: Record<string, unknown>
}

export interface RiskCellOut {
  id: number
  row: number
  col: number
  lat: number
  lon: number
  thunderstorm: number
  cloudburst: number
  flood_probability: number
  flood_risk: number
  compound_storm_cloudburst: number
  terrain_exposure: number
  overall_risk: number
  risk_category: RiskCategory
  uncertainty: Record<string, number> | null
}

export interface ForecastDetailResponse {
  run: ForecastRunSummary
  created_at: string
  updated_at: string
  duration_seconds: number | null
  checkpoint: string | null
  checkpoint_sha256: string | null
  grid: Record<string, unknown> | null
  provenance: Record<string, unknown> | null
  risk_summary: RiskSummary | null
  error: string | null
  risk_cell_count: number
  risk_cells: RiskCellOut[]
  notice: string
}

export interface PersistForecastResponse {
  status: string
  forecast_run_id: number
  idempotency_key: string
  event_id: string | null
  init_time: string
  valid_time: string
  lead_hours: number
  model_version: string
  trained_checkpoint_loaded: boolean
  is_synthetic: boolean
  risk_cells: number
  notice: string
}

export interface ApiErrorBody {
  detail: string
}

export interface RiskSummary {
  init_time: string
  lead_hours: number[]
  model_version: string
  disclaimer: string
  weights: Record<string, number>
  thresholds: number[]
  max_overall_risk: number
  mean_overall_risk: number
  area_fraction: Record<RiskCategory, number>
  max_by_hazard: Record<string, number>
  hotspots: Hotspot[]
}

export interface SpreadStats {
  mean_std: number
  max_std: number
}

export interface UncertaintyResponse {
  method: string
  n_samples: number
  spread_at_selected_step: Record<Hazard, SpreadStats>
  /**
   * 90% ensemble interval bounds. The backend serialises these as
   * `lower`/`upper` (see `app/services/inference.py`), not `lo`/`hi`.
   */
  interval_90_at_selected_step: Record<Hazard, { lower: number; upper: number }>
}
