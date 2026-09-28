/**
 * Typed client for the SIHPS FastAPI backend.
 *
 * Design rules:
 * - The base URL comes from `VITE_API_BASE_URL` (see `.env.example`) so the same
 *   build can point at localhost, a LAN host, or a deployed API.
 * - Every method maps 1:1 onto an existing backend route. No endpoint is invented
 *   and no response is reshaped; callers get the backend's own payload types.
 * - Failures surface as {@link ApiError} carrying the HTTP status and the
 *   backend's `detail` message, so the UI can show something specific instead of
 *   a generic failure.
 * - Nothing here fabricates data: a failed request throws, it never resolves to
 *   placeholder values.
 */

import type {
  CheckpointResponse,
  ExplainResponse,
  ForecastDetailResponse,
  ForecastHistoryResponse,
  ForecastResponse,
  GeoJsonResponse,
  HealthResponse,
  Hazard,
  ModelDescribeResponse,
  PersistForecastResponse,
  PointRiskResponse,
  RiskField,
} from './types'

/** A non-2xx response or a transport failure, with the backend's message. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  /** True when the request never reached the backend (offline, CORS, refused). */
  readonly isNetworkError: boolean

  constructor(message: string, status: number, detail: string, isNetworkError = false) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.isNetworkError = isNetworkError
  }
}

export const API_BASE_URL: string = (
  // Relative by default: in dev, Vite proxies /api to the backend on the same
  // origin, so no CORS handshake is involved. A production build must set
  // VITE_API_BASE_URL to the absolute backend URL (see .env.example).
  import.meta.env.VITE_API_BASE_URL ?? '/api/v1'
).replace(/\/+$/, '')

/** Default per-request timeout, so a hung backend cannot wedge the UI forever. */
const DEFAULT_TIMEOUT_MS = 30_000

export interface ForecastParams {
  event_id?: string | null
  lead_hours?: number | null
  include_uncertainty?: boolean
  mc_samples?: number | null
}

export interface GeoJsonParams {
  event_id?: string | null
  lead_hours?: number | null
  min_category?: 0 | 1 | 2 | 3
  risk_field?: RiskField
  max_features?: number
}

export interface PointRiskParams {
  lat: number
  lon: number
  event_id?: string | null
  lead_hours?: number | null
}

export interface ExplainParams {
  hazard?: Hazard
  step?: number | null
  event_id?: string | null
  include_consistency?: boolean
  include_what_if?: boolean
  perturbations?: Record<string, number> | null
}

export interface HistoryParams {
  limit?: number
  offset?: number
  status?: string | null
  is_synthetic?: boolean | null
  model_version?: string | null
}

export interface PersistParams {
  event_id?: string | null
  lead_hours?: number | null
  persist_risk?: boolean
  max_cells?: number
}

export interface ForecastDetailParams {
  include_risk?: boolean
  risk_limit?: number
  min_risk?: number
}

type FetchLike = typeof fetch

/**
 * Build a query string, omitting null/undefined so the backend keeps its own
 * defaults instead of receiving explicit nulls it would reject.
 */
export function buildQuery(params: Record<string, unknown>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue
    search.set(key, String(value))
  }
  const query = search.toString()
  return query ? `?${query}` : ''
}

/** Extract a human-readable message from a FastAPI error body. */
export function extractDetail(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    // FastAPI validation errors arrive as a list of objects.
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) => {
          if (item && typeof item === 'object' && 'msg' in item) {
            const loc = (item as { loc?: unknown[] }).loc
            const field = Array.isArray(loc) ? loc[loc.length - 1] : undefined
            const msg = String((item as { msg: unknown }).msg)
            return field ? `${String(field)}: ${msg}` : msg
          }
          return String(item)
        })
        .join('; ')
    }
  }
  return `Request failed with status ${status}`
}

export interface SihpsApiOptions {
  baseUrl?: string
  fetchImpl?: FetchLike
  timeoutMs?: number
}

export class SihpsApi {
  readonly baseUrl: string
  private readonly fetchImpl: FetchLike
  private readonly timeoutMs: number

  constructor(options: SihpsApiOptions = {}) {
    this.baseUrl = (options.baseUrl ?? API_BASE_URL).replace(/\/+$/, '')
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis)
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  }

  // ---------------------------------------------------------------- internals
  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${path}`
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)

    let response: Response
    try {
      response = await this.fetchImpl(url, {
        ...init,
        signal: controller.signal,
        headers: { Accept: 'application/json', ...(init.headers ?? {}) },
      })
    } catch (error) {
      // Transport-level failure: backend down, CORS rejected, DNS failure.
      const aborted = error instanceof Error && error.name === 'AbortError'
      throw new ApiError(
        aborted
          ? `Request to ${url} timed out after ${this.timeoutMs} ms`
          : `Could not reach the SIHPS backend at ${this.baseUrl}`,
        0,
        error instanceof Error ? error.message : String(error),
        true,
      )
    } finally {
      clearTimeout(timer)
    }

    if (!response.ok) {
      let body: unknown = null
      try {
        body = await response.json()
      } catch {
        body = null
      }
      const detail = extractDetail(body, response.status)
      throw new ApiError(detail, response.status, detail)
    }

    // 204 and friends have no body.
    if (response.status === 204) return undefined as T
    return (await response.json()) as T
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  }

  // ------------------------------------------------------------------ routes
  /** `GET /health` - component, database, schedule and connector readiness. */
  getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health')
  }

  /** `GET /model/describe` - architecture, grid and hazard configuration. */
  getModel(): Promise<ModelDescribeResponse> {
    return this.request<ModelDescribeResponse>('/model/describe')
  }

  /** `GET /model/checkpoint` - active checkpoint, provenance and calibration. */
  getCheckpoint(): Promise<CheckpointResponse> {
    return this.request<CheckpointResponse>('/model/checkpoint')
  }

  /** `POST /forecast` - run inference and return per-step fields and risk. */
  createForecast(params: ForecastParams = {}): Promise<ForecastResponse> {
    return this.post<ForecastResponse>('/forecast', {
      event_id: params.event_id ?? null,
      lead_hours: params.lead_hours ?? null,
      include_uncertainty: params.include_uncertainty ?? false,
      mc_samples: params.mc_samples ?? null,
    })
  }

  /** `POST /risk/point` - terrain-aware risk for a single coordinate. */
  getPointRisk(params: PointRiskParams): Promise<PointRiskResponse> {
    return this.post<PointRiskResponse>('/risk/point', {
      lat: params.lat,
      lon: params.lon,
      event_id: params.event_id ?? null,
      lead_hours: params.lead_hours ?? null,
    })
  }

  /** `POST /risk/geojson` - contiguous risk polygons for the map layer. */
  getRiskGeoJson(params: GeoJsonParams = {}): Promise<GeoJsonResponse> {
    return this.post<GeoJsonResponse>('/risk/geojson', {
      event_id: params.event_id ?? null,
      lead_hours: params.lead_hours ?? null,
      min_category: params.min_category ?? 1,
      risk_field: params.risk_field ?? 'overall',
      max_features: params.max_features ?? 2000,
    })
  }

  /** `POST /explain` - Grad-CAM, physical-consistency audit and what-if. */
  explain(params: ExplainParams = {}): Promise<ExplainResponse> {
    return this.post<ExplainResponse>('/explain', {
      hazard: params.hazard ?? 'cloudburst',
      step: params.step ?? null,
      event_id: params.event_id ?? null,
      include_consistency: params.include_consistency ?? true,
      include_what_if: params.include_what_if ?? false,
      perturbations: params.perturbations ?? null,
    })
  }

  /** `GET /forecast/history` - paginated, newest-first run history. */
  getHistory(params: HistoryParams = {}): Promise<ForecastHistoryResponse> {
    const query = buildQuery({
      limit: params.limit,
      offset: params.offset,
      status: params.status,
      is_synthetic: params.is_synthetic,
      model_version: params.model_version,
    })
    return this.request<ForecastHistoryResponse>(`/forecast/history${query}`)
  }

  /** `GET /forecast/{id}` - one persisted run with provenance and risk cells. */
  getForecastDetail(
    runId: number,
    params: ForecastDetailParams = {},
  ): Promise<ForecastDetailResponse> {
    const query = buildQuery({
      include_risk: params.include_risk,
      risk_limit: params.risk_limit,
      min_risk: params.min_risk,
    })
    return this.request<ForecastDetailResponse>(`/forecast/${runId}${query}`)
  }

  /** `POST /forecast/persist` - run inference and store an auditable record. */
  persistForecast(params: PersistParams = {}): Promise<PersistForecastResponse> {
    return this.post<PersistForecastResponse>('/forecast/persist', {
      event_id: params.event_id ?? null,
      lead_hours: params.lead_hours ?? null,
      persist_risk: params.persist_risk ?? true,
      max_cells: params.max_cells ?? 20000,
    })
  }
}

/** Shared instance used by the app; tests construct their own. */
export const api = new SihpsApi()
