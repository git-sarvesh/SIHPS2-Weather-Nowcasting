/**
 * Tests for the API client.
 *
 * The client is exercised against a stubbed `fetch` so URL construction, payload
 * shaping, query building and error handling are verified without a running
 * backend. The shapes asserted here mirror the backend's actual responses.
 */

import { describe, expect, it, vi } from 'vitest'

import { ApiError, SihpsApi, buildQuery, extractDetail } from '../api/client'

const BASE = 'http://api.test/api/v1'

/** Build a client whose fetch returns `body` with `status`. */
function stubApi(body: unknown, status = 200, capture?: { url?: string; init?: RequestInit }) {
  const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    if (capture) {
      capture.url = String(url)
      capture.init = init
    }
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as Response
  })
  return { api: new SihpsApi({ baseUrl: BASE, fetchImpl: fetchImpl as never }), fetchImpl }
}

describe('buildQuery', () => {
  it('returns an empty string when nothing is set', () => {
    expect(buildQuery({ a: null, b: undefined, c: '' })).toBe('')
  })

  it('keeps zero and false, which are meaningful values', () => {
    expect(buildQuery({ limit: 0, is_synthetic: false })).toBe('?limit=0&is_synthetic=false')
  })

  it('omits nulls so the backend keeps its own defaults', () => {
    expect(buildQuery({ limit: 10, offset: 0, status: null })).toBe('?limit=10&offset=0')
  })
})

describe('extractDetail', () => {
  it('returns a string detail as-is', () => {
    expect(extractDetail({ detail: 'boom' }, 500)).toBe('boom')
  })

  it('flattens FastAPI validation errors', () => {
    const body = { detail: [{ loc: ['body', 'lead_hours'], msg: 'must be > 0' }] }
    expect(extractDetail(body, 422)).toBe('lead_hours: must be > 0')
  })

  it('falls back to the status when the body is unusable', () => {
    expect(extractDetail(null, 503)).toBe('Request failed with status 503')
  })
})

describe('SihpsApi routing', () => {
  it('calls GET /health', async () => {
    const capture: { url?: string } = {}
    const { api } = stubApi({ status: 'ok' }, 200, capture)
    await api.getHealth()
    expect(capture.url).toBe(`${BASE}/health`)
  })

  it('strips a trailing slash from the base URL', async () => {
    const calls: string[] = []
    const fetchImpl = vi.fn(async (url: string | URL | Request) => {
      calls.push(String(url))
      return { ok: true, status: 200, json: async () => ({}) } as Response
    })
    const api = new SihpsApi({ baseUrl: `${BASE}///`, fetchImpl: fetchImpl as never })
    await api.getHealth()
    expect(calls[0]).toBe(`${BASE}/health`)
  })

  it('POSTs a forecast body with explicit nulls for unset fields', async () => {
    const capture: { url?: string; init?: RequestInit } = {}
    const { api } = stubApi({ fields: {} }, 200, capture)
    await api.createForecast({ lead_hours: 3, include_uncertainty: true })

    expect(capture.url).toBe(`${BASE}/forecast`)
    expect(capture.init?.method).toBe('POST')
    expect(capture.init?.headers).toMatchObject({ 'Content-Type': 'application/json' })
    expect(JSON.parse(capture.init?.body as string)).toEqual({
      event_id: null,
      lead_hours: 3,
      include_uncertainty: true,
      mc_samples: null,
    })
  })

  it('POSTs point-risk coordinates', async () => {
    const capture: { init?: RequestInit } = {}
    const { api } = stubApi({}, 200, capture)
    await api.getPointRisk({ lat: 30.5, lon: 78.5, lead_hours: 2 })

    expect(JSON.parse(capture.init?.body as string)).toEqual({
      lat: 30.5,
      lon: 78.5,
      event_id: null,
      lead_hours: 2,
    })
  })

  it('sends the risk-field and category filters for GeoJSON', async () => {
    const capture: { init?: RequestInit } = {}
    const { api } = stubApi({ features: [] }, 200, capture)
    await api.getRiskGeoJson({ risk_field: 'flood_risk', min_category: 2 })

    expect(JSON.parse(capture.init?.body as string)).toMatchObject({
      risk_field: 'flood_risk',
      min_category: 2,
    })
  })

  it('builds a history query string', async () => {
    const capture: { url?: string } = {}
    const { api } = stubApi({ runs: [] }, 200, capture)
    await api.getHistory({ limit: 20, offset: 40, is_synthetic: true })

    expect(capture.url).toBe(`${BASE}/forecast/history?limit=20&offset=40&is_synthetic=true`)
  })

  it('interpolates the run id into the detail route', async () => {
    const capture: { url?: string } = {}
    const { api } = stubApi({ run: {} }, 200, capture)
    await api.getForecastDetail(7, { include_risk: false, risk_limit: 5 })

    expect(capture.url).toBe(`${BASE}/forecast/7?include_risk=false&risk_limit=5`)
  })

  it('sends explain perturbations only when supplied', async () => {
    const capture: { init?: RequestInit } = {}
    const { api } = stubApi({}, 200, capture)

    await api.explain({ hazard: 'cloudburst' })
    expect(JSON.parse(capture.init?.body as string).perturbations).toBeNull()

    await api.explain({ hazard: 'cloudburst', include_what_if: true, perturbations: { iwv: 0.1 } })
    expect(JSON.parse(capture.init?.body as string).perturbations).toEqual({ iwv: 0.1 })
  })
})

describe('SihpsApi error handling', () => {
  it('raises ApiError with the backend message and status', async () => {
    const { api } = stubApi({ detail: 'step must be in [0, 5]; got 9' }, 400)

    await expect(api.explain({ hazard: 'cloudburst', step: 9 })).rejects.toThrowError(ApiError)
    await api.explain({ hazard: 'cloudburst', step: 9 }).catch((error: ApiError) => {
      expect(error.status).toBe(400)
      expect(error.detail).toBe('step must be in [0, 5]; got 9')
      expect(error.isNetworkError).toBe(false)
    })
  })

  it('flags an unreachable backend as a network error', async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    })
    const api = new SihpsApi({ baseUrl: BASE, fetchImpl: fetchImpl as never })

    await api.getHealth().catch((error: ApiError) => {
      expect(error).toBeInstanceOf(ApiError)
      expect(error.isNetworkError).toBe(true)
      expect(error.status).toBe(0)
      expect(error.message).toContain('Could not reach the SIHPS backend')
    })
  })

  it('never resolves to placeholder data on failure', async () => {
    const { api } = stubApi({ detail: 'database unavailable' }, 503)
    // If the client swallowed the error it would resolve to `body`; assert it throws.
    await expect(api.getHistory()).rejects.toBeTruthy()
  })
})
