/**
 * Integration check: drive the dashboard's API surface against the real FastAPI
 * app, not a mock.
 *
 * Start the backend first, then run from `frontend/`:
 *
 *     cd backend && python -m uvicorn app.main:app --port 8000
 *     cd frontend && node scripts/verify-against-backend.mjs
 *
 * Override the target with SIHPS_API_BASE_URL if the API is not on port 8000.
 */

const BASE = process.env.SIHPS_API_BASE_URL ?? 'http://localhost:8000/api/v1'

let passed = 0
let failed = 0

function check(name, condition, detail = '') {
  if (condition) {
    passed += 1
    console.log(`  PASS  ${name}`)
  } else {
    failed += 1
    console.log(`  FAIL  ${name} ${detail}`)
  }
}

async function post(path, body) {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })
  return { status: response.status, body: await response.json() }
}

async function get(path) {
  const response = await fetch(`${BASE}${path}`, { headers: { Accept: 'application/json' } })
  return { status: response.status, body: await response.json() }
}

console.log(`Verifying dashboard API contract against ${BASE}\n`)

// --- health ---------------------------------------------------------------
const health = await get('/health')
check('GET /health -> 200', health.status === 200, `got ${health.status}`)
check('health reports provenance', typeof health.body?.is_synthetic === 'boolean')
check('health exposes components', typeof health.body?.components?.model === 'object')
check('health exposes database', typeof health.body?.database?.status === 'string')
check('health exposes schedule', typeof health.body?.schedule?.schedule_enabled === 'boolean')
check('health reports no live connector', health.body?.connectors?.any_live_available === false)
check('grid has a real size', Number(health.body?.settings?.grid?.nx) > 0)

// --- model ----------------------------------------------------------------
const described = await get('/model/describe')
check('GET /model/describe -> 200', described.status === 200)
check('model describe reports a version', typeof described.body?.model_version === 'string')
check(
  'model describe has a backbone config',
  typeof described.body?.describe?.config?.backbone?.predict_steps === 'number',
)
check('model describe reports its grid', typeof described.body?.grid?.nx === 'number')

const checkpoint = await get('/model/checkpoint')
check('GET /model/checkpoint -> 200', checkpoint.status === 200)
check(
  'checkpoint reports no observational validation',
  checkpoint.body?.observational_validation === false,
)
check('checkpoint has a validation_status', typeof checkpoint.body?.validation_status === 'string')

// --- forecast -------------------------------------------------------------
const forecast = await post('/forecast', { include_uncertainty: true, mc_samples: 6 })
check('POST /forecast -> 200', forecast.status === 200, `got ${forecast.status}`)
check(
  'forecast has all three hazard fields',
  ['thunderstorm', 'cloudburst', 'flood'].every((k) => k in (forecast.body?.fields ?? {})),
)
check(
  'forecast has a per_step series',
  Array.isArray(forecast.body?.per_step) && forecast.body.per_step.length > 0,
)
check(
  'forecast has a risk summary',
  typeof forecast.body?.risk?.summary?.max_overall_risk === 'number',
)
check('forecast is flagged synthetic', forecast.body?.is_synthetic === true)
check('forecast carries a disclaimer', typeof forecast.body?.disclaimer === 'string')
check('forecast carries an accuracy_claim', typeof forecast.body?.accuracy_claim === 'string')
check(
  'forecast uncertainty is attached',
  typeof forecast.body?.uncertainty === 'object' && forecast.body?.uncertainty !== null,
)
check(
  'uncertainty reports its method',
  typeof forecast.body?.uncertainty?.method === 'string',
)

// --- point risk -----------------------------------------------------------
const point = await post('/risk/point', { lat: 30.3, lon: 78.9, lead_hours: 1 })
check('POST /risk/point -> 200', point.status === 200, `got ${point.status}`)
check('point risk has hazards', typeof point.body?.hazards?.thunderstorm === 'number')
check('point risk has a category', typeof point.body?.risk_category === 'string')

// --- geojson --------------------------------------------------------------
const geo = await post('/risk/geojson', { min_category: 0, max_features: 25 })
check('POST /risk/geojson -> 200', geo.status === 200, `got ${geo.status}`)
check('geojson is a FeatureCollection', geo.body?.type === 'FeatureCollection')
check('geojson has features', Array.isArray(geo.body?.features) && geo.body.features.length > 0)
const first = geo.body?.features?.[0]
check('feature geometry is a polygon', first?.geometry?.type === 'Polygon')
check('feature has a risk_category', typeof first?.properties?.risk_category === 'string')
check('feature has risk_max', typeof first?.properties?.risk_max === 'number')
check('geojson metadata carries the grid', typeof geo.body?.metadata?.grid?.nx === 'number')
check('geojson metadata is flagged synthetic', geo.body?.metadata?.is_synthetic === true)

// --- explain --------------------------------------------------------------
const explain = await post('/explain', { hazard: 'cloudburst' })
check('POST /explain (default step) -> 200', explain.status === 200, `got ${explain.status}`)
check(
  'explain has a gradcam raster',
  Array.isArray(explain.body?.attribution_result?.gradcam_map),
)
check(
  'explain has channel attributions',
  typeof explain.body?.attribution_result?.channel_attributions === 'object',
)
check(
  'explain has a consistency audit',
  typeof explain.body?.physical_consistency?.score === 'number',
)
check('explain step is within range', explain.body?.step >= 0)

const whatIf = await post('/explain', {
  hazard: 'cloudburst',
  include_what_if: true,
  perturbations: { iwv: 0.1 },
})
check('POST /explain what_if -> 200', whatIf.status === 200)
check('what_if returns a delta object', typeof whatIf.body?.what_if?.delta === 'object')

const badStep = await post('/explain', { hazard: 'cloudburst', step: 999 })
check('POST /explain rejects an out-of-range step', badStep.status === 400, `got ${badStep.status}`)

// --- history --------------------------------------------------------------
const history = await get('/forecast/history?limit=5')
check('GET /forecast/history -> 200', history.status === 200, `got ${history.status}`)
check(
  'history has pagination fields',
  typeof history.body?.total === 'number' && typeof history.body?.has_more === 'boolean',
)
check('history runs is an array', Array.isArray(history.body?.runs))

// --- persistence round trip ----------------------------------------------
const persisted = await post('/forecast/persist', { max_cells: 16 })
check('POST /forecast/persist -> 200', persisted.status === 200, `got ${persisted.status}`)
check('persist stores risk cells', persisted.body?.risk_cells === 16)
check('persist flags synthetic', persisted.body?.is_synthetic === true)
check('persist returns a notice', typeof persisted.body?.notice === 'string')

const runId = persisted.body?.forecast_run_id
const detail = await get(`/forecast/${runId}?risk_limit=5`)
check('GET /forecast/{id} -> 200', detail.status === 200, `got ${detail.status}`)
check('detail has stored cells', detail.body?.risk_cell_count === 16)
check('detail returns the requested cells', detail.body?.risk_cells?.length === 5)

const status = await get(`/forecast/${runId}/status`)
check('GET /forecast/{id}/status -> 200', status.status === 200)
check('status is terminal', status.body?.is_terminal === true)

console.log(`\n${passed} passed, ${failed} failed`)
process.exit(failed === 0 ? 0 : 1)
