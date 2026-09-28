/**
 * Forecast page: per-hazard probabilities, lead-time series, area mix, the
 * persisted run history from `GET /forecast/history`, and a run-detail drawer
 * from `GET /forecast/{id}`.
 */

import { useCallback, useEffect, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { ForecastDetailResponse, ForecastHistoryResponse, Hazard } from '../api/types'
import { ProvenanceBanner, SyntheticFlag } from '../components/Provenance'
import { EmptyState, ErrorState, LoadingState } from '../components/ui/States'
import { Row } from './OverviewPage'
import {
  CATEGORY_COLORS,
  RISK_CATEGORIES,
  formatLeadHours,
  formatProbability,
  formatRelative,
  formatTimestamp,
} from '../lib/format'
import { useApp } from '../state/AppContext'

const HAZARD_COLORS: Record<Hazard, string> = {
  thunderstorm: '#facc15',
  cloudburst: '#38bdf8',
  flood: '#a78bfa',
}

const AXIS_STYLE = { fontSize: 10, fill: '#94a3b8' } as const
const TOOLTIP_STYLE = {
  contentStyle: {
    background: '#0a0f1a',
    border: '1px solid #2b3648',
    borderRadius: 8,
    fontSize: 11,
    color: '#e2e8f0',
  },
  labelStyle: { color: '#94a3b8', fontSize: 10 },
} as const

export function ForecastPage() {
  const { forecast, forecastLoading, forecastError, runForecast, leadHours, setLeadHours } =
    useApp()

  const [history, setHistory] = useState<ForecastHistoryResponse | null>(null)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<unknown>(null)
  const [selectedRun, setSelectedRun] = useState<number | null>(null)
  const [detail, setDetail] = useState<ForecastDetailResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<unknown>(null)
  const [syntheticOnly, setSyntheticOnly] = useState(false)

  const { client } = useApp()

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true)
    setHistoryError(null)
    try {
      const response = await client.getHistory({
        limit: 20,
        offset: 0,
        is_synthetic: syntheticOnly ? true : null,
      })
      setHistory(response)
    } catch (error) {
      setHistoryError(error)
      setHistory(null)
    } finally {
      setHistoryLoading(false)
    }
  }, [client, syntheticOnly])

  useEffect(() => {
    void fetchHistory()
  }, [fetchHistory])

  const fetchDetail = useCallback(
    async (runId: number) => {
      setSelectedRun(runId)
      setDetailLoading(true)
      setDetailError(null)
      try {
        const response = await client.getForecastDetail(runId, {
          include_risk: true,
          risk_limit: 25,
        })
        setDetail(response)
      } catch (error) {
        setDetailError(error)
        setDetail(null)
      } finally {
        setDetailLoading(false)
      }
    },
    [client],
  )

  if (forecastError && !forecast) {
    return (
      <ErrorState error={forecastError} onRetry={() => void runForecast()} context="Forecast run" />
    )
  }

  if (!forecast) return <LoadingState label="Generating forecast…" rows={5} />

  const { fields, per_step, risk, selected } = forecast
  const summary = risk.summary

  // Chart series derived from the backend's per-step statistics.
  const leadSeries = per_step.map((step) => ({
    lead: formatLeadHours(step.lead_hours),
    leadHours: step.lead_hours,
    thunderstormMax: step.thunderstorm.max,
    cloudburstMax: step.cloudburst.max,
    floodMax: step.flood.max,
    thunderstormP95: step.thunderstorm.p95,
    cloudburstP95: step.cloudburst.p95,
    floodP95: step.flood.p95,
  }))

  const areaMix = RISK_CATEGORIES.map((category) => ({
    category,
    fraction: (summary.area_fraction?.[category] ?? 0) * 100,
  })).filter((entry) => entry.fraction > 0)

  return (
    <div className="space-y-4 p-4 lg:p-6">
      <ProvenanceBanner provenance={forecast} />

      {/* headline metrics */}
      <section className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="Hazard probabilities">
        {(['thunderstorm', 'cloudburst', 'flood'] as Hazard[]).map((hazard) => (
          <div key={hazard} className="panel p-3">
            <p className="label">{hazard === 'flood' ? 'Flood probability' : hazard}</p>
            <p
              className="mt-1 font-mono text-2xl leading-none"
              style={{ color: HAZARD_COLORS[hazard] }}
            >
              {formatProbability(fields[hazard]?.max)}
            </p>
            <p className="mt-1 font-mono text-[10px] text-slate-500">
              mean {formatProbability(fields[hazard]?.mean)} · p95{' '}
              {formatProbability(fields[hazard]?.p95)}
            </p>
          </div>
        ))}
        <div className="panel p-3">
          <p className="label">Max overall risk</p>
          <p className="mt-1 font-mono text-2xl leading-none text-slate-100">
            {formatProbability(summary.max_overall_risk)}
          </p>
          <p className="mt-1 font-mono text-[10px] text-slate-500">
            mean {formatProbability(summary.mean_overall_risk)}
          </p>
        </div>
      </section>

      {/* lead-time selector */}
      <section className="panel" aria-label="Lead-time selection">
        <div className="panel-header">
          <h3 className="panel-title">Lead time</h3>
          <span className="font-mono text-[10px] text-slate-500">
            valid {formatTimestamp(selected.valid_time)} · init{' '}
            {formatTimestamp(forecast.init_time)}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 p-4">
          {forecast.lead_times_h.map((hours) => (
            <button
              key={hours}
              type="button"
              aria-pressed={leadHours === hours}
              onClick={() => {
                setLeadHours(hours)
                void runForecast({ lead_hours: hours })
              }}
              className={`btn ${leadHours === hours ? 'btn-primary' : ''}`}
            >
              {formatLeadHours(hours)}
            </button>
          ))}
          <button
            type="button"
            className="btn"
            onClick={() => void runForecast({ lead_hours: null, include_uncertainty: true })}
          >
            Run with MC uncertainty
          </button>
          {forecastLoading && (
            <span className="flex items-center gap-2 text-[11px] text-slate-400">
              <span className="h-2 w-2 animate-pulse rounded-full bg-accent" aria-hidden="true" />
              running…
            </span>
          )}
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel" aria-label="Hazard probability by lead time">
          <div className="panel-header">
            <h3 className="panel-title">Peak probability by lead time</h3>
          </div>
          <div className="h-72 p-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={leadSeries} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
                <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
                <XAxis dataKey="lead" tick={AXIS_STYLE} stroke="#2b3648" />
                <YAxis
                  tick={AXIS_STYLE}
                  stroke="#2b3648"
                  domain={[0, 1]}
                  tickFormatter={(value: number) => value.toFixed(1)}
                />
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line
                  type="monotone"
                  dataKey="thunderstormMax"
                  name="thunderstorm"
                  stroke={HAZARD_COLORS.thunderstorm}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="cloudburstMax"
                  name="cloudburst"
                  stroke={HAZARD_COLORS.cloudburst}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="floodMax"
                  name="flood"
                  stroke={HAZARD_COLORS.flood}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel" aria-label="Risk area mix">
          <div className="panel-header">
            <h3 className="panel-title">Area fraction by category</h3>
          </div>
          {areaMix.length ? (
            <div className="h-72 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={areaMix} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
                  <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
                  <XAxis dataKey="category" tick={AXIS_STYLE} stroke="#2b3648" />
                  <YAxis
                    tick={AXIS_STYLE}
                    stroke="#2b3648"
                    tickFormatter={(value: number) => `${value.toFixed(0)}%`}
                  />
                  <Tooltip
                    {...TOOLTIP_STYLE}
                    formatter={(value) => [`${Number(value).toFixed(1)}%`, 'area fraction']}
                  />
                  <Bar dataKey="fraction" radius={[4, 4, 0, 0]}>
                    {areaMix.map((entry) => (
                      <Cell key={entry.category} fill={CATEGORY_COLORS[entry.category]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              title="No area data"
              message="The backend returned an empty area fraction."
            />
          )}
        </section>
      </div>

      <section className="panel" aria-label="p95 probability by lead time">
        <div className="panel-header">
          <h3 className="panel-title">p95 probability by lead time</h3>
        </div>
        <div className="h-64 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={leadSeries} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
              <defs>
                {(['thunderstorm', 'cloudburst', 'flood'] as Hazard[]).map((hazard) => (
                  <linearGradient key={hazard} id={`grad-${hazard}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={HAZARD_COLORS[hazard]} stopOpacity={0.5} />
                    <stop offset="95%" stopColor={HAZARD_COLORS[hazard]} stopOpacity={0.02} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
              <XAxis dataKey="lead" tick={AXIS_STYLE} stroke="#2b3648" />
              <YAxis
                tick={AXIS_STYLE}
                stroke="#2b3648"
                domain={[0, 1]}
                tickFormatter={(value: number) => value.toFixed(1)}
              />
              <Tooltip {...TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {(['thunderstorm', 'cloudburst', 'flood'] as Hazard[]).map((hazard) => (
                <Area
                  key={hazard}
                  type="monotone"
                  dataKey={`${hazard}P95`}
                  name={hazard}
                  stroke={HAZARD_COLORS[hazard]}
                  fill={`url(#grad-${hazard})`}
                  strokeWidth={1.5}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* persisted run history */}
      <section className="panel" aria-label="Forecast run history">
        <div className="panel-header">
          <h3 className="panel-title">Persisted runs</h3>
          <div className="flex items-center gap-2">
            <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-slate-400">
              <input
                type="checkbox"
                checked={syntheticOnly}
                onChange={(event) => setSyntheticOnly(event.target.checked)}
                className="h-3.5 w-3.5 rounded border-base-500 bg-base-900 accent-accent"
              />
              synthetic only
            </label>
            <button type="button" className="btn" onClick={() => void fetchHistory()}>
              Refresh
            </button>
          </div>
        </div>

        {historyError ? (
          <ErrorState
            error={historyError}
            onRetry={() => void fetchHistory()}
            context="Run history"
          />
        ) : historyLoading && !history ? (
          <LoadingState label="Loading run history…" rows={3} />
        ) : history && history.runs.length === 0 ? (
          <EmptyState
            title="No persisted runs"
            message="Runs appear here after POST /forecast/persist stores a forecast. Nothing has been persisted yet."
            action={
              <button type="button" className="btn btn-primary mt-2" onClick={() => void runForecast()}>
                Run a forecast
              </button>
            }
          />
        ) : history ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px]">
              <thead className="text-slate-500">
                <tr className="border-b border-base-600/60">
                  <th scope="col" className="px-4 py-2 font-medium">ID</th>
                  <th scope="col" className="px-4 py-2 font-medium">Status</th>
                  <th scope="col" className="px-4 py-2 font-medium">Initialised</th>
                  <th scope="col" className="px-4 py-2 font-medium">Lead</th>
                  <th scope="col" className="px-4 py-2 font-medium">Cells</th>
                  <th scope="col" className="px-4 py-2 font-medium">Provenance</th>
                  <th scope="col" className="px-4 py-2 font-medium">Model</th>
                </tr>
              </thead>
              <tbody>
                {history.runs.map((run) => (
                  <tr
                    key={run.id}
                    className={`cursor-pointer border-b border-base-600/30 transition hover:bg-base-700/40 ${
                      selectedRun === run.id ? 'bg-base-700/60' : ''
                    }`}
                    onClick={() => void fetchDetail(run.id)}
                  >
                    <td className="px-4 py-2 font-mono text-slate-300">{run.id}</td>
                    <td
                      className={`px-4 py-2 ${
                        run.status === 'succeeded'
                          ? 'text-risk-low'
                          : run.status === 'failed'
                            ? 'text-risk-extreme'
                            : 'text-slate-400'
                      }`}
                    >
                      {run.status}
                    </td>
                    <td
                      className="px-4 py-2 font-mono text-slate-400"
                      title={formatTimestamp(run.init_time)}
                    >
                      {formatRelative(run.init_time)}
                    </td>
                    <td className="px-4 py-2 font-mono text-slate-400">
                      {formatLeadHours(run.lead_hours)}
                    </td>
                    <td className="px-4 py-2 font-mono text-slate-400">{run.risk_cells}</td>
                    <td className="px-4 py-2">
                      <SyntheticFlag isSynthetic={run.is_synthetic} />
                    </td>
                    <td className="px-4 py-2 font-mono text-[10px] text-slate-500">
                      {run.model_version}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="px-4 py-2 text-[10px] text-slate-500">
              {history.count} of {history.total} runs ·{' '}
              {history.has_more ? 'more available' : 'end of list'}
            </p>
          </div>
        ) : null}
      </section>

      {/* selected run detail */}
      {selectedRun !== null && (
        <section className="panel" aria-label="Run detail">
          <div className="panel-header">
            <h3 className="panel-title">Run #{selectedRun}</h3>
            <button type="button" className="btn" onClick={() => setSelectedRun(null)}>
              Close
            </button>
          </div>
          {detailLoading ? (
            <LoadingState label="Loading run detail…" rows={3} />
          ) : detailError ? (
            <ErrorState error={detailError} onRetry={() => void fetchDetail(selectedRun)} />
          ) : detail ? (
            <div className="space-y-3 p-4">
              <p className="rounded border border-risk-moderate/30 bg-risk-moderate/10 px-3 py-2 text-[11px] leading-relaxed text-slate-300">
                {detail.notice}
              </p>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                <Row label="Status" value={detail.run.status} />
                <Row label="Kind" value={detail.run.kind} />
                <Row label="Model" value={detail.run.model_version} />
                <Row
                  label="Trained checkpoint"
                  value={detail.run.trained_checkpoint_loaded ? 'yes' : 'no'}
                />
                <Row label="Initialised" value={formatTimestamp(detail.run.init_time)} />
                <Row label="Valid" value={formatTimestamp(detail.run.valid_time)} />
                <Row label="Duration" value={`${detail.duration_seconds ?? '—'} s`} />
                <Row label="Risk cells stored" value={detail.risk_cell_count} />
                <Row label="Error" value={detail.error ?? 'none'} />
              </div>
              {detail.risk_cells.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-[11px]">
                    <caption className="sr-only">Top risk cells for run {detail.run.id}</caption>
                    <thead className="text-slate-500">
                      <tr className="border-b border-base-600/60">
                        <th scope="col" className="px-3 py-1.5 font-medium">Lat</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">Lon</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">Overall</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">TS</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">CB</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">Flood</th>
                        <th scope="col" className="px-3 py-1.5 font-medium">Category</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.risk_cells.map((cell) => (
                        <tr key={cell.id} className="border-b border-base-600/20">
                          <td className="px-3 py-1.5 font-mono">{cell.lat.toFixed(3)}</td>
                          <td className="px-3 py-1.5 font-mono">{cell.lon.toFixed(3)}</td>
                          <td className="px-3 py-1.5 font-mono">
                            {formatProbability(cell.overall_risk)}
                          </td>
                          <td className="px-3 py-1.5 font-mono">
                            {formatProbability(cell.thunderstorm)}
                          </td>
                          <td className="px-3 py-1.5 font-mono">
                            {formatProbability(cell.cloudburst)}
                          </td>
                          <td className="px-3 py-1.5 font-mono">
                            {formatProbability(cell.flood_risk)}
                          </td>
                          <td
                            className="px-3 py-1.5 font-semibold"
                            style={{ color: CATEGORY_COLORS[cell.risk_category] }}
                          >
                            {cell.risk_category}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : null}
        </section>
      )}
    </div>
  )
}
