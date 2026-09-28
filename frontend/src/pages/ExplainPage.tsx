/**
 * Explainability page: Grad-CAM heatmap, channel attribution ranking, the
 * physical-consistency audit and a what-if perturbation panel - all from
 * `POST /explain`.
 *
 * The Grad-CAM raster is a plain `[height][width]` numeric array from the
 * backend; it is normalised client-side for display only, and the raw min/max are
 * always shown so the scaling is never mistaken for a calibrated quantity.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { ExplainResponse, Hazard } from '../api/types'
import { ProvenanceBanner } from '../components/Provenance'
import { ErrorState, LoadingState } from '../components/ui/States'
import { Row } from './OverviewPage'
import { formatProbability } from '../lib/format'
import { useApp } from '../state/AppContext'

const HAZARDS: Hazard[] = ['thunderstorm', 'cloudburst', 'flood']

/** Channels the what-if panel exposes, with their physical meaning. */
const PERTURBABLE_CHANNELS = [
  { key: 'iwv', label: 'Integrated water vapour', hint: 'moisture' },
  { key: 'ctt', label: 'Cloud-top temperature', hint: 'convection' },
  { key: 'cape', label: 'CAPE', hint: 'instability' },
  { key: 'tir1_bt', label: 'TIR1 brightness temp', hint: 'cloud top' },
  { key: 'wv_bt', label: 'Water-vapour BT', hint: 'upper moisture' },
]

/** Cold-to-hot ramp for the Grad-CAM overlay. */
const HEAT_COLORS = ['#0b1220', '#1e3a8a', '#0891b2', '#facc15', '#f97316', '#f43f5e']

function heatColor(t: number): string {
  if (Number.isNaN(t)) return HEAT_COLORS[0]
  const clamped = Math.max(0, Math.min(1, t))
  const index = Math.min(HEAT_COLORS.length - 1, Math.floor(clamped * HEAT_COLORS.length))
  return HEAT_COLORS[index]
}

/**
 * Render a numeric raster as a CSS grid heatmap.
 *
 * Using a canvas here would be faster but is unreadable in tests and loses the
 * accessible fallback, so a grid of cells is used and downsampled to a sane size.
 */
function Heatmap({ raster }: { raster: number[][] }) {
  const maxSide = 96
  const height = raster.length
  const width = raster[0]?.length ?? 0
  if (height === 0 || width === 0) {
    return <p className="p-4 text-xs text-slate-500">The backend returned an empty Grad-CAM map.</p>
  }

  const step = Math.max(1, Math.ceil(Math.max(height, width) / maxSide))
  const rows: number[][] = []
  for (let y = 0; y < height; y += step) {
    const row: number[] = []
    for (let x = 0; x < width; x += step) row.push(raster[y][x])
    rows.push(row)
  }

  const flat = raster.flat()
  const min = Math.min(...flat)
  const max = Math.max(...flat)
  const span = max - min || 1

  return (
    <div className="p-3">
      <div
        role="img"
        aria-label={`Grad-CAM heatmap, ${width} by ${height} cells, values ${min.toFixed(3)} to ${max.toFixed(3)}`}
        className="mx-auto overflow-hidden rounded border border-base-600"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${rows[0]?.length ?? 1}, minmax(0, 1fr))`,
          maxWidth: '32rem',
          aspectRatio: `${width} / ${height}`,
        }}
      >
        {rows.flatMap((row, y) =>
          row.map((value, x) => (
            <span
              key={`${y}-${x}`}
              className="block h-full w-full"
              style={{ backgroundColor: heatColor((value - min) / span) }}
            />
          )),
        )}
      </div>
      <p className="mt-2 text-center font-mono text-[10px] text-slate-500">
        {width}×{height} cells · min {min.toFixed(4)} · max {max.toFixed(4)} · mean{' '}
        {(flat.reduce((a, b) => a + b, 0) / flat.length).toFixed(4)}
      </p>
    </div>
  )
}

export function ExplainPage() {
  const { client, forecast, leadHours } = useApp()

  const [hazard, setHazard] = useState<Hazard>('cloudburst')
  const [response, setResponse] = useState<ExplainResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [perturbations, setPerturbations] = useState<Record<string, number>>({})
  const [includeWhatIf, setIncludeWhatIf] = useState(false)

  // Map the selected lead time onto a step index the backend accepts.
  const selectedStep = useMemo(() => {
    if (!forecast || leadHours === null) return null
    const index = forecast.lead_times_h.findIndex((hours) => hours === leadHours)
    return index >= 0 ? index : null
  }, [forecast, leadHours])

  const runExplain = useCallback(
    async (options: { whatIf?: boolean } = {}) => {
      const withWhatIf = options.whatIf ?? includeWhatIf
      setLoading(true)
      setError(null)
      try {
        const result = await client.explain({
          hazard,
          step: selectedStep,
          include_consistency: true,
          include_what_if: withWhatIf,
          perturbations: withWhatIf ? perturbations : null,
        })
        setResponse(result)
      } catch (caught) {
        setError(caught)
        setResponse(null)
      } finally {
        setLoading(false)
      }
    },
    [client, hazard, selectedStep, includeWhatIf, perturbations],
  )

  // Re-run whenever the hazard or the selected lead time changes.
  useEffect(() => {
    void runExplain()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hazard, selectedStep])

  const consistency = response?.physical_consistency ?? null
  const whatIf = response?.what_if ?? null
  const channelData = useMemo(
    () =>
      Object.entries(response?.attribution_result.channel_attributions ?? {})
        .map(([channel, value]) => ({ channel, value }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)),
    [response],
  )

  return (
    <div className="space-y-4 p-4 lg:p-6">
      {response && <ProvenanceBanner provenance={response} />}

      {/* controls */}
      <section className="panel" aria-label="Explainability controls">
        <div className="flex flex-wrap items-end gap-4 p-4">
          <div>
            <p className="label mb-1.5">Hazard</p>
            <div className="flex gap-1.5" role="group" aria-label="Hazard">
              {HAZARDS.map((item) => (
                <button
                  key={item}
                  type="button"
                  aria-pressed={hazard === item}
                  onClick={() => setHazard(item)}
                  className={`btn ${hazard === item ? 'btn-primary' : ''}`}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="min-w-48 flex-1">
            <p className="label mb-1.5">
              What-if perturbation {includeWhatIf ? '(active)' : '(off)'}
            </p>
            <div className="flex flex-wrap gap-2">
              {PERTURBABLE_CHANNELS.map((channel) => (
                <label
                  key={channel.key}
                  className="flex items-center gap-1.5 rounded border border-base-600 bg-base-900/60 px-2 py-1 text-[10px] text-slate-300"
                  title={channel.hint}
                >
                  <span className="font-mono">{channel.key}</span>
                  <input
                    type="number"
                    step="0.05"
                    min="-1"
                    max="1"
                    value={perturbations[channel.key] ?? ''}
                    onChange={(event) => {
                      const raw = event.target.value
                      setPerturbations((current) => {
                        const next = { ...current }
                        if (raw === '') delete next[channel.key]
                        else next[channel.key] = Number(raw)
                        return next
                      })
                    }}
                    placeholder="0.0"
                    aria-label={`${channel.label} perturbation`}
                    className="w-14 bg-transparent text-right font-mono outline-none"
                  />
                </label>
              ))}
            </div>
          </div>

          <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-slate-300">
            <input
              type="checkbox"
              checked={includeWhatIf}
              onChange={(event) => setIncludeWhatIf(event.target.checked)}
              className="h-3.5 w-3.5 rounded border-base-500 bg-base-900 accent-accent"
            />
            include what-if
          </label>

          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void runExplain({ whatIf: includeWhatIf })}
          >
            {loading ? 'Computing…' : 'Run explanation'}
          </button>
        </div>
      </section>

      {error ? (
        <ErrorState error={error} onRetry={() => void runExplain()} context="Explainability" />
      ) : loading && !response ? (
        <section className="panel">
          <LoadingState label="Computing Grad-CAM attribution…" rows={4} />
        </section>
      ) : response ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <section className="panel" aria-label="Grad-CAM">
            <div className="panel-header">
              <h3 className="panel-title">Grad-CAM · {response.hazard}</h3>
              <span className="font-mono text-[10px] text-slate-500">
                step {response.step} · {response.lead_hours} h lead
              </span>
            </div>
            <Heatmap raster={response.attribution_result.gradcam_map} />
            <p className="px-4 pb-3 text-[10px] leading-relaxed text-slate-500">
              Pixel-wise contribution to the {response.hazard} output. Colours are min–max
              normalised for display only and carry no calibrated meaning.
            </p>
          </section>

          <section className="panel" aria-label="Channel attribution">
            <div className="panel-header">
              <h3 className="panel-title">Channel attribution</h3>
              <span className="font-mono text-[10px] text-slate-500">
                {channelData.length} channels
              </span>
            </div>
            <div className="h-72 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={channelData}
                  layout="vertical"
                  margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
                >
                  <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                    stroke="#2b3648"
                  />
                  <YAxis
                    type="category"
                    dataKey="channel"
                    width={96}
                    tick={{ fontSize: 10, fill: '#94a3b8' }}
                    stroke="#2b3648"
                  />
                  <Tooltip
                    contentStyle={{
                      background: '#0a0f1a',
                      border: '1px solid #2b3648',
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(value) => [Number(value).toFixed(4), 'attribution']}
                  />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                    {channelData.map((entry) => (
                      <Cell
                        key={entry.channel}
                        fill={entry.value >= 0 ? '#38bdf8' : '#fb923c'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel" aria-label="Physical consistency">
            <div className="panel-header">
              <h3 className="panel-title">Physical consistency</h3>
              <span
                className={`pill ${
                  consistency?.consistent
                    ? 'border-risk-low/40 bg-risk-low/10 text-risk-low'
                    : 'border-risk-moderate/40 bg-risk-moderate/10 text-risk-moderate'
                }`}
              >
                score {consistency ? consistency.score.toFixed(2) : '—'}
              </span>
            </div>
            {consistency ? (
              <div className="space-y-2 p-4 text-[11px]">
                {Object.entries(consistency.checks).map(([name, check]) => (
                  <div key={name} className="flex items-center justify-between gap-3">
                    <span className="text-slate-300">{name.replace(/_/g, ' ')}</span>
                    <span className="flex items-center gap-2">
                      {typeof check.conforming_fraction === 'number' && (
                        <span className="font-mono text-slate-500">
                          {(check.conforming_fraction * 100).toFixed(0)}% conforming
                        </span>
                      )}
                      <span className={check.passed ? 'text-risk-low' : 'text-risk-moderate'}>
                        {check.passed ? 'pass' : 'fail'}
                      </span>
                    </span>
                  </div>
                ))}
                {consistency.violations.length > 0 && (
                  <ul className="mt-2 space-y-1 rounded border border-risk-moderate/30 bg-risk-moderate/10 p-2 text-slate-300">
                    {consistency.violations.map((violation) => (
                      <li key={violation}>• {violation}</li>
                    ))}
                  </ul>
                )}
              </div>
            ) : (
              <p className="p-4 text-xs text-slate-500">No consistency report returned.</p>
            )}
          </section>

          <section className="panel" aria-label="What-if simulation">
            <div className="panel-header">
              <h3 className="panel-title">What-if simulation</h3>
            </div>
            {whatIf ? (
              <div className="space-y-2 p-4 text-[11px]">
                <p className="text-slate-500">
                  Perturbations:{' '}
                  {Object.entries(whatIf.perturbations_applied)
                    .map(([key, value]) => `${key} ${value > 0 ? '+' : ''}${value}`)
                    .join(', ')}
                </p>
                {Object.keys(whatIf.delta).map((key) => {
                  const delta = whatIf.delta[key] ?? 0
                  return (
                    <Row
                      key={key}
                      label={key.replace(/_/g, ' ')}
                      value={
                        <span className={delta >= 0 ? 'text-risk-high' : 'text-risk-low'}>
                          {formatProbability(whatIf.baseline[key])} →{' '}
                          {formatProbability(whatIf.counterfactual[key])} (
                          {delta > 0 ? '+' : ''}
                          {delta.toFixed(4)})
                        </span>
                      }
                    />
                  )
                })}
                <p className="rounded border border-base-600/60 bg-base-900/60 p-2 leading-relaxed text-slate-400">
                  A counterfactual sensitivity probe on an untrained network. It shows how the
                  output responds to an input perturbation; it is not a physical prediction.
                </p>
              </div>
            ) : (
              <p className="p-4 text-xs text-slate-500">
                Enable &ldquo;include what-if&rdquo; and set at least one perturbation, then run the
                explanation.
              </p>
            )}
          </section>
        </div>
      ) : (
        <section className="panel p-4 text-xs text-slate-500">No explanation loaded yet.</section>
      )}
    </div>
  )
}
