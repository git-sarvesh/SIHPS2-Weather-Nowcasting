/**
 * Uncertainty page: MC-dropout spread and 90% intervals from
 * `POST /forecast` with `include_uncertainty=true`.
 *
 * Honesty constraint: with untrained weights the ensemble spread measures
 * *epistemic* uncertainty of an untrained network, not calibrated forecast error.
 * This page therefore states the model and calibration status next to every number
 * and never labels the interval as a confidence statement.
 */

import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { Hazard, UncertaintyResponse } from '../api/types'
import { ProvenanceBanner } from '../components/Provenance'
import { EmptyState, ErrorState, LoadingState, StatTile } from '../components/ui/States'
import { Row } from './OverviewPage'
import { formatProbability } from '../lib/format'
import { useApp } from '../state/AppContext'

const HAZARDS: Hazard[] = ['thunderstorm', 'cloudburst', 'flood']

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
  },
} as const

export function UncertaintyPage() {
  const { runForecast, forecast, forecastLoading, checkpoint, health } = useApp()
  const [samples, setSamples] = useState(20)
  const [error, setError] = useState<unknown>(null)

  const uncertainty: UncertaintyResponse | null = forecast?.uncertainty ?? null

  const run = async (count: number) => {
    setError(null)
    try {
      await runForecast({ include_uncertainty: true, mc_samples: count })
    } catch (caught) {
      setError(caught)
    }
  }

  // Run once on mount if the current forecast has no ensemble attached.
  useEffect(() => {
    if (!forecast?.uncertainty && !forecastLoading) void run(samples)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Interval chart data, built from the backend's reported bounds.
  const intervalData = useMemo(() => {
    if (!uncertainty) return []
    return HAZARDS.map((hazard) => {
      const spread = uncertainty.spread_at_selected_step?.[hazard]
      const interval = uncertainty.interval_90_at_selected_step?.[hazard]
      return {
        hazard,
        meanStd: spread?.mean_std ?? 0,
        maxStd: spread?.max_std ?? 0,
        lower: interval?.lower ?? 0,
        upper: interval?.upper ?? 0,
        width: interval ? interval.upper - interval.lower : 0,
      }
    })
  }, [uncertainty])

  const trained = health?.components.model.trained_checkpoint_loaded ?? false
  const calibrated = checkpoint?.calibration_present ?? false

  return (
    <div className="space-y-4 p-4 lg:p-6">
      {forecast && <ProvenanceBanner provenance={forecast} />}

      <section className="panel" aria-label="Ensemble controls">
        <div className="flex flex-wrap items-end gap-4 p-4">
          <div>
            <p className="label mb-1.5">MC-dropout samples</p>
            <select
              className="field w-40"
              value={samples}
              onChange={(event) => setSamples(Number(event.target.value))}
              aria-label="MC-dropout sample count"
            >
              {[8, 12, 20, 32, 50].map((count) => (
                <option key={count} value={count}>
                  {count} samples
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void run(samples)}
            disabled={forecastLoading}
          >
            {forecastLoading ? 'Sampling…' : 'Run ensemble'}
          </button>
          {uncertainty && (
            <p className="font-mono text-[10px] text-slate-500">
              {uncertainty.method} · n={uncertainty.n_samples}
            </p>
          )}
        </div>
      </section>

      {/* Interpretation guard - the most important element on this page. */}
      <section
        className={`rounded-lg border p-3 text-[11px] leading-relaxed ${
          trained && calibrated
            ? 'border-risk-low/40 bg-risk-low/10 text-slate-300'
            : 'border-risk-moderate/45 bg-risk-moderate/10 text-slate-300'
        }`}
        role="note"
        aria-label="Uncertainty interpretation"
      >
        <p className="font-semibold">
          {trained && calibrated
            ? 'Trained model with calibration artefacts.'
            : 'How to read these numbers'}
        </p>
        <p className="mt-1">
          {!trained && (
            <>
              The active model has <strong>untrained (randomly initialised) weights</strong>.
              MC-dropout spread here reflects the variability of an untrained network,{' '}
              <strong>not</strong> calibrated forecast error. Treat these intervals as a
              demonstration of the ensemble machinery.
            </>
          )}
          {trained && !calibrated && (
            <>
              A trained checkpoint is loaded, but <strong>no calibration artefact is
              available</strong>. The 90% intervals are quantiles of the ensemble, not verified
              coverage.
            </>
          )}
          {trained && calibrated && (
            <>
              Intervals come from the model&rsquo;s calibration artefact. Skill is still unverified
              against independent observations.
            </>
          )}
        </p>
      </section>

      {error ? (
        <ErrorState error={error} onRetry={() => void run(samples)} context="Ensemble" />
      ) : !uncertainty ? (
        <section className="panel">
          {forecastLoading ? (
            <LoadingState label="Running MC-dropout ensemble…" rows={4} />
          ) : (
            <EmptyState
              title="No ensemble available"
              message="Run the ensemble to compute MC-dropout spread and 90% intervals for the selected lead time."
              action={
                <button
                  type="button"
                  className="btn btn-primary mt-2"
                  onClick={() => void run(samples)}
                >
                  Run ensemble
                </button>
              }
            />
          )}
        </section>
      ) : (
        <>
          <section className="grid grid-cols-1 gap-3 md:grid-cols-3" aria-label="Spread summary">
            {HAZARDS.map((hazard) => {
              const spread = uncertainty.spread_at_selected_step?.[hazard]
              const interval = uncertainty.interval_90_at_selected_step?.[hazard]
              return (
                <div key={hazard} className="panel p-3">
                  <p className="label" style={{ color: HAZARD_COLORS[hazard] }}>
                    {hazard}
                  </p>
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <StatTile
                      label="mean σ"
                      value={formatProbability(spread?.mean_std, 4)}
                      hint="spatial mean std"
                    />
                    <StatTile
                      label="max σ"
                      value={formatProbability(spread?.max_std, 4)}
                      hint="worst cell"
                    />
                  </div>
                  {interval && (
                    <p className="mt-2 font-mono text-[11px] text-slate-400">
                      90% interval [{interval.lower.toFixed(3)}, {interval.upper.toFixed(3)}]
                    </p>
                  )}
                </div>
              )
            })}
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="panel" aria-label="Ensemble spread">
              <div className="panel-header">
                <h3 className="panel-title">Spread by hazard</h3>
              </div>
              <div className="h-64 p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={intervalData}
                    margin={{ top: 8, right: 12, bottom: 4, left: -18 }}
                  >
                    <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
                    <XAxis dataKey="hazard" tick={AXIS_STYLE} stroke="#2b3648" />
                    <YAxis
                      tick={AXIS_STYLE}
                      stroke="#2b3648"
                      tickFormatter={(value: number) => value.toFixed(3)}
                    />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Bar dataKey="meanStd" name="mean σ" radius={[3, 3, 0, 0]}>
                      {intervalData.map((entry) => (
                        <Cell key={entry.hazard} fill={HAZARD_COLORS[entry.hazard]} />
                      ))}
                    </Bar>
                    <Bar dataKey="maxStd" name="max σ" fill="#475569" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="panel" aria-label="90% interval width">
              <div className="panel-header">
                <h3 className="panel-title">90% interval width</h3>
              </div>
              <div className="h-64 p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={intervalData}
                    layout="vertical"
                    margin={{ top: 4, right: 16, bottom: 4, left: 8 }}
                  >
                    <CartesianGrid stroke="#1b2434" strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} tick={AXIS_STYLE} stroke="#2b3648" />
                    <YAxis
                      type="category"
                      dataKey="hazard"
                      width={92}
                      tick={AXIS_STYLE}
                      stroke="#2b3648"
                    />
                    <Tooltip
                      {...TOOLTIP_STYLE}
                      formatter={(value) => [Number(value).toFixed(3), 'width']}
                    />
                    <ReferenceLine x={0.2} stroke="#fb923c" strokeDasharray="4 4" />
                    <Bar dataKey="width" radius={[0, 3, 3, 0]}>
                      {intervalData.map((entry) => (
                        <Cell key={entry.hazard} fill={HAZARD_COLORS[entry.hazard]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="px-4 pb-3 text-[10px] text-slate-500">
                The dashed line marks a 0.2 interval width as a visual reference only &mdash; it is
                not a pass/fail threshold.
              </p>
            </section>
          </div>

          <section className="panel" aria-label="Ensemble metadata">
            <div className="panel-header">
              <h3 className="panel-title">Ensemble metadata</h3>
            </div>
            <div className="grid gap-2 p-4 sm:grid-cols-2 lg:grid-cols-3">
              <Row label="Method" value={uncertainty.method} />
              <Row label="Samples" value={uncertainty.n_samples} />
              <Row label="Model version" value={forecast?.model_version ?? '—'} />
              <Row label="Trained weights" value={trained ? 'yes' : 'no'} />
              <Row label="Calibration artefact" value={calibrated ? 'present' : 'none'} />
              <Row
                label="Observational validation"
                value={checkpoint?.observational_validation ? 'yes' : 'none'}
              />
              {forecast && <Row label="Init time" value={forecast.init_time} />}
              {forecast && <Row label="Valid time" value={forecast.selected.valid_time} />}
              {forecast && <Row label="Lead" value={`${forecast.selected.lead_hours} h`} />}
            </div>
          </section>

        </>
      )}
    </div>
  )
}
