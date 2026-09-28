/**
 * Overview page: backend connectivity, component health, model/checkpoint status
 * and the available lead times.
 *
 * Everything shown here comes from `GET /health`, `GET /model/describe` and
 * `GET /model/checkpoint`; nothing is assumed client-side.
 */

import { useEffect, type ReactNode } from 'react'

import type { ConnectionState } from '../state/AppContext'
import { useApp } from '../state/AppContext'
import { ErrorState, LoadingState, StatTile } from '../components/ui/States'
import { DataSourcePanel, SyntheticFlag } from '../components/Provenance'
import { formatLeadHours, formatTimestamp, shortenId } from '../lib/format'
import { EXPERIMENTAL_DISCLAIMER } from '../api/types'

const CONNECTION_COPY: Record<ConnectionState, { label: string; hint: string; className: string }> = {
  connecting: {
    label: 'Connecting to backend',
    hint: 'Waiting for the first health response.',
    className: 'text-slate-400',
  },
  online: {
    label: 'Backend online',
    hint: 'All reported components responded successfully.',
    className: 'text-risk-low',
  },
  degraded: {
    label: 'Backend degraded',
    hint: 'The API responded, but at least one component is unavailable or the database is unreachable.',
    className: 'text-risk-moderate',
  },
  offline: {
    label: 'Backend unreachable',
    hint: 'No response from the API. Start it with "uvicorn app.main:app --port 8000".',
    className: 'text-risk-extreme',
  },
}

export function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="shrink-0 text-slate-500">{label}</span>
      <span className="text-right font-mono text-slate-300">{value}</span>
    </div>
  )
}

function ComponentRow({
  name,
  status,
  detail,
}: {
  name: string
  status: string
  detail?: string
}) {
  const ok = status === 'ok'
  const unknown = status === 'unknown'
  return (
    <li className="flex items-center justify-between gap-3 border-b border-base-600/40 px-4 py-2.5 last:border-b-0">
      <div className="min-w-0">
        <p className="text-xs font-semibold text-slate-200">{name}</p>
        {detail && <p className="truncate text-[10px] text-slate-500">{detail}</p>}
      </div>
      <span
        className={`pill shrink-0 ${
          ok
            ? 'border-risk-low/40 bg-risk-low/10 text-risk-low'
            : unknown
              ? 'border-slate-600 bg-slate-700/40 text-slate-400'
              : 'border-risk-high/40 bg-risk-high/10 text-risk-high'
        }`}
      >
        {status}
      </span>
    </li>
  )
}

export function OverviewPage() {
  const {
    connection,
    health,
    healthError,
    model,
    checkpoint,
    refreshHealth,
    leadHours,
    setLeadHours,
    runForecast,
    forecast,
    forecastLoading,
  } = useApp()

  // Kick off an initial forecast so the rest of the dashboard has data to show.
  useEffect(() => {
    if (!forecast && !forecastLoading) void runForecast()
    // Run once on mount; depending on `forecast` would create a request loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const copy = CONNECTION_COPY[connection]

  if (healthError && !health) {
    return (
      <ErrorState error={healthError} onRetry={() => void refreshHealth()} context="Backend health" />
    )
  }

  if (!health) return <LoadingState label="Querying backend health…" rows={5} />

  const grid = health.settings.grid
  const modelComponent = health.components.model
  const db = health.database
  const schedule = health.schedule
  const connectors = health.connectors
  const dataSources = health.data_sources
  const terrain = health.components.terrain

  return (
    <div className="space-y-4 p-4 lg:p-6">
      {/* connectivity */}
      <section className="panel p-4" aria-label="Backend connectivity">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-slate-100">
              <span className={copy.className}>{copy.label}</span>
            </h2>
            <p className="mt-0.5 text-xs text-slate-400">{copy.hint}</p>
          </div>
          <div className="flex items-center gap-2">
            <SyntheticFlag isSynthetic={health.is_synthetic} />
            <button type="button" className="btn" onClick={() => void refreshHealth()}>
              Refresh
            </button>
          </div>
        </div>
        {health.status === 'degraded' && health.error && (
          <p className="mt-3 rounded border border-risk-moderate/30 bg-risk-moderate/10 px-3 py-2 text-[11px] text-slate-300">
            Reported reason: {health.error}
          </p>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="panel lg:col-span-2" aria-label="Component health">
          <div className="panel-header">
            <h3 className="panel-title">Component health</h3>
            <span className="font-mono text-[10px] text-slate-500">env {health.env}</span>
          </div>
          <ul>
            <ComponentRow
              name="Grid"
              status={health.components.grid.status}
              detail={`${grid.nx}×${grid.ny} @ ${grid.res_km} km · ${grid.crs} · ${grid.frame_minutes} min frames`}
            />
            <ComponentRow
              name="Terrain stack"
              status={terrain.status}
              detail={
                typeof terrain.elevation_m_max === 'number'
                  ? `elevation ${Math.round(Number(terrain.elevation_m_min))}–${Math.round(
                      Number(terrain.elevation_m_max),
                    )} m`
                  : undefined
              }
            />
            <ComponentRow
              name="Nowcast model"
              status={modelComponent.status}
              detail={`${modelComponent.model_version} · backend ${
                modelComponent.resolved_backend ?? 'n/a'
              } · ${
                modelComponent.trained_checkpoint_loaded
                  ? 'trained checkpoint'
                  : 'UNTRAINED (random) weights'
              }`}
            />
            <ComponentRow
              name="Risk engine"
              status={health.components.risk_engine.status}
              detail={`weights ${(health.components.risk_engine.weights ?? health.settings.risk_weights).join(
                ' / ',
              )} · thresholds ${health.settings.risk_thresholds.join(' / ')}`}
            />
            <ComponentRow name="Explainability" status={health.components.explainability.status} />
            <ComponentRow
              name="Database"
              status={db?.status ?? 'unknown'}
              detail={
                db?.current_revision ? `${db.dialect} · revision ${db.current_revision}` : db?.error
              }
            />
            <ComponentRow
              name="Live data connectors"
              status={connectors?.any_live_available ? 'ok' : 'unavailable'}
              detail={
                connectors?.any_live_available
                  ? 'at least one real source is connected'
                  : 'no real MOSDAC/IMDAA/IMD feed is connected; the synthetic generator is the only data source'
              }
            />
          </ul>
          <div className="px-4 pb-4">
            <DataSourcePanel
              dataSources={dataSources}
              disclaimer={health.disclaimer ?? EXPERIMENTAL_DISCLAIMER}
            />
          </div>
        </section>

        <div className="space-y-4">
          <section className="panel" aria-label="Model configuration">
            <div className="panel-header">
              <h3 className="panel-title">Model</h3>
            </div>
            <div className="grid grid-cols-2 gap-2 p-4">
              <StatTile
                label="Version"
                value={<span className="text-xs">{health.settings.model_version}</span>}
              />
              <StatTile
                label="Backend"
                value={modelComponent.resolved_backend ?? '—'}
                hint={`requested ${health.settings.model_backend}`}
              />
              <StatTile
                label="Trained weights"
                value={modelComponent.trained_checkpoint_loaded ? 'yes' : 'no'}
                tone={modelComponent.trained_checkpoint_loaded ? 'accent' : 'warn'}
                hint={modelComponent.trained_checkpoint_loaded ? undefined : 'random weights'}
              />
              <StatTile
                label="Forecast steps"
                value={model?.describe.config.backbone.predict_steps ?? health.settings.forecast_steps}
                hint={`${health.settings.sequence_length} input frames`}
              />
            </div>
          </section>

          <section className="panel" aria-label="Checkpoint status">
            <div className="panel-header">
              <h3 className="panel-title">Checkpoint</h3>
            </div>
            {checkpoint ? (
              <div className="space-y-2 p-4 text-[11px]">
                <Row label="Present" value={checkpoint.checkpoint_present ? 'yes' : 'no'} />
                <Row label="Loaded" value={checkpoint.trained_checkpoint_loaded ? 'yes' : 'no'} />
                <Row
                  label="Calibration"
                  value={checkpoint.calibration_present ? 'available' : 'none'}
                />
                <Row
                  label="Observational validation"
                  value={
                    <span className="text-risk-moderate">
                      {checkpoint.observational_validation ? 'yes' : 'none'}
                    </span>
                  }
                />
                {checkpoint.checkpoint_sha256 && (
                  <Row label="SHA-256" value={shortenId(checkpoint.checkpoint_sha256, 10, 8)} />
                )}
                <p className="rounded border border-base-600/60 bg-base-900/60 p-2 leading-relaxed text-slate-400">
                  {checkpoint.validation_status}
                </p>
              </div>
            ) : (
              <LoadingState label="Reading checkpoint status…" rows={2} />
            )}
          </section>
        </div>
      </div>

      {/* lead times */}
      <section className="panel" aria-label="Lead times">
        <div className="panel-header">
          <h3 className="panel-title">Forecast lead times</h3>
          <span className="font-mono text-[10px] text-slate-500">
            {forecast ? `init ${formatTimestamp(forecast.init_time)}` : 'no run yet'}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 p-4">
          {forecast?.lead_times_h.length ? (
            forecast.lead_times_h.map((hours) => (
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
            ))
          ) : (
            <p className="text-xs text-slate-500">
              Lead times appear once the first forecast has been generated.
            </p>
          )}
          {forecastLoading && (
            <span className="flex items-center gap-2 text-[11px] text-slate-400">
              <span className="h-2 w-2 animate-pulse rounded-full bg-accent" aria-hidden="true" />
              running inference…
            </span>
          )}
        </div>
        {forecast?.selected && (
          <p className="px-4 pb-3 font-mono text-[10px] text-slate-500">
            selected: step {forecast.selected.step} · valid{' '}
            {formatTimestamp(forecast.selected.valid_time)}
          </p>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel" aria-label="Background scheduling">
          <div className="panel-header">
            <h3 className="panel-title">Background scheduling</h3>
          </div>
          <div className="space-y-2 p-4 text-[11px]">
            <Row
              label="Periodic tasks"
              value={
                <span className={schedule?.schedule_enabled ? 'text-risk-low' : 'text-slate-400'}>
                  {schedule?.schedule_enabled ? 'enabled' : 'disabled'}
                </span>
              }
            />
            <Row label="Ingestion interval" value={`${schedule?.ingest_interval_minutes ?? '—'} min`} />
            <Row
              label="Batch inference interval"
              value={`${schedule?.batch_inference_interval_minutes ?? '—'} min`}
            />
            {schedule?.note && (
              <p className="rounded border border-base-600/60 bg-base-900/60 p-2 leading-relaxed text-slate-400">
                {schedule.note}
              </p>
            )}
          </div>
        </section>

        <section className="panel" aria-label="Data provenance">
          <div className="panel-header">
            <h3 className="panel-title">Data provenance</h3>
          </div>
          <div className="space-y-2 p-4 text-[11px]">
            <Row label="Source" value={health.data_source} />
            <Row label="Attribution" value={health.attribution} />
            <p className="rounded border border-base-600/60 bg-base-900/60 p-2 leading-relaxed text-slate-400">
              {health.disclaimer}
            </p>
            <p className="leading-relaxed text-slate-400">
              <span className="font-semibold text-slate-300">Accuracy: </span>
              {health.accuracy_claim}
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
