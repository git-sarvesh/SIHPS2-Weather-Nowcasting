/**
 * Provenance banner.
 *
 * This component exists so the synthetic-data disclaimer is structurally hard to
 * remove: it renders whenever the backend reports `is_synthetic` or `demo_mode`,
 * and it always shows the backend's own `disclaimer` and `accuracy_claim` text
 * rather than a shortened paraphrase. The visual style is deliberately legible
 * rather than decorative - a polished UI must not bury an experimental-data
 * caveat.
 */

import type { DataSourcesHealth, RealSourceStatus, Provenance } from '../api/types'

export type BannerTone = 'synthetic' | 'degraded' | 'operational'

/**
 * Choose a tone. Anything not clearly operational is treated as synthetic, so a
 * new or missing flag fails safe (shows the warning) rather than open.
 */
export function bannerTone(prov: Pick<Provenance, 'is_synthetic' | 'demo_mode'> | null | undefined): BannerTone {
  if (!prov) return 'synthetic'
  if (prov.is_synthetic === false && prov.demo_mode === false) return 'operational'
  if (prov.demo_mode) return 'synthetic'
  return 'synthetic'
}

const TONE_STYLES: Record<BannerTone, { box: string; tag: string; label: string }> = {
  synthetic: {
    box: 'border-risk-moderate/45 bg-risk-moderate/10',
    tag: 'border-risk-moderate/50 bg-risk-moderate/15 text-risk-moderate',
    label: 'SYNTHETIC DEMONSTRATION',
  },
  degraded: {
    box: 'border-risk-high/45 bg-risk-high/10',
    tag: 'border-risk-high/50 bg-risk-high/15 text-risk-high',
    label: 'DEGRADED',
  },
  operational: {
    box: 'border-risk-low/40 bg-risk-low/10',
    tag: 'border-risk-low/50 bg-risk-low/15 text-risk-low',
    label: 'OPERATIONAL',
  },
}

export function ProvenanceBanner({
  provenance,
  compact = false,
  className = '',
}: {
  provenance: Pick<Provenance, 'is_synthetic' | 'demo_mode' | 'data_source' | 'attribution' | 'disclaimer' | 'accuracy_claim'>
  compact?: boolean
  className?: string
}) {
  const tone = bannerTone(provenance)
  const styles = TONE_STYLES[tone]

  return (
    <div
      className={`rounded-lg border px-3 py-2.5 ${styles.box} ${className}`}
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className={`pill ${styles.tag}`}>
          <span aria-hidden="true">◆</span>
          {styles.label}
        </span>
        <span className="text-[11px] text-slate-300">{provenance.data_source}</span>
      </div>

      {!compact && (
        <div className="mt-2 space-y-1 text-[11px] leading-relaxed text-slate-300">
          <p>{provenance.disclaimer}</p>
          <p className="text-slate-400">
            <span className="font-semibold text-slate-300">Accuracy: </span>
            {provenance.accuracy_claim}
          </p>
          {provenance.attribution && (
            <p className="text-slate-500">{provenance.attribution}</p>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Compact inline pill for dense headers and table rows.
 *
 * Renders "SYNTHETIC" whenever the payload is synthetic so the flag is never
 * more than a glance away from a number.
 */
export function SyntheticFlag({ isSynthetic }: { isSynthetic: boolean }) {
  return isSynthetic ? (
    <span
      className="pill border-risk-moderate/50 bg-risk-moderate/15 text-risk-moderate"
      title="Synthetic demonstration data - not an observation and not an official IMD warning."
    >
      <span aria-hidden="true">◆</span>
      SYNTHETIC
    </span>
  ) : (
    <span
      className="pill border-risk-low/50 bg-risk-low/15 text-risk-low"
      title="Produced from a non-synthetic data source."
    >
      <span aria-hidden="true">●</span>
      OPERATIONAL
    </span>
  )
}

/**
 * Real-data transparency panel.
 *
 * Purpose: make the difference between *synthetic demo*, *model prediction* and
 * *genuine observation/reanalysis* impossible to miss, and to state plainly
 * which real sources this deployment can currently read.
 *
 * Every claim here is driven by the backend's own `/health` payload. The
 * component never asserts that a source is connected unless the backend says
 * `available: true`, and it always shows the manual step required otherwise.
 */

/** Short label for an availability state, phrased as a status not a promise. */
const AVAILABILITY_LABEL: Record<RealSourceStatus['availability'], string> = {
  available: 'CONNECTED',
  needs_credentials: 'NEEDS ACCOUNT',
  needs_manual_download: 'NEEDS DOWNLOAD',
  unreachable: 'UNREACHABLE',
  metadata_only: 'METADATA ONLY',
  not_implemented: 'NOT IMPLEMENTED',
  not_probed: 'NOT PROBED',
}

const AVAILABILITY_TONE: Record<RealSourceStatus['availability'], string> = {
  available: 'border-risk-low/45 bg-risk-low/10 text-risk-low',
  needs_credentials: 'border-amber-400/45 bg-amber-500/15 text-amber-200',
  needs_manual_download: 'border-amber-400/45 bg-amber-500/15 text-amber-200',
  unreachable: 'border-rose-400/45 bg-rose-500/15 text-rose-200',
  metadata_only: 'border-amber-400/45 bg-amber-500/15 text-amber-200',
  not_implemented: 'border-slate-500/45 bg-slate-500/15 text-slate-300',
  not_probed: 'border-slate-500/45 bg-slate-500/15 text-slate-300',
}

export function SourceStatusRow({ source }: { source: RealSourceStatus }) {
  return (
    <li className="rounded-lg border border-base-600/60 bg-base-900/60 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-semibold text-slate-200">{source.source}</span>
        <span
          className={`pill ${AVAILABILITY_TONE[source.availability]}`}
          title={source.reason}
        >
          {AVAILABILITY_LABEL[source.availability]}
        </span>
      </div>
      <p className="mt-1 text-[10px] leading-relaxed text-slate-400">{source.reason}</p>
    </li>
  )
}

export function DataSourcePanel({
  dataSources,
  disclaimer,
}: {
  dataSources: DataSourcesHealth | null | undefined
  disclaimer: string
}) {
  if (!dataSources) return null
  const split = dataSources.split_feasibility

  return (
    <section
      className="space-y-3 rounded-xl border border-base-600/60 bg-base-900/40 p-3"
      aria-label="Real data sources"
    >
      <header className="space-y-1">
        <h2 className="text-xs font-bold uppercase tracking-[0.12em] text-slate-300">
          Data sources
        </h2>
        <p className="text-[11px] leading-relaxed text-slate-400">
          {dataSources.any_real_available
            ? 'At least one real-world source is connected. Check each label below before treating any field as an observation.'
            : 'This deployment is running on synthetic demonstration data. No live observation or reanalysis feed is connected.'}
        </p>
      </header>

      <ul className="space-y-1.5">
        {dataSources.sources.map((source) => (
          <SourceStatusRow key={source.source} source={source} />
        ))}
      </ul>

      {split && (
        <div className="space-y-1.5 rounded-lg border border-base-600/60 bg-base-900/60 p-3">
          <h3 className="text-[10px] font-bold uppercase tracking-[0.12em] text-slate-400">
            Historical evaluation coverage
          </h3>
          <p className="text-[11px] leading-relaxed text-slate-300">
            {split.feasible
              ? 'The requested train/validation/test year split can be built from the available data.'
              : `The required split cannot be built from currently accessible data. Unsatisfiable: ${split.unsatisfiable.join(', ') || 'none'}.`}
          </p>
          <ul className="list-disc space-y-1 pl-4 text-[10px] leading-relaxed text-slate-400">
            {split.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-[10px] leading-relaxed text-slate-500">{disclaimer}</p>
    </section>
  )
}

/** Compact provenance line for panel footers (timestamp + source). */
export function ProvenanceFooter({
  modelVersion,
  initTime,
  dataSource,
}: {
  modelVersion?: string | null
  initTime?: string | null
  dataSource?: string | null
}) {
  return (
    <p className="px-4 py-2 font-mono text-[10px] leading-relaxed text-slate-500">
      {modelVersion && <span>{modelVersion}</span>}
      {modelVersion && (initTime || dataSource) && <span> · </span>}
      {initTime && <span>init {initTime}</span>}
      {initTime && dataSource && <span> · </span>}
      {dataSource && <span>{dataSource}</span>}
    </p>
  )
}
