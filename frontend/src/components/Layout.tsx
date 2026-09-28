/**
 * Dashboard shell: header, navigation, connection indicator and disclaimer
 * footer. Tab navigation is used deliberately - the pages share a forecast run
 * and a persistent banner, so a router would add a dependency without benefit.
 */

import { useEffect, useState, type ReactNode } from 'react'

import type { ConnectionState } from '../state/AppContext'
import { useApp } from '../state/AppContext'
import { APP_SUBTITLE, APP_TITLE } from '../config'

export type PageId = 'overview' | 'map' | 'forecast' | 'explain' | 'uncertainty'

export interface PageDefinition {
  id: PageId
  label: string
  hint: string
  icon: string
}

export const PAGES: PageDefinition[] = [
  { id: 'overview', label: 'Overview', hint: 'System health and model status', icon: '◉' },
  { id: 'map', label: 'Risk Map', hint: 'Interactive hazard layers', icon: '⬡' },
  { id: 'forecast', label: 'Forecast', hint: 'Probabilities and run history', icon: '◫' },
  { id: 'explain', label: 'Explainability', hint: 'Grad-CAM and physical audit', icon: '◎' },
  { id: 'uncertainty', label: 'Uncertainty', hint: 'MC-dropout spread and intervals', icon: '◇' },
]

const CONNECTION_STYLES: Record<ConnectionState, { dot: string; text: string; label: string }> = {
  connecting: { dot: 'bg-slate-500 animate-pulse', text: 'text-slate-400', label: 'Connecting' },
  online: { dot: 'bg-risk-low', text: 'text-risk-low', label: 'Backend online' },
  degraded: { dot: 'bg-risk-moderate', text: 'text-risk-moderate', label: 'Degraded' },
  offline: { dot: 'bg-risk-extreme', text: 'text-risk-extreme', label: 'Backend offline' },
}

function ConnectionBadge() {
  const { connection, health, lastUpdated } = useApp()
  const styles = CONNECTION_STYLES[connection]
  const title = health
    ? `${health.app_name} · env ${health.env}`
    : 'The SIHPS backend has not responded yet.'

  return (
    <div
      className="flex items-center gap-2 rounded-lg border border-base-600/70 bg-base-800/70 px-3 py-1.5"
      title={title}
      role="status"
      aria-live="polite"
    >
      <span className={`h-2 w-2 rounded-full ${styles.dot}`} aria-hidden="true" />
      <span className={`text-[11px] font-semibold ${styles.text}`}>{styles.label}</span>
      {lastUpdated && connection === 'online' && (
        <span className="hidden font-mono text-[10px] text-slate-500 sm:inline">
          {lastUpdated.toLocaleTimeString()}
        </span>
      )}
    </div>
  )
}

export function Layout({
  page,
  onNavigate,
  children,
}: {
  page: PageId
  onNavigate: (page: PageId) => void
  children: ReactNode
}) {
  // Roving arrow-key navigation across the tab list.
  const [focusIndex, setFocusIndex] = useState(0)

  useEffect(() => {
    setFocusIndex(Math.max(0, PAGES.findIndex((p) => p.id === page)))
  }, [page])

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return
    event.preventDefault()
    const delta = event.key === 'ArrowRight' ? 1 : -1
    const next = (focusIndex + delta + PAGES.length) % PAGES.length
    setFocusIndex(next)
    onNavigate(PAGES[next].id)
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <header className="sticky top-0 z-30 border-b border-base-600/60 bg-base-900/85 backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 lg:px-6">
          <div className="flex items-center gap-3">
            <div
              aria-hidden="true"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-accent/40 bg-accent/10 text-accent shadow-glow"
            >
              ◈
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-wide text-slate-100">{APP_TITLE}</h1>
              <p className="text-[11px] text-slate-400">{APP_SUBTITLE}</p>
            </div>
          </div>
          <ConnectionBadge />
        </div>

        <div
          role="tablist"
          aria-label="Dashboard sections"
          onKeyDown={onKeyDown}
          className="flex gap-1 overflow-x-auto px-2 pb-1 lg:px-4"
        >
          {PAGES.map((definition, index) => {
            const selected = definition.id === page
            return (
              <button
                key={definition.id}
                type="button"
                role="tab"
                id={`tab-${definition.id}`}
                aria-selected={selected}
                aria-controls={`panel-${definition.id}`}
                tabIndex={index === focusIndex ? 0 : -1}
                title={definition.hint}
                onClick={() => onNavigate(definition.id)}
                className={`flex shrink-0 items-center gap-2 rounded-t-lg border-b-2 px-3 py-2 text-xs font-semibold transition ${
                  selected
                    ? 'border-accent text-accent'
                    : 'border-transparent text-slate-400 hover:border-base-500 hover:text-slate-200'
                }`}
              >
                <span aria-hidden="true">{definition.icon}</span>
                <span className="whitespace-nowrap">{definition.label}</span>
              </button>
            )
          })}
        </div>
      </header>

      <main
        role="tabpanel"
        id={`panel-${page}`}
        aria-labelledby={`tab-${page}`}
        // The panel is a flex column so a full-height child (the radar map) can
        // size against it. `flex-1` alone does not give a resolvable height
        // inside a fixed-height column, which collapses such children to 0.
        // Scrolling is opt-in per page: the radar map fills the panel, the
        // content pages scroll.
        className={`flex min-h-0 flex-1 flex-col ${
          page === 'map' ? 'overflow-hidden' : 'overflow-y-auto'
        }`}
      >
        {children}
      </main>

      <footer className="border-t border-base-600/60 px-4 py-3 lg:px-6">
        <p className="text-[11px] leading-relaxed text-slate-500">
          <span className="font-semibold text-slate-400">Experimental system.</span> Outputs are
          research demonstrations and are not official India Meteorological Department (IMD)
          warnings. No independent observational validation of forecasting skill has been performed.
          Always refer to IMD for authoritative alerts.
        </p>
      </footer>
    </div>
  )
}
