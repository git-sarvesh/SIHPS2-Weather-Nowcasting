/** Floating weather control panel: layers, hazard field, filters, basemap. */

import { LAYERS, RISK_FIELDS, type LayerKey, type MapStyleOption } from './mapTypes'
import { CATEGORY_COLORS, HAZARD_LABELS, RISK_CATEGORIES } from '../../lib/format'
import type { RiskCategory, RiskField } from '../../api/types'

export interface HazardPanelProps {
  layers: Record<LayerKey, boolean>
  onToggleLayer: (key: LayerKey) => void
  riskField: RiskField
  onRiskFieldChange: (field: RiskField) => void
  minCategory: 0 | 1 | 2 | 3
  onMinCategoryChange: (value: 0 | 1 | 2 | 3) => void
  styleOptions: MapStyleOption[]
  activeStyleId: string
  onStyleChange: (id: string) => void
  dataSource: string
  modelVersion: string
  isSynthetic: boolean
  validTime: string
  leadLabel: string
  collapsed: boolean
  onToggleCollapsed: () => void
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-white/10 px-3 py-2.5 last:border-b-0">
      <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
        {title}
      </h3>
      {children}
    </div>
  )
}

export function HazardPanel(props: HazardPanelProps) {
  const {
    layers,
    onToggleLayer,
    riskField,
    onRiskFieldChange,
    minCategory,
    onMinCategoryChange,
    styleOptions,
    activeStyleId,
    onStyleChange,
    dataSource,
    modelVersion,
    isSynthetic,
    validTime,
    leadLabel,
    collapsed,
    onToggleCollapsed,
  } = props

  return (
    <div
      className="pointer-events-auto w-[17.5rem] overflow-hidden rounded-xl border border-white/12
                 bg-slate-950/72 shadow-2xl backdrop-blur-xl"
      aria-label="Weather controls"
    >
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-sky-400">
            ◈
          </span>
          <h2 className="text-[11px] font-bold uppercase tracking-[0.12em] text-slate-200">
            Weather layers
          </h2>
        </div>
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand weather controls' : 'Collapse weather controls'}
          className="rounded p-1 text-slate-400 transition hover:bg-white/10 hover:text-slate-200"
        >
          <span aria-hidden="true">{collapsed ? '▸' : '▾'}</span>
        </button>
      </div>

      {!collapsed && (
        <>
          <Section title="Hazard layers">
            <ul className="space-y-1">
              {LAYERS.map((layer) => {
                const active = layers[layer.key]
                return (
                  <li key={layer.key}>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={active}
                      onClick={() => onToggleLayer(layer.key)}
                      title={layer.hint}
                      className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-1.5
                                  text-left transition ${
                                    active
                                      ? 'border-sky-400/40 bg-sky-400/12 text-sky-200'
                                      : 'border-transparent text-slate-400 hover:bg-white/6 hover:text-slate-200'
                                  }`}
                    >
                      <span
                        aria-hidden="true"
                        className={`text-sm ${active ? 'text-sky-300' : 'text-slate-500'}`}
                      >
                        {layer.glyph}
                      </span>
                      <span className="flex-1 text-[11px] font-semibold">{layer.label}</span>
                      <span
                        aria-hidden="true"
                        className={`relative h-3.5 w-6 shrink-0 rounded-full transition ${
                          active ? 'bg-sky-400/70' : 'bg-slate-600/70'
                        }`}
                      >
                        <span
                          className={`absolute top-0.5 h-2.5 w-2.5 rounded-full bg-white transition-all ${
                            active ? 'left-3' : 'left-0.5'
                          }`}
                        />
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </Section>

          <Section title="Hazard field">
            <select
              className="w-full rounded-lg border border-white/12 bg-slate-900/70 px-2.5 py-1.5
                         text-[11px] text-slate-200 focus:border-sky-400 focus:outline-none"
              value={riskField}
              onChange={(event) => onRiskFieldChange(event.target.value as RiskField)}
              aria-label="Hazard field"
            >
              {RISK_FIELDS.map((field) => (
                <option key={field} value={field}>
                  {HAZARD_LABELS[field] ?? field}
                </option>
              ))}
            </select>
          </Section>

          <Section title="Minimum risk category">
            <div className="grid grid-cols-4 gap-1">
              {RISK_CATEGORIES.map((category, code) => {
                const active = minCategory === code
                return (
                  <button
                    key={category}
                    type="button"
                    aria-pressed={active}
                    onClick={() => onMinCategoryChange(code as 0 | 1 | 2 | 3)}
                    className={`rounded-md border px-1 py-1 text-[9px] font-bold transition ${
                      active
                        ? 'border-transparent text-slate-950'
                        : 'border-white/12 text-slate-400 hover:text-slate-200'
                    }`}
                    style={
                      active
                        ? { backgroundColor: CATEGORY_COLORS[category as RiskCategory] }
                        : undefined
                    }
                    title={`Show ${category} and above`}
                  >
                    {category.slice(0, 3)}
                  </button>
                )
              })}
            </div>
          </Section>

          <Section title="Basemap">
            <select
              className="w-full rounded-lg border border-white/12 bg-slate-900/70 px-2.5 py-1.5
                         text-[11px] text-slate-200 focus:border-sky-400 focus:outline-none"
              value={activeStyleId}
              onChange={(event) => onStyleChange(event.target.value)}
              aria-label="Basemap style"
            >
              {styleOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
            </select>
          </Section>

          <Section title="Data">
            <dl className="space-y-1 text-[10px]">
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Lead</dt>
                <dd className="font-mono text-slate-300">{leadLabel}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Valid</dt>
                <dd className="truncate font-mono text-slate-300">{validTime || '—'}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Model</dt>
                <dd className="truncate font-mono text-slate-300">{modelVersion || '—'}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-slate-500">Source</dt>
                <dd className="truncate text-right text-slate-300">{dataSource || '—'}</dd>
              </div>
            </dl>
            {isSynthetic && (
              <p
                className="mt-2 rounded border border-amber-400/35 bg-amber-400/10 px-2 py-1
                           text-[9px] font-semibold leading-tight text-amber-300"
              >
                SYNTHETIC DEMO — not a live radar or IMD product
              </p>
            )}
          </Section>
        </>
      )}
    </div>
  )
}
