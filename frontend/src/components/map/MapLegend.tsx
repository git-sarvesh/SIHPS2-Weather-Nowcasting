/** Precipitation intensity legend and the map tool cluster. */

import { PRECIP_RAMP, illustrativeMmPerHour, rampCss } from '../../lib/radar'
import { CATEGORY_COLORS, RISK_CATEGORIES } from '../../lib/format'
import type { RiskCategory } from '../../api/types'

/** Continuous gradient strip built from the ramp stops. */
export function PrecipLegend({ visible }: { visible: boolean }) {
  if (!visible) return null
  const stops = PRECIP_RAMP.filter((stop) => stop.from > 0)
  const gradient = `linear-gradient(to right, ${stops
    .map((stop, index) => {
      const offset = (index / (stops.length - 1)) * 100
      return `rgba(${stop.rgba[0]}, ${stop.rgba[1]}, ${stop.rgba[2]}, ${stop.rgba[3] / 255}) ${offset}%`
    })
    .join(', ')})`

  return (
    <div
      className="pointer-events-auto w-52 rounded-xl border border-white/12 bg-slate-950/72 px-3 py-2.5
                 shadow-2xl backdrop-blur-xl"
      aria-label="Forecast risk probability legend"
    >
      <h3 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
        Forecast risk probability
      </h3>

      <div
        className="h-2.5 w-full rounded-full border border-white/10"
        style={{ backgroundImage: gradient }}
        role="img"
        aria-label="Colour scale of forecast risk probability from low in blue through cyan and green, then yellow and orange, to red and magenta for extreme."
      />

      <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-500">
        <span>0</span>
        <span>0.5</span>
        <span>1.0</span>
      </div>

      <p className="mt-2 text-[9px] leading-relaxed text-slate-500">
        Scale is <strong className="text-slate-400">model risk probability</strong>, the quantity the
        backend returns. The mm/h row below is an{' '}
        <strong className="text-slate-400">illustrative equivalence only</strong> — this system does
        not measure rainfall rate.
      </p>

      <ul className="mt-1.5 space-y-0.5">
        {stops.map((stop, index) => {
          const next = stops[index + 1]?.from ?? 1
          return (
            <li key={stop.from} className="flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: rampCss(stop.from + 0.01) }}
              />
              <span className="flex-1 text-[9px] text-slate-400">{stop.label}</span>
              <span className="font-mono text-[9px] text-slate-600">
                {stop.from.toFixed(2)}–{next.toFixed(2)} · ~{illustrativeMmPerHour(next)} mm/h
              </span>
            </li>
          )
        })}
      </ul>

      <div className="mt-2.5 border-t border-white/10 pt-2">
        <h4 className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
          Risk category
        </h4>
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
          {RISK_CATEGORIES.map((category) => (
            <span key={category} className="flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="h-2 w-2 rounded-sm"
                style={{ backgroundColor: CATEGORY_COLORS[category] }}
              />
              <span className="text-[9px] text-slate-400">{category}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export interface MapToolButton {
  label: string
  glyph: string
  onClick: () => void
  disabled?: boolean
  active?: boolean
}

/** Right-side map tools: zoom, reset, compass. */
export function MapTools({ buttons }: { buttons: MapToolButton[] }) {
  return (
    <div
      className="pointer-events-auto flex flex-col gap-1.5"
      aria-label="Map tools"
    >
      {buttons.map((button) => (
        <button
          key={button.label}
          type="button"
          onClick={button.onClick}
          disabled={button.disabled}
          aria-label={button.label}
          title={button.label}
          aria-pressed={button.active}
          className={`flex h-9 w-9 items-center justify-center rounded-lg border text-sm
                      shadow-lg backdrop-blur-xl transition disabled:opacity-40 ${
                        button.active
                          ? 'border-sky-400/50 bg-sky-400/18 text-sky-200'
                          : 'border-white/12 bg-slate-950/72 text-slate-300 hover:border-sky-400/40 hover:text-sky-200'
                      }`}
        >
          <span aria-hidden="true">{button.glyph}</span>
        </button>
      ))}
    </div>
  )
}

/** Category colour used for a cell's risk band, exported for the popup. */
export function categoryColorFor(category: string): string {
  return CATEGORY_COLORS[category as RiskCategory] ?? '#64748b'
}
