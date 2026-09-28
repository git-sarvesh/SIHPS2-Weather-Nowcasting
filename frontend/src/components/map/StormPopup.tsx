/** Storm-cell popup content, rendered inside a MapLibre Popup. */

import { HAZARD_LABELS, formatLeadHours, formatProbability } from '../../lib/format'
import { illustrativeMmPerHour, rampCss } from '../../lib/radar'
import { categoryColorFor } from './MapLegend'
import type { StormCell } from '../../lib/radar'

/**
 * Build the popup DOM for a storm cell.
 *
 * Shows only values the backend actually returned. Where a conventional field
 * (rainfall rate) is not available it is omitted rather than invented, and the
 * illustrative mm/h figure is explicitly marked as such.
 */
export function buildCellPopup(cell: StormCell, modelVersion: string): HTMLElement {
  const root = document.createElement('div')
  root.className = 'sihps-popup'
  root.style.cssText = 'font-family: ui-monospace, monospace; font-size: 11px; line-height: 1.55; min-width: 190px'

  const color = categoryColorFor(cell.riskCategory)

  const rows: [string, string][] = [
    ['Hazard', HAZARD_LABELS[cell.hazard] ?? cell.hazard],
    ['Event type', cell.eventType],
    ['Risk category', cell.riskCategory],
    ['Peak risk', formatProbability(cell.riskMax)],
    ['Mean risk', formatProbability(cell.riskMean)],
    ['Cells', String(cell.nCells)],
    ['Area', `${cell.areaKm2.toFixed(1)} km²`],
    ['Lead time', formatLeadHours(cell.leadHours)],
    ['Valid', cell.validTime ? cell.validTime.replace('T', ' ').replace(/\+00:00$/, ' UTC') : '—'],
  ]

  root.innerHTML = `
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
      <span style="width:8px;height:8px;border-radius:50%;background:${color};
                   box-shadow:0 0 8px ${color}"></span>
      <strong style="font-size:12px;letter-spacing:0.02em">${cell.riskCategory} risk cell</strong>
    </div>
    <div style="height:4px;border-radius:2px;margin-bottom:7px;
                background:linear-gradient(to right, rgba(34,211,238,0.15), ${rampCss(cell.riskMax)})"></div>
    <table style="border-collapse:collapse;width:100%">
      ${rows
        .map(
          ([label, value]) => `<tr>
        <td style="color:#94a3b8;padding-right:10px;white-space:nowrap">${label}</td>
        <td style="text-align:right;color:#e2e8f0">${value}</td>
      </tr>`,
        )
        .join('')}
      <tr>
        <td style="color:#94a3b8;padding-right:10px">Intensity (ill.)</td>
        <td style="text-align:right;color:#fbbf24">~${illustrativeMmPerHour(cell.riskMax)} mm/h</td>
      </tr>
    </table>
    <div style="margin-top:7px;padding-top:6px;border-top:1px solid rgba(148,163,184,0.22);
                color:#94a3b8;font-size:10px">
      ${cell.lat.toFixed(3)}°N, ${cell.lon.toFixed(3)}°E · ${modelVersion || 'model n/a'}
    </div>
    <div style="margin-top:5px;color:#facc15;font-weight:600;font-size:10px">
      SYNTHETIC DEMO — model output, not an observation
    </div>
  `
  return root
}
