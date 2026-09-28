/** Formatting and risk-scale helpers shared across the dashboard. */

import type { GridInfo, RiskCategory } from '../api/types'

/** Risk categories in ascending order, matching the backend's 0-3 codes. */
export const RISK_CATEGORIES: RiskCategory[] = ['LOW', 'MODERATE', 'HIGH', 'EXTREME']

/**
 * Display colour per category.
 *
 * Chosen for contrast on the dark base and ordered so adjacent categories stay
 * distinguishable; they are not a substitute for a colour-blind-safe ramp, so the
 * UI always pairs colour with a text label.
 */
export const CATEGORY_COLORS: Record<RiskCategory, string> = {
  LOW: '#22d3ee',
  MODERATE: '#facc15',
  HIGH: '#fb923c',
  EXTREME: '#f43f5e',
}

/** Tailwind text classes matching {@link CATEGORY_COLORS}. */
export const CATEGORY_TEXT: Record<RiskCategory, string> = {
  LOW: 'text-risk-low',
  MODERATE: 'text-risk-moderate',
  HIGH: 'text-risk-high',
  EXTREME: 'text-risk-extreme',
}

/** Fallback used when the backend threshold list is unavailable. */
export const DEFAULT_THRESHOLDS = [0.3, 0.6, 0.85]

/** Categorise a 0-1 risk value using the backend's thresholds. */
export function categorise(risk: number, thresholds: number[] = DEFAULT_THRESHOLDS): RiskCategory {
  if (risk >= thresholds[2]) return 'EXTREME'
  if (risk >= thresholds[1]) return 'HIGH'
  if (risk >= thresholds[0]) return 'MODERATE'
  return 'LOW'
}

/** Format a 0-1 probability as a percentage string. */
export function formatPercent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** Format a 0-1 value for compact metric tiles. */
export function formatProbability(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

/** Format a lead time in hours for display ("30 min", "3 h"). */
export function formatLeadHours(hours: number): string {
  if (hours < 1) return `${Math.round(hours * 60)} min`
  const rounded = Number.isInteger(hours) ? hours : Number(hours.toFixed(2))
  return `${rounded} h`
}

/** Format an ISO timestamp for display, tolerating an unparsable value. */
export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toISOString().replace('T', ' ').replace(/\.\d+Z$/, 'Z').replace('Z', ' UTC')
}

/** Format an ISO timestamp as a compact relative age ("4 min ago"). */
export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '—'
  const time = new Date(iso).getTime()
  if (Number.isNaN(time)) return iso
  const seconds = Math.round((now - time) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  return `${Math.round(hours / 24)} d ago`
}

/** Truncate a long identifier for display, keeping both ends recognisable. */
export function shortenId(value: string | null | undefined, head = 8, tail = 6): string {
  if (!value) return '—'
  if (value.length <= head + tail + 1) return value
  return `${value.slice(0, head)}…${value.slice(-tail)}`
}

/** Centre of the grid bounding box, used as the initial map view. */
export function gridCentre(grid: GridInfo): { lat: number; lon: number } {
  return {
    lat: (grid.min_lat + grid.max_lat) / 2,
    lon: (grid.min_lon + grid.max_lon) / 2,
  }
}

/** Human label for a hazard key. */
export const HAZARD_LABELS: Record<string, string> = {
  thunderstorm: 'Thunderstorm',
  cloudburst: 'Cloudburst',
  flood: 'Flood',
  flood_risk: 'Flood risk',
  flood_probability: 'Flood probability',
  compound_storm_cloudburst: 'Storm × cloudburst',
  overall: 'Overall risk',
}

/** Pretty-print any structured value for a provenance panel. */
export function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}
