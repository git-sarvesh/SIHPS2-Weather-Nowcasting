/** Shared types for the radar map controls. */

import type { RiskField } from '../../api/types'

/** Every toggleable layer on the radar map. */
export type LayerKey = 'precipitation' | 'stormCells' | 'floodRisk' | 'riskPolygons'

export interface LayerToggle {
  key: LayerKey
  label: string
  hint: string
  /** Two-letter glyph shown in the control row. */
  glyph: string
}

export const LAYERS: LayerToggle[] = [
  {
    key: 'precipitation',
    label: 'Precipitation',
    hint: 'Smoothed intensity field derived from backend risk polygons',
    glyph: '☂',
  },
  {
    key: 'stormCells',
    label: 'Storm cells',
    hint: 'Model-derived cell markers, strongest first',
    glyph: '◉',
  },
  {
    key: 'floodRisk',
    label: 'Flood risk',
    hint: 'Terrain-aware flood risk field from the risk engine',
    glyph: '≈',
  },
  {
    key: 'riskPolygons',
    label: 'Risk polygons',
    hint: 'Raw GeoJSON contours returned by the backend',
    glyph: '▦',
  },
]

/** Basemap style options. Only the configured style ships by default. */
export interface MapStyleOption {
  id: string
  label: string
  url: string
}

export const RISK_FIELDS: RiskField[] = [
  'overall',
  'thunderstorm',
  'cloudburst',
  'flood_risk',
  'compound_storm_cloudburst',
]

/** A forecast frame available for the timeline. */
export interface TimelineFrame {
  index: number
  leadHours: number
  validTime: string
  /** Ticks of loaded cells at this lead time. */
  cellCount: number
  loaded: boolean
}
