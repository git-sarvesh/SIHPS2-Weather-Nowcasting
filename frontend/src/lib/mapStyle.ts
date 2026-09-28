/**
 * Basemap style resolution and an offline fallback style.
 *
 * The configured style (`VITE_MAP_STYLE_URL`) is a third-party service, so the
 * map must not assume it is reachable. Historically a style failure left an
 * empty dark canvas: MapLibre never fired `load`, every custom source stayed
 * unadded, and nothing told the user why. This module makes the failure
 * explicit and guarantees *some* geographically accurate frame of reference.
 *
 * The fallback is a real MapLibre style built entirely from the backend's own
 * grid bounds - a graticule and the AOI outline. It contains no invented
 * imagery, needs no glyphs, sprite or tiles, and therefore works with no
 * network and no API key.
 */

import type { StyleSpecification } from 'maplibre-gl'

/** How long to wait for the configured style before using the fallback. */
export const STYLE_PROBE_TIMEOUT_MS = 6000

export type StyleProbe = { ok: true } | { ok: false; reason: string }

/**
 * Check that the configured style URL is actually reachable and returns a
 * usable MapLibre style, before the map is constructed.
 *
 * Probing first means a bad style is reported as a bad style, rather than
 * surfacing later as a mysteriously empty map.
 */
export async function probeMapStyle(
  url: string,
  timeoutMs: number = STYLE_PROBE_TIMEOUT_MS,
): Promise<StyleProbe> {
  // An inline style object (or a data/blob URL) is already local.
  if (url.trim().startsWith('{')) {
    try {
      JSON.parse(url)
      return { ok: true }
    } catch {
      return { ok: false, reason: 'inline style is not valid JSON' }
    }
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { signal: controller.signal })
    if (!response.ok) {
      return { ok: false, reason: `HTTP ${response.status} from style URL` }
    }
    const style = (await response.json()) as Partial<StyleSpecification>
    if (typeof style !== 'object' || style === null || !Array.isArray(style.layers)) {
      return { ok: false, reason: 'style response has no layers array' }
    }
    return { ok: true }
  } catch (caught) {
    const message =
      caught instanceof Error && caught.name === 'AbortError'
        ? `timed out after ${timeoutMs} ms`
        : caught instanceof Error
          ? caught.message
          : 'unknown network failure'
    return { ok: false, reason: message }
  } finally {
    clearTimeout(timer)
  }
}

/** Source/layer ids used by the fallback style, so overlays can sit above it. */
export const FALLBACK_AOI_SOURCE = 'sihps-fallback-aoi'
export const FALLBACK_GRID_SOURCE = 'sihps-fallback-grid'

/** Geographic box the fallback style should frame. */
export interface FallbackBounds {
  minLon: number
  minLat: number
  maxLon: number
  maxLat: number
}

/** Choose a round graticule step that yields roughly 4-8 lines per axis. */
function graticuleStep(span: number): number {
  const raw = span / 6
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const normalised = raw / magnitude
  const nice = normalised >= 5 ? 5 : normalised >= 2 ? 2 : 1
  return nice * magnitude
}


/**
 * Build a self-contained style from the backend's real grid bounds.
 *
 * Everything is derived from `bounds`, so the result is an accurate local
 * reference frame for the AOI rather than fabricated cartography.
 */
export function buildOfflineStyle(bounds: FallbackBounds): StyleSpecification {
  const { minLon, minLat, maxLon, maxLat } = bounds
  // Pad so the AOI is not flush against the viewport edge.
  const padLon = Math.max(0.05, (maxLon - minLon) * 0.18)
  const padLat = Math.max(0.05, (maxLat - minLat) * 0.18)
  const west = minLon - padLon
  const east = maxLon + padLon
  const south = minLat - padLat
  const north = maxLat + padLat

  // Graticule: evenly spaced meridians and parallels across the padded box.
  // Coordinates are GeoJSON position arrays ([lon, lat]), not objects.
  const lines: [number, number][][] = []
  const stepLon = graticuleStep(east - west)
  const stepLat = graticuleStep(north - south)

  const firstLon = Math.ceil(west / stepLon) * stepLon
  for (let lon = firstLon; lon <= east; lon += stepLon) {
    lines.push([
      [lon, south],
      [lon, north],
    ])
  }
  const firstLat = Math.ceil(south / stepLat) * stepLat
  for (let lat = firstLat; lat <= north; lat += stepLat) {
    lines.push([
      [west, lat],
      [east, lat],
    ])
  }

  const gridFeature = (coords: [number, number][], index: number) => ({
    type: 'Feature' as const,
    properties: { index },
    geometry: { type: 'LineString' as const, coordinates: coords },
  })

  return {
    version: 8,
    name: 'SIHPS offline reference frame',
    sources: {
      [FALLBACK_GRID_SOURCE]: {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: lines.map(gridFeature),
        },
      },
      [FALLBACK_AOI_SOURCE]: {
        type: 'geojson',
        data: {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: { name: 'SIHPS model AOI' },
              geometry: {
                type: 'Polygon',
                coordinates: [
                  [
                    [minLon, minLat],
                    [maxLon, minLat],
                    [maxLon, maxLat],
                    [minLon, maxLat],
                    [minLon, minLat],
                  ],
                ],
              },
            },
          ],
        },
      },
    },
    layers: [
      {
        id: 'fallback-background',
        type: 'background',
        paint: { 'background-color': '#0b1a2b' },
      },
      {
        id: 'fallback-grid',
        type: 'line',
        source: FALLBACK_GRID_SOURCE,
        paint: {
          'line-color': '#16324a',
          'line-width': 1,
        },
      },
      {
        id: 'fallback-aoi-fill',
        type: 'fill',
        source: FALLBACK_AOI_SOURCE,
        // Outline-only reference frame. A filled box would read as a solid
        // rectangle over the map, which is the artefact this module avoids.
        paint: { 'fill-color': '#0f2f3f', 'fill-opacity': 0.0 },
      },
      {
        id: 'fallback-aoi-line',
        type: 'line',
        source: FALLBACK_AOI_SOURCE,
        paint: {
          'line-color': '#22d3ee',
          'line-width': 1.5,
          'line-dasharray': [3, 2],
          'line-opacity': 0.7,
        },
      },
    ],
  }
}

/**
 * Camera for the operational AOI.
 *
 * `zoom` is derived from the viewport width so the region frames well on a
 * phone and on a desktop without a separate breakpoint. The bounds are clamped
 * so a very wide viewport does not zoom out far enough to lose the Himalaya.
 */
export function uttarakhandCamera(
  width: number,
): { center: [number, number]; zoom: number; bounds: FallbackBounds } {
  const bounds: FallbackBounds = {
    minLon: 77.4,
    minLat: 28.5,
    maxLon: 81.7,
    maxLat: 31.6,
  }
  // ~7.6 fits the AOI across a 1200 px viewport; narrow viewports step down.
  const zoom = width < 640 ? 6.9 : width < 1024 ? 7.4 : 7.8
  return { center: [79.5, 30.05], zoom, bounds }
}
