/**
 * Basemap style resolution.
 *
 * These cover the regression where an unreachable style host left the Risk Map
 * as an empty dark canvas: `probeMapStyle` must detect that, and
 * `buildOfflineStyle` must still produce a usable, self-contained style.
 */

import { describe, expect, it, vi, afterEach } from 'vitest'
import {
  buildOfflineStyle,
  probeMapStyle,
  FALLBACK_AOI_SOURCE,
  FALLBACK_GRID_SOURCE,
  type FallbackBounds,
} from '../lib/mapStyle'

const BOUNDS: FallbackBounds = {
  minLon: 78.4,
  minLat: 29.4,
  maxLon: 81.0,
  maxLat: 31.5,
}

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('probeMapStyle', () => {
  it('accepts a reachable style with a layers array', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ version: 8, layers: [] }))),
    )
    await expect(probeMapStyle('https://example.test/style.json')).resolves.toEqual({ ok: true })
  })

  it('rejects a non-2xx response and reports the status', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 503 })))
    const result = await probeMapStyle('https://example.test/style.json')
    expect(result.ok).toBe(false)
    expect(result.ok === false && result.reason).toContain('503')
  })

  it('rejects JSON that is not a MapLibre style', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ message: 'not a style' }))),
    )
    const result = await probeMapStyle('https://example.test/style.json')
    expect(result.ok).toBe(false)
    expect(result.ok === false && result.reason).toContain('layers')
  })

  it('reports a network failure rather than throwing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    const result = await probeMapStyle('https://example.test/style.json')
    expect(result.ok).toBe(false)
    expect(result.ok === false && result.reason).toContain('Failed to fetch')
  })

  it('accepts an inline style without touching the network', async () => {
    const spy = vi.fn()
    vi.stubGlobal('fetch', spy)
    const inline = JSON.stringify(buildOfflineStyle(BOUNDS))
    await expect(probeMapStyle(inline)).resolves.toEqual({ ok: true })
    expect(spy).not.toHaveBeenCalled()
  })

  it('rejects malformed inline JSON', async () => {
    vi.stubGlobal('fetch', vi.fn())
    const result = await probeMapStyle('{not json')
    expect(result.ok).toBe(false)
  })
})

describe('buildOfflineStyle', () => {
  it('produces a self-contained style needing no external resources', () => {
    const style = buildOfflineStyle(BOUNDS)
    expect(style.version).toBe(8)
    // No glyphs/sprite/tiles means it renders with no network at all.
    expect(style.glyphs).toBeUndefined()
    expect(style.sprite).toBeUndefined()
    for (const source of Object.values(style.sources)) {
      expect(source.type).toBe('geojson')
    }
  })

  it('draws the real AOI rectangle from the supplied bounds', () => {
    const style = buildOfflineStyle(BOUNDS)
    const source = style.sources[FALLBACK_AOI_SOURCE]
    if (source.type !== 'geojson') throw new Error('expected geojson source')
    const feature = source.data.features[0]
    if (feature.geometry.type !== 'Polygon') throw new Error('expected polygon')
    const ring = feature.geometry.coordinates[0]
    expect(ring[0]).toEqual([BOUNDS.minLon, BOUNDS.minLat])
    expect(ring[2]).toEqual([BOUNDS.maxLon, BOUNDS.maxLat])
    // Ring must be closed.
    expect(ring[ring.length - 1]).toEqual(ring[0])
  })

  it('emits a graticule of meridians and parallels', () => {
    const style = buildOfflineStyle(BOUNDS)
    const source = style.sources[FALLBACK_GRID_SOURCE]
    if (source.type !== 'geojson') throw new Error('expected geojson source')
    const features = source.data.features
    expect(features.length).toBeGreaterThanOrEqual(4)
    for (const feature of features) {
      expect(feature.geometry.type).toBe('LineString')
      const points = feature.geometry.coordinates
      expect(points).toHaveLength(2)
      const [a, b] = points
      // Each line is either a constant-longitude meridian or a
      // constant-latitude parallel.
      const isMeridian = a[0] === b[0] && a[1] !== b[1]
      const isParallel = a[1] === b[1] && a[0] !== b[0]
      expect(isMeridian || isParallel).toBe(true)
    }
  })

  it('orders the background first so overlays paint above it', () => {
    const style = buildOfflineStyle(BOUNDS)
    expect(style.layers[0]).toMatchObject({ type: 'background' })
  })
})
