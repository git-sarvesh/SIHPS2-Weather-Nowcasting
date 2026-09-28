/**
 * Real geographic basemap styles and the Uttarakhand camera.
 *
 * The regression these guard against: the Risk Map shipped with MapLibre's
 * `demotiles` demonstration style, which draws each country as one flat pastel
 * polygon. It is a test fixture, not cartography, and it produced the "flat
 * pastel silhouette" basemap. These tests pin the replacement providers, their
 * keyless nature, the layer order, and the camera.
 */

import { describe, expect, it } from 'vitest'

import type { LayerSpecification, StyleSpecification } from 'maplibre-gl'

import {
  BASEMAP_ATTRIBUTION,
  OFM_GLYPHS_URL,
  SATELLITE_TILE_URL,
  STREET_STYLE_URL,
  TERRAIN_DEM_URL,
  TOPO_TILE_URL,
  UTTARAKHAND_BOUNDS,
  UTTARAKHAND_CENTRE,
  basemapChoices,
  buildSatelliteStyle,
  buildTerrainStyle,
} from '../lib/basemapStyles'
import { uttarakhandCamera } from '../lib/mapStyle'

const layersOf = (style: StyleSpecification): LayerSpecification[] => style.layers ?? []

describe('basemap styles', () => {
  it('never references the demotiles demonstration style', () => {
    const serialised = JSON.stringify([
      basemapChoices(),
      buildSatelliteStyle(),
      buildTerrainStyle(),
      STREET_STYLE_URL,
    ])
    expect(serialised).not.toContain('demotiles')
  })

  it('exposes satellite, terrain and street basemaps', () => {
    expect(basemapChoices().map((b) => b.id)).toEqual(['satellite', 'terrain', 'street'])
  })

  it('uses keyless public tile endpoints and needs no credentials', () => {
    const serialised = JSON.stringify(basemapChoices())
    expect(serialised).toContain('server.arcgisonline.com')
    expect(serialised).toContain('opentopomap.org')
    expect(serialised).toContain('tiles.openfreemap.org')
    // No provider token or key placeholder may appear anywhere.
    expect(serialised).not.toMatch(/api[_-]?key/i)
    expect(serialised).not.toMatch(/access_token/i)
  })

  it('credits the tile providers for attribution', () => {
    expect(BASEMAP_ATTRIBUTION).toContain('OpenFreeMap')
    expect(BASEMAP_ATTRIBUTION).toContain('OpenTopoMap')
  })

  it('builds a valid style with imagery, hillshade and a label overlay', () => {
    for (const style of [buildSatelliteStyle(), buildTerrainStyle()]) {
      expect(style.version).toBe(8)
      expect(style.glyphs).toBe(OFM_GLYPHS_URL)
      expect(layersOf(style).length).toBeGreaterThan(4)

      const types = layersOf(style).map((l) => l.type)
      expect(types).toContain('raster')
      expect(types).toContain('hillshade')
      expect(types).toContain('symbol')

      // The DEM source must be wired for MapLibre's hillshade.
      const dem = style.sources?.dem as { type: string; encoding?: string }
      expect(dem.type).toBe('raster-dem')
      expect(dem.encoding).toBe('terrarium')
    }
  })

  it('orders imagery, then hillshade, then labels', () => {
    // Labels must come last so relief shading never paints over them.
    const ids = layersOf(buildSatelliteStyle()).map((l) => l.id)
    const raster = ids.indexOf('satellite-imagery')
    const shade = ids.indexOf('hillshade')
    const label = ids.findIndex((id) => id.startsWith('ovl-label'))
    expect(raster).toBeGreaterThanOrEqual(0)
    expect(raster).toBeLessThan(shade)
    expect(shade).toBeLessThan(label)
  })

  it('draws real administrative boundaries from the vector tile source', () => {
    const boundaries = layersOf(buildSatelliteStyle()).filter((l) =>
      l.id.startsWith('ovl-boundary'),
    )
    expect(boundaries.length).toBeGreaterThan(0)
    for (const layer of boundaries) {
      // `source-layer: boundary` is what makes these genuine OSM boundaries
      // rather than an invented outline.
      expect((layer as { 'source-layer'?: string })['source-layer']).toBe('boundary')
    }
  })

  it('points the DEM and topo services at the documented URLs', () => {
    expect(TERRAIN_DEM_URL).toContain('terrarium')
    expect(TOPO_TILE_URL).toContain('opentopomap.org')
    expect(SATELLITE_TILE_URL).toContain('{z}')
  })

  it('accepts a custom satellite tile service', () => {
    const custom = 'https://tiles.example.invalid/imagery/{z}/{x}/{y}.png'
    const style = buildSatelliteStyle(custom)
    expect(JSON.stringify(style.sources)).toContain('tiles.example.invalid')
  })
})

describe('Uttarakhand camera', () => {
  it('centres on Uttarakhand, not the whole subcontinent', () => {
    expect(UTTARAKHAND_CENTRE.lon).toBeGreaterThan(77)
    expect(UTTARAKHAND_CENTRE.lon).toBeLessThan(82)
    expect(UTTARAKHAND_CENTRE.lat).toBeGreaterThan(28)
    expect(UTTARAKHAND_CENTRE.lat).toBeLessThan(32)
  })

  it('spans a regional extent, not a continental one', () => {
    const lonSpan = UTTARAKHAND_BOUNDS.maxLon - UTTARAKHAND_BOUNDS.minLon
    expect(lonSpan).toBeLessThan(6)
    expect(lonSpan).toBeGreaterThan(2)
  })

  it('zooms out on narrow viewports and in on wide ones', () => {
    const narrow = uttarakhandCamera(420)
    const wide = uttarakhandCamera(1600)
    expect(narrow.zoom).toBeLessThan(wide.zoom)
    // A regional view: far tighter than the previous world-scale 6.6 default.
    expect(wide.zoom).toBeGreaterThan(7)
  })

  it('centres on Uttarakhand and returns usable fallback bounds', () => {
    const camera = uttarakhandCamera(1200)
    expect(camera.center[0]).toBeCloseTo(UTTARAKHAND_CENTRE.lon)
    expect(camera.center[1]).toBeCloseTo(UTTARAKHAND_CENTRE.lat)
    expect(camera.bounds.minLon).toBeLessThan(camera.bounds.maxLon)
    expect(camera.bounds.minLat).toBeLessThan(camera.bounds.maxLat)
  })
})
