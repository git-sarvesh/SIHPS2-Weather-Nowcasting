/**
 * Real geographic basemap styles for the Uttarakhand AOI.
 *
 * ## Why this module exists
 *
 * The dashboard previously used `https://demotiles.maplibre.org/style.json`.
 * That is MapLibre's *demonstration* style: it draws each country as one flat
 * pastel polygon and contains no roads, rivers, terrain, or place labels. It is
 * a test fixture, not cartography, which is exactly why the map looked like a
 * flat pastel silhouette.
 *
 * ## Providers
 *
 * All three styles below are built from **keyless, CORS-enabled** endpoints, so
 * the default configuration needs no credentials and no paid plan:
 *
 * | Style     | Imagery / cartography                                   | Hillshade |
 * |-----------|---------------------------------------------------------|-----------|
 * | Satellite | Esri World Imagery (raster) + OpenFreeMap vector overlay | Terrarium |
 * | Terrain   | OpenTopoMap (raster) + OpenFreeMap vector overlay       | Terrarium |
 * | Street    | OpenFreeMap "liberty" (full vector style)                | Terrarium |
 *
 * - **OpenFreeMap** (`tiles.openfreemap.org`) serves OpenMapTiles-derived
 *   vector tiles plus public fonts/sprites, with no API key.
 * - **Esri World Imagery** is the public basemap tile service behind Esri's own
 *   web map and is served without a key. It is the one external dependency that
 *   Esri could restrict, so `VITE_SATELLITE_TILE_URL` can point at any
 *   equivalent keyless XYZ raster service.
 * - **OpenTopoMap** renders genuine topographic cartography (contours, relief
 *   shading, land cover) as keyless raster tiles.
 * - **Terrarium** elevation tiles (`elevation-tiles-prod`, AWS Open Data) are
 *   Mapzen-derived elevation rasters. MapLibre reads the `terrarium` encoding
 *   natively, so no decoding is required here.
 *
 * ## Honesty constraints
 *
 * These are genuine geographic basemaps. None of them fabricate terrain: if a
 * tile service is unreachable the caller (`probeMapStyle`) reports a real
 * failure and the UI says so, rather than silently substituting a solid colour.
 */

import type { StyleSpecification } from 'maplibre-gl'

/** OpenMapTiles-derived vector tiles: roads, rivers, boundaries, place labels. */
export const OFM_VECTOR_URL = 'https://tiles.openfreemap.org/planet'

/** Public glyph (font) endpoint resolving `{fontstack}` and `{range}`. */
export const OFM_GLYPHS_URL = 'https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf'

/** Public sprite endpoint for icons referenced by the vector overlay. */
export const OFM_SPRITE_URL = 'https://tiles.openfreemap.org/sprites/ofm_f384/ofm'

/** Keyless satellite/orthophoto raster service. Overridable via env. */
export const SATELLITE_TILE_URL =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

/** Keyless topographic raster service (contours, relief, land cover). */
export const TOPO_TILE_URL = 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png'

/** Terrarium-encoded elevation raster used for `raster-dem` + `hillshade`. */
export const TERRAIN_DEM_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'

/** Full OpenFreeMap vector style, used directly for the street basemap. */
export const STREET_STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty'

/** Font stacks available from the public OpenFreeMap glyph endpoint. */
const FONT_REGULAR = ['Noto Sans Regular']
const FONT_ITALIC = ['Noto Sans Italic']
const FONT_BOLD = ['Noto Sans Bold']

/** Operational AOI: Uttarakhand and the surrounding Himalayan belt. */
export const UTTARAKHAND_BOUNDS = {
  minLon: 77.4,
  minLat: 28.5,
  maxLon: 81.7,
  maxLat: 31.6,
} as const

/** Geographic centre of Uttarakhand, used as the default camera. */
export const UTTARAKHAND_CENTRE = { lon: 79.5, lat: 30.05 }

/**
 * Attribution for every basemap.
 *
 * Required by the tile providers. MapLibre renders this through the built-in
 * attribution control; the dashboard shows synthetic-data provenance separately.
 */
export const BASEMAP_ATTRIBUTION =
  '<a href="https://openfreemap.org/" target="_blank" rel="noreferrer">OpenFreeMap</a>' +
  ' &middot; <a href="https://opentopomap.org/" target="_blank" rel="noreferrer">OpenTopoMap</a>' +
  ' &middot; Esri World Imagery &middot; Terrarium elevation (Mapzen, AWS Open Data)'

/** Vector-tile source shared by the satellite and terrain overlays. */
const vectorSource = {
  ofm: { type: 'vector' as const, url: OFM_VECTOR_URL },
}

/** A keyless raster source. */
function rasterSource(tiles: string[], maxzoom: number, tileSize = 256) {
  return { type: 'raster' as const, tiles, tileSize, maxzoom }
}

/** Terrarium DEM source, wired for hillshade. */
const demSource = {
  type: 'raster-dem' as const,
  tiles: [TERRAIN_DEM_URL],
  tileSize: 256,
  encoding: 'terrarium' as const,
  maxzoom: 15,
}

/**
 * Vector overlay drawn *on top of* imagery: administrative boundaries, rivers,
 * and place labels.
 *
 * Imagery alone gives terrain but no orientation detail, so this is what makes
 * the satellite and terrain styles read as maps rather than photographs. Roads
 * are deliberately omitted here: over orthophoto they become visual noise that
 * competes with the weather overlay.
 */
function overlayLayers(): NonNullable<StyleSpecification['layers']> {
  return [
    {
      id: 'ovl-boundary-country',
      type: 'line',
      source: 'ofm',
      'source-layer': 'boundary',
      filter: ['<=', ['get', 'admin_level'], 4] as never,
      paint: {
        'line-color': 'rgba(226,232,240,0.55)',
        'line-width': ['interpolate', ['linear'], ['zoom'], 4, 0.8, 10, 1.8] as never,
        'line-dasharray': [3, 2],
      },
    },
    {
      id: 'ovl-boundary-state',
      type: 'line',
      source: 'ofm',
      'source-layer': 'boundary',
      filter: ['==', ['get', 'admin_level'], 4] as never,
      paint: {
        'line-color': 'rgba(148,163,184,0.5)',
        'line-width': ['interpolate', ['linear'], ['zoom'], 4, 0.7, 10, 1.5] as never,
      },
    },
    {
      id: 'ovl-water',
      type: 'fill',
      source: 'ofm',
      'source-layer': 'water',
      paint: { 'fill-color': 'rgba(56,130,246,0.35)' },
    },
    {
      id: 'ovl-river',
      type: 'line',
      source: 'ofm',
      'source-layer': 'waterway',
      filter: ['==', ['get', 'class'], 'river'] as never,
      paint: {
        'line-color': 'rgba(125,211,252,0.8)',
        'line-width': ['interpolate', ['exponential', 1.2], ['zoom'], 6, 0.6, 14, 3] as never,
      },
    },
    {
      id: 'ovl-label-village',
      type: 'symbol',
      source: 'ofm',
      'source-layer': 'place',
      minzoom: 9,
      filter: ['==', ['get', 'class'], 'village'] as never,
      layout: {
        'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']] as never,
        'text-font': FONT_REGULAR as never,
        'text-size': ['interpolate', ['linear'], ['zoom'], 9, 10, 14, 13] as never,
        'text-anchor': 'top',
        'text-offset': [0, 0.4],
      },
      paint: { 'text-color': '#f8fafc', 'text-halo-color': 'rgba(2,6,23,0.9)', 'text-halo-width': 1.4 },
    },
    {
      id: 'ovl-label-town',
      type: 'symbol',
      source: 'ofm',
      'source-layer': 'place',
      minzoom: 6,
      filter: ['==', ['get', 'class'], 'town'] as never,
      layout: {
        'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']] as never,
        'text-font': FONT_REGULAR as never,
        'text-size': ['interpolate', ['linear'], ['zoom'], 6, 11, 14, 15] as never,
        'text-anchor': 'top',
        'text-offset': [0, 0.4],
      },
      paint: { 'text-color': '#f8fafc', 'text-halo-color': 'rgba(2,6,23,0.9)', 'text-halo-width': 1.5 },
    },
    {
      id: 'ovl-label-city',
      type: 'symbol',
      source: 'ofm',
      'source-layer': 'place',
      minzoom: 4,
      filter: ['==', ['get', 'class'], 'city'] as never,
      layout: {
        'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']] as never,
        'text-font': FONT_BOLD as never,
        'text-size': ['interpolate', ['linear'], ['zoom'], 4, 12, 12, 18] as never,
        'text-anchor': 'top',
        'text-offset': [0, 0.5],
      },
      paint: { 'text-color': '#ffffff', 'text-halo-color': 'rgba(2,6,23,0.95)', 'text-halo-width': 1.8 },
    },
    {
      id: 'ovl-label-state',
      type: 'symbol',
      source: 'ofm',
      'source-layer': 'place',
      minzoom: 4,
      maxzoom: 9,
      filter: ['==', ['get', 'class'], 'state'] as never,
      layout: {
        'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']] as never,
        'text-font': FONT_ITALIC as never,
        'text-size': ['interpolate', ['linear'], ['zoom'], 4, 11, 8, 15] as never,
        'text-transform': 'uppercase',
        'text-letter-spacing': 0.15,
      },
      paint: { 'text-color': 'rgba(226,232,240,0.9)', 'text-halo-color': 'rgba(2,6,23,0.9)', 'text-halo-width': 1.4 },
    },
  ]
}

/**
 * Shared tail of the inline styles: hillshade then the vector overlay.
 *
 * Layer order matters: imagery, then relief shading to reveal mountain
 * structure, then hydrography and labels on top.
 */
function baseLayers(imageryId: string): NonNullable<StyleSpecification['layers']> {
  return [
    { id: 'background', type: 'background', paint: { 'background-color': '#0b1220' } },
    {
      id: imageryId,
      type: 'raster',
      source: 'imagery',
      paint: {
        'raster-opacity': 1,
        // A short cross-fade hides the tile pop-in that satellite imagery shows
        // while panning, without lagging behind the camera.
        'raster-fade-duration': 250,
      },
    },
    // Terrain relief. `hillshade` derives lighting from a `raster-dem` source, so
    // mountains and valleys get real shading rather than a painted-on effect.
    {
      id: 'hillshade',
      type: 'hillshade',
      source: 'dem',
      paint: {
        'hillshade-illumination-direction': 315,
        'hillshade-illumination-anchor': 'map',
        // Kept subtle so the weather overlay stays the brightest thing on screen.
        'hillshade-shadow-color': '#0b1220',
        'hillshade-highlight-color': '#ffffff',
        'hillshade-exaggeration': 0.28,
        'hillshade-accent-color': '#1e293b',
      },
    },
    ...overlayLayers(),
  ]
}

/** Satellite basemap: real orthophoto imagery with relief and place labels. */
export function buildSatelliteStyle(satelliteUrl = SATELLITE_TILE_URL): StyleSpecification {
  return {
    version: 8,
    name: 'SIHPS satellite',
    glyphs: OFM_GLYPHS_URL,
    sprite: OFM_SPRITE_URL,
    sources: {
      imagery: rasterSource([satelliteUrl], 19),
      dem: demSource,
      ...vectorSource,
    },
    layers: baseLayers('satellite-imagery'),
  }
}

/** Terrain basemap: OpenTopoMap relief cartography with hillshade. */
export function buildTerrainStyle(topoUrl = TOPO_TILE_URL): StyleSpecification {
  return {
    version: 8,
    name: 'SIHPS terrain',
    glyphs: OFM_GLYPHS_URL,
    sprite: OFM_SPRITE_URL,
    sources: {
      imagery: rasterSource([topoUrl], 17),
      dem: demSource,
      ...vectorSource,
    },
    layers: baseLayers('topo-imagery'),
  }
}

/** A basemap the style selector can switch between. */
export interface BasemapChoice {
  id: string
  label: string
  /** Short description shown under the selector. */
  hint: string
  /** Resolves to a style URL or an inline style object. */
  style: string | StyleSpecification
  /** True when the basemap is dark, which drives overlay contrast treatment. */
  dark: boolean
}

/**
 * The basemaps offered by the style selector, in display order.
 *
 * Satellite leads because it is the mode an operational user reads first:
 * terrain is what actually matters for orographic precipitation in Uttarakhand.
 */
export function basemapChoices(satelliteUrl = SATELLITE_TILE_URL): BasemapChoice[] {
  return [
    {
      id: 'satellite',
      label: 'Satellite',
      hint: 'Orthophoto imagery with hillshade, rivers and place labels',
      style: buildSatelliteStyle(satelliteUrl),
      dark: true,
    },
    {
      id: 'terrain',
      label: 'Terrain',
      hint: 'OpenTopoMap relief with hillshade, rivers and boundaries',
      style: buildTerrainStyle(),
      dark: false,
    },
    {
      id: 'street',
      label: 'Street',
      hint: 'OpenFreeMap vector cartography: roads, towns, rivers, boundaries',
      style: STREET_STYLE_URL,
      dark: false,
    },
  ]
}
