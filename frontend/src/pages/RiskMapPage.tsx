/**
 * Radar weather map.
 *
 * Full-screen command-centre view driven entirely by the existing backend:
 * `POST /risk/geojson` supplies the polygons, `POST /forecast` supplies the
 * per-lead-time statistics, and `GET /health` supplies the grid and model status.
 *
 * Design notes
 * ------------
 * - **Frames are real.** One GeoJSON response is fetched per available lead
 *   time and cached. Playback interpolates *between those responses*; no frame
 *   is invented and no random motion is used.
 * - **No wind or lightning layers.** The API exposes no wind field and no
 *   lightning data, so neither is simulated.
 * - **Animation is off the React render path.** The interpolation position lives
 *   in a ref and is written straight to the MapLibre image source, so a 60 fps
 *   loop causes at most a few re-renders per second.
 * - Everything is torn down on unmount: animation frame, listeners, popup and
 *   the MapLibre instance.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import maplibregl, { type ImageSource, type Map as MapLibreMap } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'

import type { GeoJsonResponse, RiskField } from '../api/types'
import { HazardPanel } from '../components/map/HazardPanel'
import { MapTools, PrecipLegend } from '../components/map/MapLegend'
import { RadarTimeline } from '../components/map/RadarTimeline'
import { buildCellPopup } from '../components/map/StormPopup'
import { MapNoticeBanner, type MapNotice } from '../components/map/MapNoticeBanner'
import { describeError } from '../components/ui/States'
import type { LayerKey, MapStyleOption, TimelineFrame } from '../components/map/mapTypes'
import { formatLeadHours, formatProbability } from '../lib/format'
import {
  blendStormCells,
  extractStormCells,
  rasterisePrecip,
  type StormCell,
} from '../lib/radar'
import { useApp } from '../state/AppContext'
import { MAP_STYLE_PRESET, SATELLITE_TILE_URL } from '../config'
import { basemapChoices, BASEMAP_ATTRIBUTION, UTTARAKHAND_CENTRE, type BasemapChoice } from '../lib/basemapStyles'
import {
  buildOfflineStyle,
  probeMapStyle,
  uttarakhandCamera,
  type FallbackBounds,
} from '../lib/mapStyle'

const PRECIP_SOURCE = 'sihps-precip'
const POLY_SOURCE = 'sihps-poly'
const POLY_FILL = 'sihps-poly-fill'
const POLY_LINE = 'sihps-poly-line'
const CELL_SOURCE = 'sihps-cells'
const CELL_LAYER = 'sihps-cell-layer'
const CELL_GLOW = 'sihps-cell-glow'

/** 1x1 transparent PNG used to seed the image source before the first frame. */
const TRANSPARENT_PIXEL =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

/** Milliseconds each lead-time step takes during playback. */
const FRAME_DURATION_MS = 1400

/**
 * Basemap options, built from the real tile providers in `lib/basemapStyles`.
 *
 * The style is carried as a value (URL string or inline `StyleSpecification`)
 * rather than only a URL, because the satellite and terrain styles are assembled
 * in-process: they combine a raster imagery source, a Terrarium DEM for
 * hillshade, and an OpenFreeMap vector overlay for labels and boundaries.
 */
const BASEMAPS: BasemapChoice[] = basemapChoices(SATELLITE_TILE_URL)

/** Options shaped for the existing `HazardPanel` select control. */
const STYLE_OPTIONS: MapStyleOption[] = BASEMAPS.map((b) => ({
  id: b.id,
  label: b.label,
  url: typeof b.style === 'string' ? b.style : '',
}))

/**
 * Apply a basemap style to a live map, falling back to the offline reference
 * frame when the provider is unreachable.
 *
 * The style selector and the initial map construction must share this path:
 * calling `map.setStyle(url)` directly reintroduces exactly the blank-map
 * failure this module exists to prevent.
 *
 * Inline styles (satellite, terrain) are assembled in-process, so they are not
 * probed over the network - only remote URL styles are.
 *
 * @returns whether the fallback was used, and the probe failure reason.
 */
async function applyBasemapStyle(
  map: MapLibreMap,
  style: string | maplibregl.StyleSpecification,
  bounds: FallbackBounds,
): Promise<{ usedFallback: boolean; reason: string | null }> {
  if (typeof style !== 'string') {
    map.setStyle(style)
    return { usedFallback: false, reason: null }
  }
  const probe = await probeMapStyle(style)
  if (probe.ok) {
    map.setStyle(style)
    return { usedFallback: false, reason: null }
  }
  map.setStyle(buildOfflineStyle(bounds))
  return { usedFallback: true, reason: probe.reason }
}

/** Per-lead-time frame cache, filled once and reused by playback. */
interface CachedFrame {
  cells: StormCell[]
  features: GeoJsonResponse['features']
  isSynthetic: boolean
  dataSource: string
  modelVersion: string
  validTime: string
}

export function RiskMapPage() {
  const { client, forecast, health, runForecast, connection } = useApp()

  const mapContainer = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const popupRef = useRef<maplibregl.Popup | null>(null)
  const rafRef = useRef<number | null>(null)
  const frameCache = useRef<Map<number, CachedFrame>>(new Map())
  const positionRef = useRef(0)
  const lastTickRef = useRef<number | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [loadingFrames, setLoadingFrames] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    precipitation: true,
    stormCells: true,
    floodRisk: true,
    riskPolygons: true,
  })
  /** Latest layer toggles, readable from the rAF loop without re-binding it. */
  const layersRef = useRef(layers)
  const [riskField, setRiskField] = useState<RiskField>('overall')
  const [minCategory, setMinCategory] = useState<0 | 1 | 2 | 3>(0)
  const [activeStyleId, setActiveStyleId] = useState(MAP_STYLE_PRESET)
  const [panelCollapsed, setPanelCollapsed] = useState(false)
  const [reducedMotion, setReducedMotion] = useState(false)
  /**
   * Map lifecycle. `initialising` and `style-unavailable` are rendered states,
   * so a failed basemap can never masquerade as a healthy, empty map.
   */
  const [mapStatus, setMapStatus] = useState<'initialising' | 'ready' | 'style-unavailable'>(
    'initialising',
  )
  /** Why the configured style could not be used, or '' when it is fine. */
  const [styleIssue, setStyleIssue] = useState('')
  /**
   * True when the offline reference frame replaced the configured basemap.
   * The map is fully usable, but it has no roads/terrain/labels, so the UI
   * must say so rather than let a plain grid masquerade as a street basemap.
   */
  const [usingFallback, setUsingFallback] = useState(false)
  /** Incremented by the retry button to re-run style resolution. */
  const [styleRetry, setStyleRetry] = useState(0)

  const grid = forecast?.grid ?? health?.settings.grid ?? null

  /** Currently selected basemap, for notices and the panel label. */
  const activeBasemap = useMemo(
    () => BASEMAPS.find((b) => b.id === activeStyleId) ?? BASEMAPS[0],
    [activeStyleId],
  )

  layersRef.current = layers

  const rasterBounds = useMemo(() => {
    if (!grid) return null
    return {
      minLon: grid.min_lon,
      minLat: grid.min_lat,
      maxLon: grid.max_lon,
      maxLat: grid.max_lat,
    }
  }, [grid])

  // Honour the OS reduced-motion preference.
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!query) return
    const apply = () => setReducedMotion(query.matches)
    apply()
    query.addEventListener('change', apply)
    return () => query.removeEventListener('change', apply)
  }, [])

  // ------------------------------------------------------------- map setup
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return
    const container = mapContainer.current
    // Guards against the async style probe resolving after unmount.
    let cancelled = false

    // The configured style is a third-party service. Resolve it *before*
    // building the map so a failure is reported explicitly and the map still
    // gets a real (local) basemap instead of an empty canvas.
    const start = async () => {
      setMapStatus('initialising')

      // The configured basemap. `VITE_MAP_STYLE_PRESET` selects one of the real
      // providers; an unknown value falls back to satellite rather than failing.
      const choice =
        BASEMAPS.find((b) => b.id === MAP_STYLE_PRESET) ?? BASEMAPS[0]
      setActiveStyleId(choice.id)

      // Probe only remote styles; inline styles have no fetch to fail.
      const probe =
        typeof choice.style === 'string' ? await probeMapStyle(choice.style) : { ok: true as const }
      if (cancelled) return

      const camera = uttarakhandCamera(container.clientWidth || 1200)
      const offlineBounds: FallbackBounds = camera.bounds

      const style = probe.ok ? choice.style : buildOfflineStyle(offlineBounds)
      setStyleIssue(probe.ok ? '' : probe.reason)
      setUsingFallback(!probe.ok)
      if (!probe.ok) {
        console.warn(
          `[sihps] basemap "${choice.id}" unavailable (${probe.reason}); ` +
            'using the offline reference frame instead.',
        )
      }

      const map = new maplibregl.Map({
        container,
        style,
        center: camera.center,
        zoom: camera.zoom,
        attributionControl: false,
        // North-up meteorological view: rotation would fight the radar read.
        dragRotate: false,
        pitchWithRotate: false,
        maxPitch: 0,
      })
      // Required attribution for the tile providers.
      map.addControl(
        new maplibregl.AttributionControl({ compact: true, customAttribution: BASEMAP_ATTRIBUTION }),
      )
      map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left')
      mapRef.current = map

      const onLoad = () => {
        // Overlays are inserted *before* the basemap's first symbol layer, so
        // place labels stay legible on top of the weather raster instead of being
        // painted over by it. MapLibre's own layer order is otherwise append-only,
        // so without this the overlays would bury every label in the style.
        const firstSymbolLayer = map
          .getStyle()
          .layers?.find((layer) => layer.type === 'symbol')?.id

        // 1. Smoothed precipitation field (an ImageSource, so frames can be
        //    swapped without touching the basemap or recreating the map).
        if (!map.getSource(PRECIP_SOURCE) && rasterBounds) {
          map.addSource(PRECIP_SOURCE, {
            type: 'image',
            url: TRANSPARENT_PIXEL,
            coordinates: [
              [rasterBounds.minLon, rasterBounds.maxLat],
              [rasterBounds.maxLon, rasterBounds.maxLat],
              [rasterBounds.maxLon, rasterBounds.minLat],
              [rasterBounds.minLon, rasterBounds.minLat],
            ],
          })
          map.addLayer(
            {
              id: PRECIP_SOURCE,
              type: 'raster',
              source: PRECIP_SOURCE,
              paint: {
                // Kept below full strength so hills, rivers and roads stay
                // visible through the precipitation field.
                'raster-opacity': 0.72,
                'raster-fade-duration': 0,
              },
            },
            firstSymbolLayer,
          )
        }

        // 2. Raw backend contours, kept as a toggleable categorical layer.
        if (!map.getSource(POLY_SOURCE)) {
          map.addSource(POLY_SOURCE, {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          })
          // Colour expression is a `match` on the backend's risk_category string.
          const colorExpr = [
            'match',
            ['get', 'risk_category'],
            'LOW',
            '#22d3ee',
            'MODERATE',
            '#facc15',
            'HIGH',
            '#fb923c',
            'EXTREME',
            '#f43f5e',
            '#64748b',
          ] as never
          // Risk polygons are the backend's raw categorical contours. They stay
          // a low-opacity wash plus a crisp outline, so they read as an overlay
          // annotation rather than competing with the precipitation field.
          map.addLayer(
            {
              id: POLY_FILL,
              type: 'fill',
              source: POLY_SOURCE,
              paint: { 'fill-color': colorExpr, 'fill-opacity': 0.1 },
            },
            firstSymbolLayer,
          )
          map.addLayer(
            {
              id: POLY_LINE,
              type: 'line',
              source: POLY_SOURCE,
              paint: { 'line-color': colorExpr, 'line-width': 0.9, 'line-opacity': 0.55 },
            },
            firstSymbolLayer,
          )
        }

        // 3. Storm-cell markers: a soft glow plus a crisp core.
        if (!map.getSource(CELL_SOURCE)) {
          map.addSource(CELL_SOURCE, {
            type: 'geojson',
            data: { type: 'FeatureCollection', features: [] },
          })
          map.addLayer(
            {
              id: CELL_GLOW,
              type: 'circle',
              source: CELL_SOURCE,
              paint: {
                'circle-radius': ['interpolate', ['linear'], ['get', 'risk_max'], 0, 12, 1, 30] as never,
                'circle-color': [
                  'interpolate', ['linear'], ['get', 'risk_max'],
                  0, 'rgba(34,211,238,0.05)',
                  0.3, 'rgba(74,222,128,0.16)',
                  0.6, 'rgba(249,115,22,0.24)',
                  0.85, 'rgba(244,63,94,0.34)',
                ] as never,
                'circle-blur': 0.85,
              },
            },
            firstSymbolLayer,
          )
          map.addLayer(
            {
              id: CELL_LAYER,
              type: 'circle',
              source: CELL_SOURCE,
              paint: {
                'circle-radius': ['interpolate', ['linear'], ['get', 'risk_max'], 0, 2.5, 1, 7] as never,
                'circle-color': [
                  'interpolate', ['linear'], ['get', 'risk_max'],
                  0, '#22d3ee', 0.45, '#facc15', 0.7, '#f97316', 0.9, '#f43f5e',
                ] as never,
                'circle-opacity': 0.95,
                'circle-stroke-width': 1,
                'circle-stroke-color': 'rgba(255,255,255,0.55)',
              },
            },
            firstSymbolLayer,
          )
        }
        setReady(true)
        setMapStatus('ready')
      }

      // Layers must be re-registered whenever the style is replaced, so keep
      // this handler attached to `styledata` as well as the initial `load`.
      const onStyleReady = () => {
        if (map.isStyleLoaded()) onLoad()
      }
      map.on('load', onLoad)
      map.on('styledata', onStyleReady)

      const onError = (event: { error?: Error }) => {
        const message = event.error?.message ?? 'unknown'
        // Individual tile errors are expected while offline; log but do not
        // destroy an otherwise working map. A failure that stops the style
        // resolving is promoted to a visible state.
        console.warn('[sihps] map error:', message)
        if (!map.isStyleLoaded() && !mapRef.current?.getLayer(PRECIP_SOURCE)) {
          setStyleIssue((current) => current || message)
          setMapStatus('style-unavailable')
        }
      }
      map.on('error', onError)
    }

    // Kick off style resolution; the map is created inside `start`.
    void start()

    return () => {
      cancelled = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      popupRef.current?.remove()
      popupRef.current = null
      // `map` is created inside the async `start`, so tear down via the ref.
      // `remove()` also drops every listener registered above. This is safe
      // when unmounting mid-probe, because the map is simply never created.
      mapRef.current?.remove()
      mapRef.current = null
      setReady(false)
    }
    // Initialise once. Centre/bounds are applied by the effects below.
    // `styleRetry` forces a genuine remount when the basemap is retried.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [styleRetry])

  // The camera is set once at construction from the responsive Uttarakhand
  // view. The backend grid is deliberately *not* used to frame the map: it
  // spans 77.5-80.5E / 29-31.5N, which pulls the view out to show much of the
  // northern subcontinent. The grid still bounds the raster overlay below.
  useEffect(() => {
    if (!ready || !mapRef.current) return
    mapRef.current.easeTo({ center: UTTARAKHAND_CENTRE, zoom: 7.8, duration: 600 })
  }, [ready])

  // Keep the raster's geographic box aligned if the grid changes.
  useEffect(() => {
    const source = mapRef.current?.getSource(PRECIP_SOURCE) as
      | (ImageSource & { setCoordinates?: (c: unknown) => void })
      | undefined
    if (!source?.setCoordinates || !rasterBounds) return
    source.setCoordinates([
      [rasterBounds.minLon, rasterBounds.maxLat],
      [rasterBounds.maxLon, rasterBounds.maxLat],
      [rasterBounds.maxLon, rasterBounds.minLat],
      [rasterBounds.minLon, rasterBounds.minLat],
    ])
  }, [ready, rasterBounds])

  // ------------------------------------------------------------ frame data
  /** Lead times the backend actually offers for this event. */
  const leadTimes: number[] = useMemo(
    () => forecast?.lead_times_h ?? health?.settings.lead_times_h ?? [],
    [forecast, health],
  )

  const loadFrames = useCallback(async () => {
    if (leadTimes.length === 0) return
    setLoadingFrames(true)
    setError(null)
    try {
      // Sequential rather than parallel: each request runs model inference, so
      // firing six at once would stall the worker and the UI together.
      for (let index = 0; index < leadTimes.length; index += 1) {
        const response = await client.getRiskGeoJson({
          lead_hours: leadTimes[index],
          risk_field: riskField,
          min_category: minCategory,
          max_features: 400,
        })
        frameCache.current.set(index, {
          cells: extractStormCells(response.features),
          features: response.features,
          isSynthetic: response.metadata.is_synthetic,
          dataSource: response.metadata.data_source,
          modelVersion: response.metadata.model_version,
          validTime: response.metadata.summary?.init_time ?? '',
        })
      }
    } catch (caught) {
      setError(caught)
    } finally {
      setLoadingFrames(false)
    }
  }, [client, leadTimes, riskField, minCategory])

  useEffect(() => {
    void loadFrames()
  }, [loadFrames])

  /** Current and next frame for a fractional position. */
  const framesAt = useCallback(
    (pos: number) => {
      const maxIndex = Math.max(0, leadTimes.length - 1)
      const clamped = Math.max(0, Math.min(maxIndex, pos))
      const index = Math.floor(clamped)
      const nextIndex = Math.min(maxIndex, index + 1)
      return {
        current: frameCache.current.get(index),
        next: frameCache.current.get(nextIndex),
        index,
        nextIndex,
        t: clamped - index,
        leadHours: leadTimes[clamped] ?? 0,
      }
    },
    [leadTimes],
  )

  // ------------------------------------------------------------- rendering
  /**
   * Paint the interpolated frame.
   *
   * Called from the 60 fps loop, so it performs no React state updates - it only
   * writes to MapLibre sources. `layersRef` keeps the latest toggle values
   * without making this callback change identity on every toggle.
   */
  const paintFrame = useCallback(
    (pos: number) => {
      const map = mapRef.current
      if (!map || !rasterBounds) return
      const { current, next, t, index, nextIndex, leadHours } = framesAt(pos)
      const active = layersRef.current

      // Blend cells so motion between lead times is continuous.
      const cells =
        current && next && nextIndex !== index
          ? blendStormCells(current.cells, next.cells, t)
          : (current?.cells ?? [])

      // 1. Precipitation raster.
      const precipSource = map.getSource(PRECIP_SOURCE) as ImageSource | undefined
      if (precipSource) {
        if (active.precipitation && cells.length > 0) {
          const canvas = rasterisePrecip(cells, rasterBounds, { width: 384, opacity: 1 })
          if (canvas) precipSource.updateImage({ url: canvas.toDataURL('image/png') })
        } else {
          precipSource.updateImage({ url: TRANSPARENT_PIXEL })
        }
      }

      // 2. Storm-cell markers, capped so the layer stays cheap to render.
      const cellSource = map.getSource(CELL_SOURCE) as
        | { setData: (d: unknown) => void }
        | undefined
      if (cellSource) {
        const features = active.stormCells
          ? cells.slice(0, 220).map((cell) => ({
              type: 'Feature' as const,
              geometry: { type: 'Point' as const, coordinates: [cell.lon, cell.lat] },
              properties: {
                risk_max: cell.riskMax,
                risk_mean: cell.riskMean,
                risk_category: cell.riskCategory,
                hazard: cell.hazard,
                lead_hours: leadHours,
              },
            }))
          : []
        cellSource.setData({ type: 'FeatureCollection', features })
      }

      // 3. Raw contours; the flood layer reuses the same geometry.
      const polySource = map.getSource(POLY_SOURCE) as
        | { setData: (d: unknown) => void }
        | undefined
      if (polySource) {
        const showPolys = active.riskPolygons || active.floodRisk
        polySource.setData({
          type: 'FeatureCollection',
          features: showPolys ? (current?.features ?? []) : [],
        })
      }
    },
    [rasterBounds, framesAt],
  )

  // Re-paint on toggle/filter change so the effect is immediate.
  useEffect(() => {
    if (ready) paintFrame(positionRef.current)
  }, [ready, layers, riskField, paintFrame])

  // -------------------------------------------------------------- animation
  useEffect(() => {
    if (!playing || loadingFrames || leadTimes.length < 2) return
    // Reduced motion: no auto-advance. Manual stepping still works.
    if (reducedMotion) return

    let cancelled = false
    lastTickRef.current = null
    const maxIndex = leadTimes.length - 1

    const tick = (now: number) => {
      if (cancelled) return
      const last = lastTickRef.current
      if (last === null) {
        lastTickRef.current = now
      } else if (now - last >= FRAME_DURATION_MS) {
        const nextPos = positionRef.current + (now - last) / FRAME_DURATION_MS
        positionRef.current = nextPos > maxIndex ? 0 : nextPos
        // Commit to React only when the displayed frame index changes.
        setPosition(Math.floor(positionRef.current))
        lastTickRef.current = now
        paintFrame(positionRef.current)
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      cancelled = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastTickRef.current = null
    }
  }, [playing, loadingFrames, leadTimes.length, reducedMotion, paintFrame])

  // ------------------------------------------------------------ interactions
  const scrubTo = useCallback(
    (pos: number) => {
      const clamped = Math.max(0, Math.min(Math.max(0, leadTimes.length - 1), pos))
      positionRef.current = clamped
      setPosition(Math.floor(clamped))
      paintFrame(clamped)
    },
    [leadTimes.length, paintFrame],
  )

  const step = useCallback(
    (delta: number) => {
      setPlaying(false)
      scrubTo(Math.round(positionRef.current) + delta)
    },
    [scrubTo],
  )

  /** Return to the operational Uttarakhand view. */
  const resetView = useCallback(() => {
    const map = mapRef.current
    if (!map) return
    const camera = uttarakhandCamera(map.getContainer().clientWidth || 1200)
    map.flyTo({ center: camera.center, zoom: camera.zoom, duration: 900 })
  }, [])

  const onCellClick = useCallback(
    (event: maplibregl.MapLayerMouseEvent) => {
      const map = mapRef.current
      if (!map) return
      const hits = map.queryRenderedFeatures(event.point, { layers: [CELL_LAYER] })
      popupRef.current?.remove()
      if (hits.length === 0) return

      const properties = hits[0].properties as Record<string, unknown>
      const coordinates = (hits[0].geometry as { coordinates: [number, number] }).coordinates
      const { index, leadHours } = framesAt(positionRef.current)
      const frame = frameCache.current.get(index)

      // Prefer the cached cell (it carries area, cell count and ring); fall
      // back to the rendered feature's own properties if it is not in the cache.
      const cached = frame?.cells.find(
        (c) => Math.abs(c.lon - coordinates[0]) < 1e-6 && Math.abs(c.lat - coordinates[1]) < 1e-6,
      )
      const cell: StormCell = cached ?? {
        id: String(properties.hazard ?? 'cell'),
        lon: coordinates[0],
        lat: coordinates[1],
        hazard: String(properties.hazard ?? 'unknown'),
        eventType: '—',
        riskCategory: String(properties.risk_category ?? 'LOW'),
        riskMax: Number(properties.risk_max ?? 0),
        riskMean: Number(properties.risk_mean ?? 0),
        areaKm2: 0,
        nCells: 0,
        leadHours,
        validTime: frame?.validTime ?? '',
        ring: [],
      }

      popupRef.current = new maplibregl.Popup({ closeButton: true, maxWidth: '290px' })
        .setLngLat([cell.lon, cell.lat])
        .setDOMContent(
          buildCellPopup(cell, frame?.modelVersion ?? forecast?.model_version ?? ''),
        )
        .addTo(map)
    },
    [framesAt, forecast],
  )

  // Attach cell click/hover handlers once the layers exist.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map || !map.getLayer(CELL_LAYER)) return
    const enter = () => {
      map.getCanvas().style.cursor = 'pointer'
    }
    const leave = () => {
      map.getCanvas().style.cursor = ''
    }
    map.on('click', CELL_LAYER, onCellClick)
    map.on('mouseenter', CELL_LAYER, enter)
    map.on('mouseleave', CELL_LAYER, leave)
    return () => {
      map.off('click', CELL_LAYER, onCellClick)
      map.off('mouseenter', CELL_LAYER, enter)
      map.off('mouseleave', CELL_LAYER, leave)
    }
  }, [ready, onCellClick])

  // Layer visibility.
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !map) return
    const setVisible = (id: string, visible: boolean) => {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none')
      }
    }
    setVisible(PRECIP_SOURCE, layers.precipitation)
    setVisible(CELL_LAYER, layers.stormCells)
    setVisible(CELL_GLOW, layers.stormCells)
    setVisible(POLY_FILL, layers.riskPolygons || layers.floodRisk)
    setVisible(POLY_LINE, layers.riskPolygons)
  }, [ready, layers])

  const toggleLayer = useCallback((key: LayerKey) => {
    setLayers((current) => ({ ...current, [key]: !current[key] }))
  }, [])

  /**
   * Re-resolve the basemap. A full remount is the only reliable way to swap a
   * style that failed during construction, so the retry bumps `styleRetry`,
   * which the init effect depends on.
   */
  const retryStyle = useCallback(() => {
    mapRef.current?.remove()
    mapRef.current = null
    setReady(false)
    setStyleIssue('')
    setUsingFallback(false)
    setMapStatus('initialising')
    setStyleRetry((n) => n + 1)
  }, [])

  /**
   * Basemap selection goes through the same preflight/fallback path as the
   * initial load, so switching styles can never produce a blank map either.
   */
  const onStyleChange = useCallback(
    (id: string) => {
      const option = BASEMAPS.find((entry) => entry.id === id)
      const map = mapRef.current
      if (!option || !map) return
      setActiveStyleId(id)
      setMapStatus('initialising')

      const camera = uttarakhandCamera(map.getContainer().clientWidth || 1200)

      void applyBasemapStyle(map, option.style, camera.bounds).then(
        ({ usedFallback, reason }) => {
          if (mapRef.current !== map) return
          setUsingFallback(usedFallback)
          setStyleIssue(usedFallback ? reason ?? '' : '')
          // Custom sources and layers are re-registered by the `styledata`
          // handler, so readiness is only claimed once that has happened.
          setMapStatus('ready')
        },
      )
    },
    [],
  )

  // Derived view state. `loadingFrames` is a deliberate dependency: the frame
  // cache is a ref (so the rAF loop can read it without re-rendering), and this
  // is where those cached frames are folded into renderable state.
  const timelineFrames: TimelineFrame[] = useMemo(
    () =>
      leadTimes.map((leadHours, index) => ({
        index,
        leadHours,
        validTime: frameCache.current.get(index)?.validTime ?? '',
        cellCount: frameCache.current.get(index)?.cells.length ?? 0,
        loaded: frameCache.current.has(index),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [leadTimes, loadingFrames],
  )

  const activeFrame = frameCache.current.get(Math.floor(position))
  const displayedLead = leadTimes[Math.min(leadTimes.length - 1, Math.floor(position))] ?? 0
  const isSynthetic = activeFrame?.isSynthetic ?? forecast?.is_synthetic ?? true

  const maxRisk = useMemo(
    () => activeFrame?.cells.reduce((peak, cell) => Math.max(peak, cell.riskMax), 0) ?? 0,
    [activeFrame],
  )
  const activeCellCount = activeFrame?.cells.length ?? 0

  // Make sure a forecast exists so lead times are known.
  useEffect(() => {
    if (!forecast) void runForecast()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /**
   * Every non-healthy map condition, most severe first. The map is never
   * reported as operational while any of these is present.
   */
  const notices: MapNotice[] = useMemo(() => {
    const list: MapNotice[] = []

    if (mapStatus === 'initialising') {
      list.push({
        tone: 'info',
        title: 'Loading map',
        detail: `Loading the ${activeBasemap?.label ?? 'basemap'} tiles over Uttarakhand.`,
      })
    }

    if (mapStatus === 'style-unavailable') {
      list.push({
        tone: 'error',
        title: 'Basemap unavailable',
        detail:
          `The basemap tiles could not be loaded${styleIssue ? `: ${styleIssue}` : ''}. ` +
          'Overlays and the AOI frame are still shown, but no roads, terrain or place labels ' +
          'are available. Check the network connection, or set VITE_SATELLITE_TILE_URL to a ' +
          'reachable raster tile service.',
        action: 'Retry',
        onAction: retryStyle,
      })
    }

    // The fallback reaches `ready`, so without this it would be indistinguishable
    // from a normal street basemap. It is a working coordinate reference frame,
    // not cartography, and the user must be told which one they are looking at.
    if (usingFallback && mapStatus !== 'style-unavailable') {
      list.push({
        tone: 'warn',
        title: 'Offline reference basemap',
        detail:
          `The configured basemap service is unreachable${styleIssue ? ` (${styleIssue})` : ''}. ` +
          'Showing a coordinate grid and the model AOI only — no roads, terrain, rivers or ' +
          'place labels. Risk layers are unaffected. Retry once the network is available.',
        action: 'Retry',
        onAction: retryStyle,
      })
    }

    if (connection === 'offline') {
      list.push({
        tone: 'error',
        title: 'Backend unavailable',
        detail:
          'No response from the API, so grid bounds and risk data cannot be loaded. ' +
          'Start it with "uvicorn app.main:app --port 8000".',
        action: 'Refresh',
        onAction: () => void runForecast(),
      })
    }

    // Empty-data is reported only once the backend answered and there is
    // genuinely nothing to draw; it must not mask a broken basemap.
    const framesLoaded = timelineFrames.some((f) => f.loaded)
    if (
      mapStatus === 'ready' &&
      !loadingFrames &&
      connection !== 'offline' &&
      !error &&
      framesLoaded &&
      activeCellCount === 0
    ) {
      list.push({
        tone: 'warn',
        title: 'No risk data available',
        detail:
          'The backend returned an empty feature collection for this hazard field and ' +
          'risk-category filter. Lower the minimum category or choose a different field.',
      })
    }

    if (error && connection !== 'offline') {
      list.push({
        tone: 'error',
        title: 'Risk data request failed',
        detail: describeError(error),
        action: 'Retry',
        onAction: () => void loadFrames(),
      })
    }

    return list
  }, [
    mapStatus,
    activeBasemap,
    styleIssue,
    usingFallback,
    connection,
    loadingFrames,
    error,
    timelineFrames,
    activeCellCount,
    retryStyle,
    runForecast,
    loadFrames,
  ])

  return (
    // `flex-1 min-h-0` lets the map fill the layout's content panel without
    // viewport arithmetic, so it stays correct at any window size.
    <div className="relative min-h-[30rem] w-full flex-1 overflow-hidden bg-[#050a14]">
      {/* MapLibre adds its own `.maplibregl-map` class to this element, and
          its stylesheet forces `position: relative`, which would defeat an
          `absolute inset-0` utility. `h-full w-full` is set explicitly so the
          canvas is sized regardless of which rule wins. */}
      <div ref={mapContainer} className="h-full w-full" aria-label="Radar weather map" />

      {/* Explicit map states. Without this a failed style renders as an empty
          dark canvas that looks like a healthy, data-less map. */}
      <MapNoticeBanner notices={notices} onRetry={retryStyle} />

      {/* Dim scrim so basemap labels stay readable under the floating UI. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-gradient-to-r
                   from-slate-950/55 via-transparent to-slate-950/35"
      />

      {/* Left: hazard controls. Capped and scrollable so it can never run under
          the precipitation legend that is anchored bottom-left. */}
      <div
        className="pointer-events-none absolute left-3 top-3 z-10
                   max-h-[calc(100%-1.5rem)] overflow-y-auto overscroll-contain
                   lg:left-4 lg:top-4 lg:max-h-[min(calc(100%-13.5rem),40rem)]"
      >
        <HazardPanel
          layers={layers}
          onToggleLayer={toggleLayer}
          riskField={riskField}
          onRiskFieldChange={setRiskField}
          minCategory={minCategory}
          onMinCategoryChange={setMinCategory}
          styleOptions={STYLE_OPTIONS}
          activeStyleId={activeStyleId}
          onStyleChange={onStyleChange}
          dataSource={activeFrame?.dataSource ?? forecast?.data_source ?? ''}
          modelVersion={activeFrame?.modelVersion ?? forecast?.model_version ?? ''}
          isSynthetic={isSynthetic}
          validTime={activeFrame?.validTime ?? ''}
          leadLabel={`T+${formatLeadHours(displayedLead)}`}
          collapsed={panelCollapsed}
          onToggleCollapsed={() => setPanelCollapsed((value) => !value)}
        />
      </div>

      {/* Right side map tools. */}
      <div className="pointer-events-none absolute right-3 top-3 z-10 flex flex-col items-end gap-2 lg:right-4 lg:top-4">
        <MapTools
          buttons={[
            {
              label: 'Zoom in',
              glyph: '+',
              onClick: () => {
                void mapRef.current?.zoomIn({ duration: 300 })
              },
            },
            {
              label: 'Zoom out',
              glyph: '-',
              onClick: () => {
                void mapRef.current?.zoomOut({ duration: 300 })
              },
            },
            { label: 'Reset to AOI', glyph: 'O', onClick: resetView },
          ]}
        />

        {/* Live status readout, so the map is never a bare canvas. */}
        <div
          className="pointer-events-auto w-44 rounded-xl border border-white/12 bg-slate-950/90
                     px-3 py-2 text-[10px] shadow-2xl backdrop-blur-xl"
          role="status"
          aria-live="polite"
        >
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
            Live readout
          </p>
          <p className="flex justify-between">
            <span className="text-slate-500">Cells</span>
            <span className="font-mono text-slate-200">{activeCellCount}</span>
          </p>
          <p className="flex justify-between">
            <span className="text-slate-500">Peak risk</span>
            <span className="font-mono text-slate-200">{formatProbability(maxRisk)}</span>
          </p>
          <p className="flex justify-between">
            <span className="text-slate-500">Frames</span>
            <span className="font-mono text-slate-200">
              {timelineFrames.filter((f) => f.loaded).length}/{timelineFrames.length}
            </span>
          </p>
        </div>
      </div>

      {/* Bottom-left: legend. */}
      <div className="pointer-events-none absolute bottom-3 left-3 z-10 lg:bottom-4 lg:left-4">
        <PrecipLegend visible={layers.precipitation} />
      </div>

      {/* Bottom: timeline, centred so it never covers the legend. */}
      <div className="pointer-events-none absolute inset-x-3 bottom-3 z-10 flex justify-center px-0 lg:inset-x-auto lg:bottom-4 lg:left-1/2 lg:w-[min(58rem,calc(100%-26rem))] lg:-translate-x-1/2">
        <RadarTimeline
          frames={timelineFrames}
          position={position}
          playing={playing}
          reducedMotion={reducedMotion}
          onScrub={scrubTo}
          onTogglePlay={() => {
            // Restart from the first frame if playback had run to the end.
            if (!playing && position >= timelineFrames.length - 1.01) scrubTo(0)
            setPlaying((value) => !value)
          }}
          onStep={step}
          leadHours={displayedLead}
          validTime={activeFrame?.validTime ?? ''}
          isSynthetic={isSynthetic}
        />
      </div>

      {/* Error and loading overlays. `error` is `unknown`, so render via a boolean. */}
      {error !== null && error !== undefined && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/70 p-6">
          <div className="max-w-md rounded-xl border border-rose-400/40 bg-slate-900/90 p-4 text-center">
            <p className="text-sm font-semibold text-rose-300">Radar data unavailable</p>
            <p className="mt-1.5 text-xs text-slate-400">
              {error instanceof Error
                ? error.message
                : typeof error === 'string'
                  ? error
                  : 'The backend request failed.'}
            </p>
            <button
              type="button"
              className="mt-3 rounded-lg border border-white/15 px-3 py-1.5 text-xs text-slate-200
                         transition hover:bg-white/10"
              onClick={() => void loadFrames()}
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {loadingFrames && !error && (
        <div className="pointer-events-none absolute inset-x-0 top-1/2 z-20 -translate-y-1/2">
          <div className="mx-auto w-max rounded-full border border-white/12 bg-slate-950/85 px-4 py-2 text-[11px] text-sky-300 shadow-xl">
            Loading forecast frames…
          </div>
        </div>
      )}

      {/* Prominent experimental-data disclaimer. */}
      <p
        className="pointer-events-none absolute bottom-1 left-1/2 z-10 -translate-x-1/2
                   whitespace-nowrap px-2 text-[9px] font-semibold tracking-wide text-amber-300/85"
      >
        Experimental AI prediction. Not an official IMD warning. Refer to IMD for authoritative alerts.
      </p>
    </div>
  )
}
