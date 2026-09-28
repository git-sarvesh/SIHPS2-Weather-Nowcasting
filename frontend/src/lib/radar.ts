/**
 * Radar rendering core.
 *
 * Everything here is **derived from backend values only**. Given the polygons
 * `POST /risk/geojson` returns, it produces:
 *   - a smooth rasterised precipitation field (no rectangular grid artefacts),
 *   - a meteorological colour ramp,
 *   - storm-cell centroids for markers and popups.
 *
 * Honesty constraints, enforced deliberately:
 *  - The backend exposes **risk probability**, not rainfall rate in mm/h. The
 *    legend labels the scale in probability terms, and any mm/h figure is an
 *    explicitly-labelled illustrative equivalence, never a measurement claim.
 *  - There is **no wind field and no lightning data** in the API, so neither is
 *    synthesised here. Animation is strictly interpolation between real
 *    forecast lead times.
 */

/** One stop of the precipitation colour ramp. */
export interface RampStop {
  /** Lower bound of the stop, in probability units (0-1). */
  from: number
  /** [r, g, b, a] with a in 0-255. */
  rgba: [number, number, number, number]
  label: string
}

/**
 * Meteorological ramp: blue -> cyan/green -> yellow/orange -> red/magenta.
 *
 * Alpha rises with intensity so light precipitation reads as a wash and extreme
 * cells read as solid, which is what keeps the overlay legible over a basemap.
 */
export const PRECIP_RAMP: RampStop[] = [
  { from: 0.0, rgba: [56, 189, 248, 0], label: 'none' },
  { from: 0.05, rgba: [56, 189, 248, 40], label: 'very light' },
  { from: 0.15, rgba: [34, 211, 238, 90], label: 'light' },
  { from: 0.3, rgba: [74, 222, 128, 130], label: 'moderate' },
  { from: 0.45, rgba: [250, 204, 21, 160], label: 'heavy' },
  { from: 0.6, rgba: [249, 115, 22, 195], label: 'very heavy' },
  { from: 0.75, rgba: [244, 63, 94, 220], label: 'extreme' },
  { from: 0.88, rgba: [217, 70, 239, 240], label: 'violent' },
]

/** Colour for a probability value, following {@link PRECIP_RAMP}. */
export function rampColor(value: number): [number, number, number, number] {
  const v = Math.max(0, Math.min(1, value))
  let chosen = PRECIP_RAMP[0]
  for (const stop of PRECIP_RAMP) {
    if (v >= stop.from) chosen = stop
    else break
  }
  return chosen.rgba
}

/** CSS colour string, used for DOM-side legends and markers. */
export function rampCss(value: number, alphaScale = 1): string {
  const [r, g, b, a] = rampColor(value)
  return `rgba(${r}, ${g}, ${b}, ${(a / 255) * alphaScale})`
}

/**
 * Illustrative mm/h equivalent for a probability, shown in the legend.
 *
 * Display convention only: the backend does not return a rainfall rate, and this
 * is not a measurement. The legend labels it as illustrative.
 */
export function illustrativeMmPerHour(value: number): number {
  return Math.round(Math.pow(Math.max(0, Math.min(1, value)), 2.1) * 80 * 10) / 10
}

/** A resolved storm cell derived from one backend polygon. */
export interface StormCell {
  id: string
  lon: number
  lat: number
  hazard: string
  eventType: string
  riskCategory: string
  riskMax: number
  riskMean: number
  areaKm2: number
  nCells: number
  leadHours: number
  validTime: string
  /** Polygon outline in [lon, lat] pairs, for the focused-cell highlight. */
  ring: [number, number][]
}

/** Longitude/latitude bounds of a feature's outer ring. */
function boundsOfRing(ring: [number, number][]): { min: [number, number]; max: [number, number] } {
  let minLon = Infinity
  let minLat = Infinity
  let maxLon = -Infinity
  let maxLat = -Infinity
  for (const [lon, lat] of ring) {
    if (lon < minLon) minLon = lon
    if (lon > maxLon) maxLon = lon
    if (lat < minLat) minLat = lat
    if (lat > maxLat) maxLat = lat
  }
  return { min: [minLon, minLat], max: [maxLon, maxLat] }
}

/**
 * Convert a GeoJSON response into storm cells.
 *
 * Cells are built from the polygons the backend actually produced, so their
 * positions and intensities are the model's own output - nothing here invents a
 * location or a strength.
 */
export function extractStormCells(features: readonly unknown[]): StormCell[] {
  const cells: StormCell[] = []
  for (const raw of features) {
    const feature = raw as {
      geometry?: { type?: string; coordinates?: unknown }
      properties?: Record<string, unknown>
    }
    if (feature.geometry?.type !== 'Polygon') continue
    const rings = feature.geometry.coordinates as [number, number][][]
    const ring = rings?.[0]
    if (!Array.isArray(ring) || ring.length < 3) continue

    const properties = feature.properties ?? {}
    const { min, max } = boundsOfRing(ring)
    const leadHours = Array.isArray(properties.lead_hours)
      ? Number(properties.lead_hours[0] ?? 0)
      : 0

    cells.push({
      id: `${String(properties.hazard)}-${min[0].toFixed(3)}-${min[1].toFixed(3)}`,
      lon: (min[0] + max[0]) / 2,
      lat: (min[1] + max[1]) / 2,
      hazard: String(properties.hazard ?? 'unknown'),
      eventType: String(properties.event_type ?? 'unknown'),
      riskCategory: String(properties.risk_category ?? 'LOW'),
      riskMax: Number(properties.risk_max ?? 0),
      riskMean: Number(properties.risk_mean ?? 0),
      areaKm2: Number(properties.area_km2 ?? 0),
      nCells: Number(properties.n_cells ?? 0),
      leadHours,
      validTime: String(properties.valid_time ?? ''),
      ring: ring.map(([lon, lat]) => [lon, lat] as [number, number]),
    })
  }
  // Strongest first, so markers and popups lead with the worst cell.
  return cells.sort((a, b) => b.riskMax - a.riskMax)
}

/** Geospatial bounds of a rasterised frame. */
export interface RasterBounds {
  minLon: number
  minLat: number
  maxLon: number
  maxLat: number
}

export interface RasterOptions {
  /** Raster width in pixels. Height follows the AOI aspect ratio. */
  width?: number
  /** Base opacity multiplier, so layers dim without re-rasterising. */
  opacity?: number
}

/**
 * Rasterise the forecast risk field into a smooth, geographically aligned
 * precipitation raster.
 *
 * ## Why the previous implementation was wrong
 *
 * It projected each polygon's **centroid** and painted a circular
 * radial-gradient blob per cell. That is a decorative heatmap, not a field: it
 * ignored the polygon geometry entirely, so the output was a cloud of unrelated
 * circles bearing no relationship to the model's grid. Over a real basemap it
 * read as "synthetic" immediately.
 *
 * ## What this does instead
 *
 * The backend's GeoJSON features are merged, axis-aligned rectangles that tile a
 * real model grid (`nx` x `ny` at `res_km`, exposed via `/health`). The actual
 * field is therefore recoverable:
 *
 *  1. Each polygon is scan-converted into a scalar field using its true
 *     footprint rather than a single point.
 *  2. A short separable blur removes the stair-steps introduced by rectangle
 *     merging. This corrects a known artefact of the vectorisation step; it does
 *     not add meteorological content.
 *  3. The field is colour-mapped through {@link PRECIP_RAMP}, with alpha rising
 *     from the intensity, and left fully transparent where the field is zero.
 *
 * The result is spatially continuous, aligned to the real grid bounds, and has
 * no straight-line cell edges.
 *
 * ## Honesty
 *
 * The field encodes the backend's per-polygon risk probability. It is *not* a
 * radar reflectivity measurement and carries no rainfall-rate claim; the legend
 * labels the scale in probability terms and marks any mm/h figure as an
 * illustrative equivalence.
 */
export function rasterisePrecip(
  cells: readonly StormCell[],
  bounds: RasterBounds,
  options: RasterOptions = {},
): HTMLCanvasElement | null {
  const width = options.width ?? 320
  const opacity = options.opacity ?? 1
  const spanLon = bounds.maxLon - bounds.minLon
  const spanLat = bounds.maxLat - bounds.minLat
  if (spanLon <= 0 || spanLat <= 0) return null

  // Correct for Mercator: a degree of latitude is narrower on the ground than a
  // degree of longitude at these latitudes, so an uncorrected raster would
  // stretch the field east-west.
  const midLat = (bounds.minLat + bounds.maxLat) / 2
  const mercatorScale = 1 / Math.max(0.05, Math.cos((midLat * Math.PI) / 180))
  const height = Math.max(
    64,
    Math.min(720, Math.round(width * (spanLat / spanLon) * mercatorScale)),
  )

  const fw = width
  const fh = height
  const field = new Float32Array(fw * fh)

  const project = (lon: number, lat: number): [number, number] => [
    ((lon - bounds.minLon) / spanLon) * fw,
    ((bounds.maxLat - lat) / spanLat) * fh,
  ]

  let painted = 0
  for (const cell of cells) {
    if (cell.riskMax <= 0.02) continue
    if (cell.ring.length >= 3) {
      painted += fillPolygon(field, fw, fh, cell.ring, project, cell.riskMax)
    } else {
      // Degenerate geometry: fall back to a small disc at the centroid rather
      // than dropping the cell entirely.
      const [cx, cy] = project(cell.lon, cell.lat)
      const r = Math.max(2, Math.min(fw, fh) * 0.02)
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(fh - 1, Math.ceil(cy + r)); y += 1) {
        for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(fw - 1, Math.ceil(cx + r)); x += 1) {
          if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) > r) continue
          const i = y * fw + x
          if (cell.riskMax > field[i]) field[i] = cell.riskMax
        }
      }
      painted += 1
    }
  }
  if (painted === 0) return null

  smoothField(field, fw, fh, 2)

  // Colour-map. Alpha is driven by intensity so the basemap stays visible under
  // light precipitation, and zero-valued pixels stay fully transparent.
  const image = new ImageData(fw, fh)
  const data = image.data
  for (let i = 0; i < field.length; i += 1) {
    const v = field[i]
    if (v <= 0.02) continue
    const [r, g, b, a] = rampColor(v)
    const alpha = a * Math.min(1, opacity * Math.max(0.25, v * 1.35))
    if (alpha <= 1) continue
    const p = i * 4
    data[p] = r
    data[p + 1] = g
    data[p + 2] = b
    data[p + 3] = Math.min(255, Math.round(alpha))
  }

  const canvas = document.createElement('canvas')
  canvas.width = fw
  canvas.height = fh
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.putImageData(image, 0, 0)
  return canvas
}

/**
 * Scan-convert a ring into `field`, keeping the maximum at each cell.
 *
 * Even-odd scanline fill over the projected ring. Returns the number of cells
 * written, so the caller can distinguish an empty result from a rendered one.
 */
function fillPolygon(
  field: Float32Array,
  width: number,
  height: number,
  ring: readonly [number, number][],
  project: (lon: number, lat: number) => [number, number],
  value: number,
): number {
  const pts: [number, number][] = []
  for (const [lon, lat] of ring) {
    const [x, y] = project(lon, lat)
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y])
  }
  if (pts.length < 3) return 0

  let minY = Infinity
  let maxY = -Infinity
  for (const [, y] of pts) {
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  const y0 = Math.max(0, Math.floor(minY))
  const y1 = Math.min(height - 1, Math.ceil(maxY))
  let written = 0

  for (let y = y0; y <= y1; y += 1) {
    const sampleY = y + 0.5
    // X-intersections of the ring edges with this scanline.
    const xs: number[] = []
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i, i += 1) {
      const [xi, yi] = pts[i]
      const [xj, yj] = pts[j]
      if (yi === yj) continue
      if (sampleY >= Math.min(yi, yj) && sampleY < Math.max(yi, yj)) {
        xs.push(xi + ((sampleY - yi) / (yj - yi)) * (xj - xi))
      }
    }
    if (xs.length < 2) continue
    xs.sort((a, b) => a - b)
    // Fill between successive crossing pairs (even-odd rule).
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xa = Math.max(0, Math.ceil(xs[k] - 0.5))
      const xb = Math.min(width - 1, Math.floor(xs[k + 1] - 0.5))
      for (let x = xa; x <= xb; x += 1) {
        const i = y * width + x
        if (value > field[i]) {
          field[i] = value
          written += 1
        }
      }
    }
  }
  return written
}

/**
 * Separable 1-2-1 blur applied `passes` times, in place.
 *
 * A narrow, fixed kernel is deliberate: it removes the stair-step seams left by
 * the backend's rectangle merging without inventing structure. Wider smoothing
 * would smear real gradients and overstate the field's smoothness.
 */
function smoothField(field: Float32Array, width: number, height: number, passes: number): void {
  const tmp = new Float32Array(field.length)
  for (let pass = 0; pass < passes; pass += 1) {
    for (let y = 0; y < height; y += 1) {
      const row = y * width
      for (let x = 0; x < width; x += 1) {
        const l = field[row + Math.max(0, x - 1)]
        const c = field[row + x]
        const r = field[row + Math.min(width - 1, x + 1)]
        tmp[row + x] = (l + 2 * c + r) / 4
      }
    }
    for (let x = 0; x < width; x += 1) {
      for (let y = 0; y < height; y += 1) {
        const u = tmp[Math.max(0, y - 1) * width + x]
        const c = tmp[y * width + x]
        const d = tmp[Math.min(height - 1, y + 1) * width + x]
        field[y * width + x] = (u + 2 * c + d) / 4
      }
    }
  }
}

/** Linear interpolation used to cross-fade between two forecast lead times. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Blend two storm-cell lists by matching on cell id.
 *
 * The backend recomputes polygons per lead time, so ids occasionally appear or
 * disappear. Blending runs over the union of ids, so a cell present in only one
 * frame fades rather than popping.
 */
export function blendStormCells(
  from: readonly StormCell[],
  to: readonly StormCell[],
  t: number,
): StormCell[] {
  const fromById = new Map(from.map((c) => [c.id, c]))
  const toById = new Map(to.map((c) => [c.id, c]))
  const blended: StormCell[] = []

  for (const id of new Set([...fromById.keys(), ...toById.keys()])) {
    const a = fromById.get(id)
    const b = toById.get(id)
    if (a && b) {
      blended.push({
        ...b,
        lon: lerp(a.lon, b.lon, t),
        lat: lerp(a.lat, b.lat, t),
        riskMax: lerp(a.riskMax, b.riskMax, t),
        riskMean: lerp(a.riskMean, b.riskMean, t),
        areaKm2: lerp(a.areaKm2, b.areaKm2, t),
      })
    } else if (a) {
      // Fading out: hold position, collapse intensity to nothing.
      blended.push({ ...a, riskMax: a.riskMax * (1 - t), riskMean: a.riskMean * (1 - t) })
    } else if (b) {
      // Fading in: grow from nothing at the destination.
      blended.push({ ...b, riskMax: b.riskMax * t, riskMean: b.riskMean * t })
    }
  }
  return blended.sort((x, y) => y.riskMax - x.riskMax)
}
