/** Tests for the radar derivation helpers.
 *
 * These decide what the map claims: colour mapping, cell extraction from backend
 * polygons, frame blending and rasterisation. They are pure functions, so they
 * are tested directly rather than through the map.
 */

import { beforeEach, describe, expect, it } from 'vitest'

import { getCanvasPixels, resetCanvasPaintCount } from './setup'
import {
  PRECIP_RAMP,
  blendStormCells,
  extractStormCells,
  illustrativeMmPerHour,
  lerp,
  rampColor,
  rampCss,
  rasterisePrecip,
  type StormCell,
} from '../lib/radar'

/** Build a minimal backend-shaped polygon feature. */
function polygon(
  lon: number,
  lat: number,
  riskMax: number,
  overrides: Record<string, unknown> = {},
) {
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [lon, lat],
          [lon + 0.2, lat],
          [lon + 0.2, lat + 0.2],
          [lon, lat + 0.2],
          [lon, lat],
        ],
      ],
    },
    properties: {
      hazard: 'cloudburst',
      event_type: 'compound',
      risk_category: 'MODERATE',
      risk_category_code: 1,
      risk_max: riskMax,
      risk_mean: riskMax * 0.8,
      area_km2: 120,
      n_cells: 30,
      lead_hours: [1.0],
      valid_time: '2024-08-02T14:00:00+00:00',
      model_version: 'sihps-convlstm-cha-v0.1.0',
      is_synthetic: true,
      ...overrides,
    },
  }
}

describe('rampColor', () => {
  it('is fully transparent below the first stop', () => {
    expect(rampColor(0)[3]).toBe(0)
  })

  it('reaches the hottest ramp colour at the top', () => {
    expect(rampColor(1)).toEqual(PRECIP_RAMP[PRECIP_RAMP.length - 1].rgba)
  })

  it('clamps out-of-range input instead of throwing', () => {
    expect(rampColor(-5)).toEqual(rampColor(0))
    expect(rampColor(99)).toEqual(rampColor(1))
  })

  it('moves from cool to warm across the scale', () => {
    // Light precipitation is blue-dominant; heavy is red-dominant. The very top
    // stop is violet, so it is compared against the orange band, not the top.
    const light = rampColor(0.1)
    const heavy = rampColor(0.65)
    expect(light[2]).toBeGreaterThan(light[0]) // blue dominant
    expect(heavy[0]).toBeGreaterThan(heavy[2]) // red dominant
  })

  it('increases alpha monotonically with intensity', () => {
    let previous = -1
    for (const value of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const alpha = rampColor(value)[3]
      expect(alpha).toBeGreaterThan(previous)
      previous = alpha
    }
  })

  it('renders a CSS colour with a scaled alpha', () => {
    expect(rampCss(0.5)).toMatch(/^rgba\(\d+, \d+, \d+, [\d.]+\)$/)
  })
})

describe('illustrativeMmPerHour', () => {
  it('is monotonic and bounded', () => {
    expect(illustrativeMmPerHour(0)).toBe(0)
    const a = illustrativeMmPerHour(0.5)
    const b = illustrativeMmPerHour(0.9)
    expect(a).toBeLessThan(b)
    expect(b).toBeLessThanOrEqual(80)
  })
})

describe('extractStormCells', () => {
  it('builds a cell from a backend polygon', () => {
    const cells = extractStormCells([polygon(78, 30, 0.6)])
    expect(cells).toHaveLength(1)
    expect(cells[0].riskMax).toBeCloseTo(0.6)
    expect(cells[0].hazard).toBe('cloudburst')
    expect(cells[0].leadHours).toBe(1)
    // Centroid of the 0.2-degree box.
    expect(cells[0].lon).toBeCloseTo(78.1)
    expect(cells[0].lat).toBeCloseTo(30.1)
  })

  it('sorts strongest first', () => {
    const cells = extractStormCells([
      polygon(78, 30, 0.2),
      polygon(79, 31, 0.9),
      polygon(77, 29, 0.5),
    ])
    expect(cells.map((c) => c.riskMax)).toEqual([0.9, 0.5, 0.2])
  })

  it('ignores non-polygon geometries', () => {
    const point = {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [78, 30] },
      properties: {},
    }
    expect(extractStormCells([point])).toHaveLength(0)
  })

  it('ignores degenerate rings', () => {
    const sliver = {
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: [[[78, 30], [78, 30]]] },
      properties: {},
    }
    expect(extractStormCells([sliver])).toHaveLength(0)
  })

  it('tolerates missing properties without throwing', () => {
    const bare = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [78, 30],
            [78.2, 30],
            [78.2, 30.2],
            [78, 30.2],
          ],
        ],
      },
      properties: {},
    }
    const cells = extractStormCells([bare])
    expect(cells[0].hazard).toBe('unknown')
    expect(cells[0].riskMax).toBe(0)
  })
})

describe('lerp', () => {
  it('interpolates linearly and clamps naturally at the ends', () => {
    expect(lerp(0, 10, 0)).toBe(0)
    expect(lerp(0, 10, 0.5)).toBe(5)
    expect(lerp(0, 10, 1)).toBe(10)
  })
})

describe('blendStormCells', () => {
  const cell = (id: string, lon: number, risk: number): StormCell => ({
    id,
    lon,
    lat: 30,
    hazard: 'cloudburst',
    eventType: 'compound',
    riskCategory: 'MODERATE',
    riskMax: risk,
    riskMean: risk * 0.8,
    areaKm2: 100,
    nCells: 10,
    leadHours: 1,
    validTime: 't',
    ring: [],
  })

  it('interpolates position and intensity for a shared id', () => {
    const blended = blendStormCells([cell('a', 78, 0.2)], [cell('a', 79, 0.8)], 0.5)
    expect(blended).toHaveLength(1)
    expect(blended[0].lon).toBeCloseTo(78.5)
    expect(blended[0].riskMax).toBeCloseTo(0.5)
  })

  it('fades a cell that only exists in the outgoing frame', () => {
    const blended = blendStormCells([cell('a', 78, 1)], [], 0.5)
    expect(blended[0].riskMax).toBeCloseTo(0.5)
  })

  it('grows a cell that only exists in the incoming frame', () => {
    const blended = blendStormCells([], [cell('b', 79, 1)], 0.25)
    expect(blended[0].riskMax).toBeCloseTo(0.25)
  })

  it('is stable at t=0 and t=1', () => {
    const from = [cell('a', 78, 0.3)]
    const to = [cell('a', 79, 0.7)]
    expect(blendStormCells(from, to, 0)[0].lon).toBeCloseTo(78)
    expect(blendStormCells(from, to, 1)[0].lon).toBeCloseTo(79)
  })

  it('keeps the strongest cell first', () => {
    const blended = blendStormCells(
      [cell('a', 78, 0.1), cell('b', 79, 0.2)],
      [cell('a', 78, 0.9), cell('b', 79, 0.8)],
      0.5,
    )
    expect(blended[0].riskMax).toBeGreaterThanOrEqual(blended[1].riskMax)
  })
})

describe('rasterisePrecip', () => {
  const bounds = { minLon: 77, minLat: 29, maxLon: 81, maxLat: 32 }

  beforeEach(() => {
    resetCanvasPaintCount()
  })

  /** Count pixels that are not fully transparent. */
  function opaqueCount(canvas: HTMLCanvasElement | null): number {
    const pixels = getCanvasPixels(canvas)
    if (!pixels) return 0
    let count = 0
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) count += 1
    }
    return count
  }

  it('returns null for a degenerate bounding box', () => {
    const same = { minLon: 1, minLat: 1, maxLon: 1, maxLat: 1 }
    expect(rasterisePrecip([], same)).toBeNull()
  })

  it('returns null when there is no visible risk at all', () => {
    const weak = extractStormCells([polygon(78, 30, 0.001)])
    expect(rasterisePrecip(weak, bounds)).toBeNull()
  })

  it('produces a canvas sized to the AOI aspect ratio', () => {
    const cells = extractStormCells([polygon(78, 30, 0.8)])
    const canvas = rasterisePrecip(cells, bounds, { width: 200 })
    expect(canvas).not.toBeNull()
    expect(canvas?.width).toBe(200)
    // 3 deg lat / 4 deg lon, Mercator-corrected near 30.5 deg N.
    expect(canvas?.height).toBeGreaterThan(100)
    expect(canvas?.height).toBeLessThan(400)
  })

  it('paints the polygon footprint rather than a blob at the centroid', () => {
    // The old implementation drew one radial gradient per cell. The new one
    // scan-converts the ring, so painted pixels must cover a contiguous area
    // proportional to the polygon's real size.
    const cells = extractStormCells([polygon(78, 30, 0.9)])
    const canvas = rasterisePrecip(cells, bounds, { width: 240 })
    const painted = opaqueCount(canvas)
    // A 0.2 deg box out of 4 deg across, at 240 px wide, is ~12 px on a side.
    // Allow generous slack for the blur, but require a real area.
    expect(painted).toBeGreaterThan(40)
    expect(painted).toBeLessThan(240 * 240 * 0.25)
  })

  it('leaves the rest of the raster transparent', () => {
    // A small polygon must not produce an opaque rectangle over the whole AOI.
    const cells = extractStormCells([polygon(78, 30, 0.9)])
    const canvas = rasterisePrecip(cells, bounds, { width: 240 })
    const total = 240 * (canvas?.height ?? 0)
    expect(opaqueCount(canvas)).toBeLessThan(total * 0.2)
  })

  it('merges overlapping cells into one continuous field', () => {
    const cells = extractStormCells([
      polygon(78, 30, 0.9),
      polygon(78.05, 30.05, 0.8),
    ])
    const canvas = rasterisePrecip(cells, bounds, { width: 240 })
    // Both are drawn, so the union is larger than either alone but still a
    // single field - no separate disc per cell.
    expect(opaqueCount(canvas)).toBeGreaterThan(40)
  })

  it('respects the opacity multiplier', () => {
    const cells = extractStormCells([polygon(78, 30, 0.9)])
    const solid = rasterisePrecip(cells, bounds, { width: 240 })
    const faded = rasterisePrecip(cells, bounds, { width: 240, opacity: 0.4 })
    const a = getCanvasPixels(solid)
    const b = getCanvasPixels(faded)
    expect(a).not.toBeNull()
    expect(b).not.toBeNull()
    // Alpha is written at pixel index 3, 7, 11 ... Compare the first lit pixel.
    let checked = false
    for (let i = 3; i < (a?.length ?? 0); i += 4) {
      if ((a?.[i] ?? 0) > 40) {
        expect(b?.[i] ?? 0).toBeLessThan(a?.[i] ?? 0)
        checked = true
        break
      }
    }
    expect(checked).toBe(true)
  })

  it('falls back to a centroid disc for a degenerate ring', () => {
    // A cell with no usable ring must still be drawn, not silently dropped.
    const cell = extractStormCells([polygon(78, 30, 0.9)])[0]
    const ringless = { ...cell, ring: [] as [number, number][] }
    const canvas = rasterisePrecip([ringless], bounds, { width: 240 })
    expect(canvas).not.toBeNull()
    expect(opaqueCount(canvas)).toBeGreaterThan(0)
  })
})
