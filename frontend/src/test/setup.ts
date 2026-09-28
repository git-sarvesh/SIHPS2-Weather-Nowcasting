/**
 * Vitest setup.
 *
 * jsdom lacks a few browser APIs the dashboard touches, and MapLibre needs a
 * canvas that does not exist here. The map page is therefore not unit-tested in
 * jsdom; the API client and the state/UI primitives are.
 */

import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// Recharts measures its container; jsdom returns zero, so give it a size.
Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
  configurable: true,
  value: 800,
})
Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
  configurable: true,
  value: 400,
})
Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
  configurable: true,
  value: 800,
})
Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
  configurable: true,
  value: 400,
})

// jsdom does not implement ResizeObserver, which MapLibre and Recharts use.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
}

/**
 * jsdom also has no 2D canvas context, which the radar rasteriser needs, and no
 * `ImageData` constructor. These minimal stubs record the pixel output so
 * rasterisation can be exercised in unit tests. Real pixel output is verified in
 * a browser.
 */

/** Pixels written by the most recent `putImageData` call, per canvas. */
const paintedPixels = new Map<HTMLCanvasElement, Uint8ClampedArray>()

/** Number of `fill()` calls since the last reset. */
let fillCount = 0

if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = function getContext(this: HTMLCanvasElement, kind: string) {
    if (kind !== '2d') return null
    return {
      globalCompositeOperation: 'source-over',
      fillStyle: '',
      createRadialGradient: () => ({
        addColorStop: (offset: number, color: string) => {
          void offset
          void color
        },
      }),
      fill: () => {
        fillCount += 1
      },
      beginPath: () => {},
      arc: () => {},
      clearRect: () => {},
      putImageData: (image: { data: Uint8ClampedArray }) => {
        paintedPixels.set(this, image.data)
      },
      getImageData: (_x: number, _y: number, width: number, height: number) => ({
        data: new Uint8ClampedArray(Math.max(4, width * height * 4)),
      }),
    } as unknown as HTMLCanvasElement['getContext']
  } as HTMLCanvasElement['getContext']

  HTMLCanvasElement.prototype.toDataURL = function toDataURL() {
    return 'data:image/png;base64,'
  }
}

// jsdom implements no ImageData at all, so provide the constructor the
// rasteriser uses.
if (typeof globalThis.ImageData === 'undefined') {
  class StubImageData {
    data: Uint8ClampedArray
    width: number
    height: number
    constructor(dataOrWidth: number | Uint8ClampedArray, widthOrHeight?: number, height?: number) {
      if (typeof dataOrWidth === 'number') {
        this.width = dataOrWidth
        this.height = widthOrHeight ?? 0
        this.data = new Uint8ClampedArray(Math.max(4, this.width * this.height * 4))
      } else {
        this.data = dataOrWidth
        this.width = widthOrHeight ?? 0
        this.height = height ?? 0
      }
    }
  }
  globalThis.ImageData = StubImageData as unknown as typeof ImageData
}

/** Reset the canvas paint counters between tests. */
export function resetCanvasPaintCount(): void {
  fillCount = 0
  paintedPixels.clear()
}

/** Number of `fill()` calls since the last reset. */
export function getCanvasPaintCount(): number {
  return fillCount
}

/**
 * RGBA pixels written to a canvas by the last `putImageData`, or `null` when the
 * canvas was never painted. Lets tests assert on real pixel output.
 */
export function getCanvasPixels(canvas: HTMLCanvasElement | null): Uint8ClampedArray | null {
  if (!canvas) return null
  return paintedPixels.get(canvas) ?? null
}
