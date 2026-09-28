/**
 * Runtime configuration read from Vite environment variables.
 *
 * This lives in a normal module (not the ambient `vite-env.d.ts`) so the bundler
 * can resolve it at build time. Values are documented in `.env.example` and reuse
 * the variable names already declared in the repository `.env.example`.
 */

/**
 * Satellite / orthophoto raster tile template.
 *
 * Defaults to Esri's public World Imagery service, which needs no API key.
 * Override to point at any equivalent keyless XYZ raster service
 * (`{z}/{x}/{y}` placeholders are substituted by MapLibre).
 */
export const SATELLITE_TILE_URL: string =
  import.meta.env.VITE_SATELLITE_TILE_URL ??
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

/** Which basemap the Risk Map opens with: `satellite` | `terrain` | `street`. */
export const MAP_STYLE_PRESET: string = import.meta.env.VITE_MAP_STYLE_PRESET ?? 'satellite'


/** How often the dashboard re-polls `/health`, in ms. */
export const HEALTH_POLL_MS: number = (() => {
  const parsed = Number(import.meta.env.VITE_HEALTH_POLL_MS)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 30_000
})()

/** Header title and subtitle. */
export const APP_TITLE = 'SIHPS Hyper-Local Nowcasting'
export const APP_SUBTITLE = 'Uttarakhand AOI · 0–6 h'
