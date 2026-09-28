/// <reference types="vite/client" />

/**
 * Ambient types for the Vite-injected `import.meta.env`.
 *
 * Runtime values live in `src/config.ts`; this file only declares the types and
 * must stay free of runtime exports so the bundler does not try to resolve it.
 */

interface ImportMetaEnv {
  /** Base URL of the FastAPI backend including the version prefix. */
  readonly VITE_API_BASE_URL?: string
  /** MapLibre style URL. The default demotiles style needs no API key. */
  readonly VITE_MAP_STYLE_URL?: string
  /** Health poll interval in milliseconds. */
  readonly VITE_HEALTH_POLL_MS?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
