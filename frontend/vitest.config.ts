import { defineConfig, mergeConfig } from 'vitest/config'

import viteConfig from './vite.config'

// Kept separate from vite.config.ts so the Vite config stays free of test-only
// types, and so `vitest` can resolve the jsdom environment and setup file.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      css: false,
    },
  }),
)
