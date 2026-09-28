/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark geospatial palette.
        base: {
          900: '#05070d',
          800: '#0a0f1a',
          700: '#111827',
          600: '#1b2434',
          500: '#2b3648',
        },
        // Risk categories, ordered low -> extreme. Kept consistent with the
        // backend thresholds (0.3 / 0.6 / 0.85) in app/services/risk_engine.py.
        risk: {
          low: '#22d3ee',
          moderate: '#facc15',
          high: '#fb923c',
          extreme: '#f43f5e',
        },
        accent: {
          DEFAULT: '#38bdf8',
          dim: '#0ea5e9',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 24px -6px rgba(56, 189, 248, 0.45)',
        'glow-lg': '0 0 48px -12px rgba(56, 189, 248, 0.5)',
      },
      keyframes: {
        pulseGlow: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        pulseGlow: 'pulseGlow 2.4s ease-in-out infinite',
        sweep: 'sweep 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
