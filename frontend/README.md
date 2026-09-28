# SIHPS Dashboard

React + Vite + TypeScript front end for the SIH 2026 hyper-local weather nowcasting
system (Uttarakhand AOI). It is a **client only** - it holds no forecasting logic of
its own and renders whatever the FastAPI backend returns.

> **Experimental.** The backend currently serves synthetic data from an untrained
> model. Every view is labelled accordingly. Nothing here is an official India
> Meteorological Department (IMD) warning, and no independent observational
> validation of forecasting skill has been performed.

## Requirements

- Node 20+ (developed on Node 24)
- The backend running on `http://localhost:8000`

## Quick start

```bash
cd backend
python -m app.db upgrade            # create/apply database migrations
python -m uvicorn app.main:app --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev                         # http://localhost:5173
```

The dev server proxies `/api` to `localhost:8000`. The backend's default
`SIHPS_CORS_ORIGINS` already includes `http://localhost:5173`, so direct
cross-origin calls work too.

## Configuration

Copy `.env.example` to `.env.local` and edit as needed:

| Variable                 | Default                                 | Purpose                              |
| ------------------------ | --------------------------------------- | ------------------------------------ |
| `VITE_API_BASE_URL`      | `/api/v1` (relative)                    | Backend base URL including `/api/v1` |
| `VITE_MAP_STYLE_URL`     | `https://demotiles.maplibre.org/style.json` | MapLibre style (keyless default) |
| `VITE_HEALTH_POLL_MS`    | `30000`                                 | `/health` poll interval              |

These reuse the variable names already declared in the repository root
`.env.example`, so there is one place to look.

**Leave `VITE_API_BASE_URL` unset in development.** The default is the relative
`/api/v1`, which the Vite dev server proxies to `http://127.0.0.1:8000` on the
same origin. That removes the CORS handshake entirely. Set an absolute URL only
for a production build served from a different host.

## Troubleshooting

### "Port 5173 is already in use"

`server.strictPort` is enabled on purpose. If something else already holds 5173,
Vite **fails** rather than quietly moving to 5174. That is deliberate: a silent
port move breaks the HMR websocket *and* moves the app to an origin the backend's
`SIHPS_CORS_ORIGINS` does not include, so the UI keeps rendering while silently
losing all API access.

Find and stop the stale process, then retry:

```powershell
Get-NetTCPConnection -LocalPort 5173 -State Listen
Get-Process -Id <OwningProcess> | Format-List Id, Name, StartTime
Stop-Process -Id <OwningProcess> -Force
```

If `Stop-Process` reports **Access is denied**, that process is running
elevated. Close it from an **Administrator** PowerShell, or restart Windows.

### `sw.js` errors in the console

**This application registers no service worker and ships none.** There is no
`sw.js`, no PWA plugin, and no `serviceWorker.register(...)` call anywhere in
`src/`. A request for `/sw.js` is answered by Vite's SPA history fallback, which
returns `index.html` with `Content-Type: text/html` — a browser attempting to
install that as a service worker fails. Any `chrome-extension://` entries in the
log come from a browser extension, not from this project.

To clear a service worker left behind by another app on `localhost:5173`:

1. DevTools → **Application** → **Service Workers** → *Unregister* each entry.
2. DevTools → **Application** → **Storage** → *Clear site data*.
3. Close other tabs on that origin, then hard-reload (Ctrl+Shift+R).

Use an Incognito window, or Chrome with `--disable-extensions`, to confirm the
errors disappear once third-party code is out of the picture.

## Scripts

| Script              | What it does                                        |
| ------------------- | --------------------------------------------------- |
| `npm run dev`       | Dev server with HMR on port 5173                     |
| `npm run lint`      | ESLint (flat config)                                 |
| `npm run typecheck` | `tsc -b` with no emit                                 |
| `npm test`          | Vitest unit + component tests                         |
| `npm run build`     | Typecheck then production build to `dist/`            |
| `npm run preview`   | Serve the production build                            |

## Verifying against the real backend

With the backend running, this exercises every endpoint the dashboard consumes
and asserts the response shapes the UI depends on:

```bash
node scripts/verify-against-backend.mjs
# override the target: SIHPS_API_BASE_URL=http://host:8000/api/v1
```

## Pages

| Page              | Backend endpoints                                                        |
| ----------------- | ------------------------------------------------------------------------ |
| **Overview**      | `GET /health`, `GET /model/describe`, `GET /model/checkpoint`             |
| **Risk Map**      | `POST /risk/geojson` (one request per lead time), `POST /forecast`, `GET /health` |
| **Forecast**      | `POST /forecast`, `GET /forecast/history`, `GET /forecast/{id}`           |
| **Explainability**| `POST /explain` (Grad-CAM, consistency audit, what-if)                    |
| **Uncertainty**   | `POST /forecast` with `include_uncertainty=true`                          |

## Radar map

The Risk Map is a full-screen command-centre view. How it works, and what it
deliberately does not do:

- **Frames are real.** On entering the map the app fetches `POST /risk/geojson`
  once per lead time the backend reports and caches the result. Playback
  interpolates *between those responses* - it is a deterministic replay of the
  model's own output, not generated motion.
- **Smooth precipitation.** Storm cells are rasterised to an offscreen canvas as
  additive radial falloffs, then pushed to the map as an `ImageSource` raster
  layer. This is what removes the blocky polygon look; overlapping cells merge
  into a continuous field.
- **No wind or lightning.** The API exposes no wind field and no lightning data,
  so neither is simulated. Adding either would mean inventing meteorology.
- **Intensity scale.** The colour ramp is driven by the backend's **risk
  probability**. The legend's mm/h column is labelled an illustrative
  equivalence - this system does not measure rainfall rate.
- **Performance.** The animation position lives in a ref and is written straight
  to MapLibre sources, so the 60 fps loop does not re-render React. Every frame,
  listener and the map instance itself are released on unmount.
- **Reduced motion.** With `prefers-reduced-motion: reduce`, playback does not
  auto-advance; manual stepping and scrubbing still work.

### Basemap style

Only the style in `VITE_MAP_STYLE_URL` ships by default (the keyless
`demotiles` sample). To add terrain or satellite layers, append keyless style
URLs to `STYLE_OPTIONS` in `src/pages/RiskMapPage.tsx`; the selector is already
wired. Styles requiring an API key are intentionally not included.

## Design notes

- **Provenance is structural, not decorative.** `components/Provenance.tsx` renders
  the backend's own `disclaimer` and `accuracy_claim` text verbatim whenever
  `is_synthetic` or `demo_mode` is set, and `bannerTone` fails safe: a missing or
  inconsistent flag shows the warning rather than reading as operational.
- **No fabricated values.** The API client throws on failure and never resolves to
  placeholders; empty states render "—" rather than a zero.
- **Uncertainty is labelled honestly.** The Uncertainty page states that with
  untrained weights the MC-dropout spread reflects an untrained network, not
  calibrated forecast error.
- **Bundle splitting.** MapLibre (~1 MB) and Recharts (~400 kB) are split into
  separate chunks and the map, explainability and uncertainty pages are lazily
  loaded, so the initial bundle stays small.
- **Accessibility.** Keyboard-navigable tabs with arrow keys, visible focus rings,
  `aria-label` on panels and chart tables, and `prefers-reduced-motion` support.
