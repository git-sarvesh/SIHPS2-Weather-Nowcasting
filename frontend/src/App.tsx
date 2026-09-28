/**
 * Application root.
 *
 * Owns the current page and provides {@link AppProvider}. Page components are
 * mounted lazily so the heavy MapLibre bundle is only downloaded when the map is
 * first opened.
 */

import { lazy, Suspense, useState } from 'react'

import { Layout, type PageId } from './components/Layout'
import { LoadingState } from './components/ui/States'
import { AppProvider } from './state/AppContext'
import { OverviewPage } from './pages/OverviewPage'
import { ForecastPage } from './pages/ForecastPage'

// MapLibre is the heaviest dependency; keep these views out of the initial bundle.
const RiskMapPage = lazy(() =>
  import('./pages/RiskMapPage').then((m) => ({ default: m.RiskMapPage })),
)
const ExplainPage = lazy(() =>
  import('./pages/ExplainPage').then((m) => ({ default: m.ExplainPage })),
)
const UncertaintyPage = lazy(() =>
  import('./pages/UncertaintyPage').then((m) => ({ default: m.UncertaintyPage })),
)

function PageFallback() {
  return (
    <div className="p-6">
      <LoadingState label="Loading view…" rows={4} />
    </div>
  )
}

function renderPage(page: PageId) {
  switch (page) {
    case 'overview':
      return <OverviewPage />
    case 'map':
      return <RiskMapPage />
    case 'forecast':
      return <ForecastPage />
    case 'explain':
      return <ExplainPage />
    case 'uncertainty':
      return <UncertaintyPage />
    default:
      return <OverviewPage />
  }
}

export default function App() {
  const [page, setPage] = useState<PageId>('overview')

  return (
    <AppProvider>
      <Layout page={page} onNavigate={setPage}>
        <Suspense fallback={<PageFallback />}>{renderPage(page)}</Suspense>
      </Layout>
    </AppProvider>
  )
}
