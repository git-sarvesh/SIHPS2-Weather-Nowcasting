/**
 * App-wide data context.
 *
 * Owns the requests that several pages need - backend health, model
 * description, checkpoint status, and the currently selected forecast - so the
 * map, charts and history views stay consistent with each other and the backend
 * is polled once rather than per component.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { api, SihpsApi, type ForecastParams } from '../api/client'
import type {
  CheckpointResponse,
  ForecastResponse,
  HealthResponse,
  ModelDescribeResponse,
} from '../api/types'
import { HEALTH_POLL_MS } from '../config'

/** Coarse connectivity state, derived from the health probe. */
export type ConnectionState = 'connecting' | 'online' | 'degraded' | 'offline'

export interface AppState {
  client: SihpsApi
  connection: ConnectionState
  health: HealthResponse | null
  healthError: unknown
  model: ModelDescribeResponse | null
  checkpoint: CheckpointResponse | null
  /** Lead time selected across the map, charts and detail pages. */
  leadHours: number | null
  setLeadHours: (hours: number | null) => void
  forecast: ForecastResponse | null
  forecastLoading: boolean
  forecastError: unknown
  runForecast: (params?: ForecastParams) => Promise<void>
  refreshHealth: () => Promise<void>
  lastUpdated: Date | null
}

const AppContext = createContext<AppState | null>(null)

export function AppProvider({
  children,
  client: injected,
}: {
  children: ReactNode
  client?: SihpsApi
}) {
  const client = useMemo(() => injected ?? api, [injected])

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<unknown>(null)
  const [model, setModel] = useState<ModelDescribeResponse | null>(null)
  const [checkpoint, setCheckpoint] = useState<CheckpointResponse | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [forecastLoading, setForecastLoading] = useState(false)
  const [forecastError, setForecastError] = useState<unknown>(null)
  const [leadHours, setLeadHours] = useState<number | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  // Guards against a slow first forecast overwriting a newer one.
  const requestId = useRef(0)

  const refreshHealth = useCallback(async () => {
    try {
      const response = await client.getHealth()
      setHealth(response)
      setHealthError(null)
      setLastUpdated(new Date())
    } catch (error) {
      // Keep the last good payload so the UI degrades rather than blanks out.
      setHealthError(error)
    }
  }, [client])

  // Initial load: health first (cheap), then the two model descriptors.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      await refreshHealth()
      if (cancelled) return
      const [modelResult, checkpointResult] = await Promise.allSettled([
        client.getModel(),
        client.getCheckpoint(),
      ])
      if (cancelled) return
      if (modelResult.status === 'fulfilled') setModel(modelResult.value)
      if (checkpointResult.status === 'fulfilled') setCheckpoint(checkpointResult.value)
    })()
    return () => {
      cancelled = true
    }
  }, [client, refreshHealth])

  // Poll health so connectivity status stays live.
  useEffect(() => {
    const timer = setInterval(() => void refreshHealth(), HEALTH_POLL_MS)
    return () => clearInterval(timer)
  }, [refreshHealth])

  const runForecast = useCallback(
    async (params: ForecastParams = {}) => {
      const id = ++requestId.current
      setForecastLoading(true)
      setForecastError(null)
      try {
        const response = await client.createForecast({
          ...params,
          lead_hours: params.lead_hours ?? leadHours,
        })
        if (id !== requestId.current) return
        setForecast(response)
        // Adopt the backend's lead times on the first run so the selector is
        // driven by real data rather than a hardcoded list.
        setLeadHours((current) => current ?? response.lead_times_h[0] ?? null)
      } catch (error) {
        if (id !== requestId.current) return
        setForecastError(error)
      } finally {
        if (id === requestId.current) setForecastLoading(false)
      }
    },
    [client, leadHours],
  )

  const connection: ConnectionState = useMemo(() => {
    if (healthError && !health) return 'offline'
    if (!health) return 'connecting'
    if (health.status === 'degraded') return 'degraded'
    if (healthError) return 'degraded'
    return 'online'
  }, [health, healthError])

  const value: AppState = {
    client,
    connection,
    health,
    healthError,
    model,
    checkpoint,
    leadHours,
    setLeadHours,
    forecast,
    forecastLoading,
    forecastError,
    runForecast,
    refreshHealth,
    lastUpdated,
  }

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

/** Access the app state. Throws if used outside {@link AppProvider}. */
export function useApp(): AppState {
  const context = useContext(AppContext)
  if (!context) throw new Error('useApp must be used inside <AppProvider>')
  return context
}
