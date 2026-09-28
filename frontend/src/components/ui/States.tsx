/** Loading, error and empty states used by every data-driven panel. */

import type { ReactNode } from 'react'

import { ApiError } from '../../api/client'

/** Shimmering placeholder used while a panel's data is in flight. */
export function LoadingState({ label = 'Loading…', rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div className="flex flex-col gap-3 p-4" role="status" aria-live="polite">
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
        </span>
        {label}
      </div>
      {Array.from({ length: rows }, (_, index) => (
        <div
          key={index}
          className="h-3 w-full overflow-hidden rounded bg-base-700"
          style={{ width: `${100 - index * 12}%` }}
        >
          <div className="h-full w-1/3 animate-sweep rounded bg-gradient-to-r from-transparent via-base-500 to-transparent" />
        </div>
      ))}
    </div>
  )
}

/**
 * Error panel that shows the backend's own message.
 *
 * Distinguishes an unreachable backend (a network/CORS problem, actionable for
 * the operator) from a rejected request, so the fix differs in each case.
 */
/** Extract a human-readable message from an API or generic error. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.detail || error.message
  if (error instanceof Error) return error.message
  return 'An unexpected error occurred.'
}

export function ErrorState({
  error,
  onRetry,
  context,
}: {
  error: unknown
  onRetry?: () => void
  context?: string
}) {
  const isApi = error instanceof ApiError
  const isNetwork = isApi && error.isNetworkError
  const message = describeError(error)

  return (
    <div
      className="m-4 rounded-lg border border-risk-extreme/40 bg-risk-extreme/10 p-4"
      role="alert"
    >
      <div className="flex items-start gap-3">
        <span aria-hidden="true" className="mt-0.5 text-lg leading-none text-risk-extreme">
          ⚠
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-risk-extreme">
            {isNetwork ? 'Backend unreachable' : (context ?? 'Request failed')}
          </p>
          <p className="mt-1 break-words text-xs text-slate-300">{message}</p>
          {isNetwork && (
            <p className="mt-2 text-[11px] text-slate-400">
              Start the backend with{' '}
              <code className="rounded bg-base-900 px-1 py-0.5 font-mono text-accent">
                uvicorn app.main:app --port 8000
              </code>{' '}
              and confirm CORS allows this origin.
            </p>
          )}
          {!isNetwork && isApi && error.status > 0 && (
            <p className="mt-2 font-mono text-[11px] text-slate-400">HTTP {error.status}</p>
          )}
        </div>
        {onRetry && (
          <button type="button" className="btn shrink-0" onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
    </div>
  )
}

/** Neutral empty state for a panel that loaded successfully but has no rows. */
export function EmptyState({
  title = 'No data',
  message,
  action,
}: {
  title?: string
  message?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center">
      <span aria-hidden="true" className="text-2xl text-slate-600">
        ◌
      </span>
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {message && <p className="max-w-sm text-xs text-slate-500">{message}</p>}
      {action}
    </div>
  )
}

/**
 * Small labelled metric tile.
 *
 * `value` accepts a string so a caller can render "—" instead of a fabricated
 * zero when a value is genuinely absent.
 */
export function StatTile({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'default' | 'accent' | 'warn' | 'danger'
}) {
  const toneClass = {
    default: 'text-slate-100',
    accent: 'text-accent',
    warn: 'text-risk-moderate',
    danger: 'text-risk-extreme',
  }[tone]

  return (
    <div className="rounded-lg border border-base-600/60 bg-base-900/50 px-3 py-2.5">
      <p className="label truncate">{label}</p>
      <p className={`mt-1 font-mono text-base leading-tight ${toneClass}`}>{value}</p>
      {hint && <p className="mt-0.5 text-[10px] leading-tight text-slate-500">{hint}</p>}
    </div>
  )
}
