/**
 * Map status banner.
 *
 * The regression this guards against: when the basemap style could not be
 * loaded, MapLibre rendered an empty dark canvas while the rest of the UI
 * still reported a healthy, fully populated map. Every non-ready state is
 * therefore shown explicitly, with an actionable next step.
 */

export type MapNoticeTone = 'info' | 'warn' | 'error'

export interface MapNotice {
  tone: MapNoticeTone
  title: string
  detail: string
  /** Label for the recovery action, when one exists. */
  action?: string
  onAction?: () => void
}

const TONE_STYLES: Record<MapNoticeTone, string> = {
  info: 'border-sky-400/40 bg-sky-500/15 text-sky-100',
  warn: 'border-amber-400/45 bg-amber-500/15 text-amber-100',
  error: 'border-rose-400/45 bg-rose-500/15 text-rose-100',
}

export interface MapNoticeBannerProps {
  notices: MapNotice[]
  onRetry: () => void
}

export function MapNoticeBanner({ notices, onRetry }: MapNoticeBannerProps) {
  if (notices.length === 0) return null
  return (
    <div
      className="pointer-events-none absolute inset-x-3 top-3 z-20 flex flex-col items-center gap-2
                 lg:inset-x-auto lg:left-1/2 lg:top-4 lg:-translate-x-1/2 lg:w-[min(30rem,calc(100%-24rem))]"
      role="status"
      aria-live="polite"
    >
      {notices.map((notice) => (
        <div
          key={`${notice.title}-${notice.detail}`}
          className={`pointer-events-auto flex w-full items-start gap-3 rounded-xl border
                      px-3.5 py-2.5 shadow-2xl backdrop-blur-xl ${TONE_STYLES[notice.tone]}`}
        >
          <span aria-hidden="true" className="mt-0.5 text-sm leading-none">
            {notice.tone === 'error' ? '⚠' : notice.tone === 'warn' ? '⚡' : '◈'}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-bold uppercase tracking-[0.12em]">{notice.title}</p>
            <p className="mt-0.5 text-[11px] leading-snug opacity-90">{notice.detail}</p>
          </div>
          {notice.action ? (
            <button
              type="button"
              onClick={notice.onAction ?? onRetry}
              className="shrink-0 rounded-md border border-current/40 px-2 py-1 text-[10px]
                         font-semibold uppercase tracking-wide transition hover:bg-white/10"
            >
              {notice.action}
            </button>
          ) : (
            <button
              type="button"
              onClick={onRetry}
              className="shrink-0 rounded-md border border-current/40 px-2 py-1 text-[10px]
                         font-semibold uppercase tracking-wide transition hover:bg-white/10"
            >
              Retry
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
