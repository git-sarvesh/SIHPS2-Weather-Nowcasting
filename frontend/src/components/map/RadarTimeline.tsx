/** Radar playback timeline: play/pause, step, and a draggable lead-time scrubber. */

import type { TimelineFrame } from './mapTypes'
import { formatLeadHours } from '../../lib/format'

export interface RadarTimelineProps {
  frames: TimelineFrame[]
  /** Fractional position between frames, 0 .. frames.length - 1. */
  position: number
  playing: boolean
  reducedMotion: boolean
  onScrub: (position: number) => void
  onTogglePlay: () => void
  onStep: (delta: number) => void
  /** Lead time of the frame currently on screen. */
  leadHours: number
  validTime: string
  isSynthetic: boolean
}

function TransportButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/12
                 bg-slate-900/70 text-slate-200 transition hover:border-sky-400/50 hover:bg-sky-400/12
                 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {children}
    </button>
  )
}

export function RadarTimeline(props: RadarTimelineProps) {
  const {
    frames,
    position,
    playing,
    reducedMotion,
    onScrub,
    onTogglePlay,
    onStep,
    leadHours,
    validTime,
    isSynthetic,
  } = props

  const max = Math.max(0, frames.length - 1)
  const atStart = position <= 0.001
  const atEnd = position >= max - 0.001

  return (
    <div
      className="pointer-events-auto w-full rounded-xl border border-white/12 bg-slate-950/72
                 px-4 py-3 shadow-2xl backdrop-blur-xl"
      aria-label="Radar timeline"
    >
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <TransportButton label="Previous frame" onClick={() => onStep(-1)} disabled={atStart}>
            <span aria-hidden="true">⏮</span>
          </TransportButton>

          <button
            type="button"
            onClick={onTogglePlay}
            aria-label={playing ? 'Pause radar animation' : 'Play radar animation'}
            aria-pressed={playing}
            title={playing ? 'Pause' : 'Play'}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-sky-400/45
                       bg-sky-400/15 text-sky-200 transition hover:bg-sky-400/25
                       disabled:opacity-40"
          >
            <span aria-hidden="true">{playing ? '❚❚' : '▶'}</span>
          </button>

          <TransportButton label="Next frame" onClick={() => onStep(1)} disabled={atEnd}>
            <span aria-hidden="true">⏭</span>
          </TransportButton>
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex items-baseline justify-between gap-3">
            <p className="truncate font-mono text-[11px] text-slate-200">
              <span className="text-sky-300">T+{formatLeadHours(leadHours)}</span>
              <span className="mx-1.5 text-slate-600">·</span>
              <span className="text-slate-400">{validTime || 'valid time unavailable'}</span>
            </p>
            <p className="shrink-0 font-mono text-[10px] text-slate-500">
              frame {Math.round(position) + 1} / {frames.length}
            </p>
          </div>

          <input
            type="range"
            min={0}
            max={max}
            step={0.01}
            value={position}
            onChange={(event) => onScrub(Number(event.target.value))}
            aria-label="Forecast lead time"
            aria-valuetext={`Lead time ${formatLeadHours(leadHours)}`}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-700/80
                       accent-sky-400 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5
                       [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-sky-400 [&::-webkit-slider-thumb]:shadow-glow"
          />

          <div className="mt-1.5 flex justify-between">
            {frames.map((frame) => (
              <button
                key={frame.index}
                type="button"
                onClick={() => onScrub(frame.index)}
                disabled={!frame.loaded}
                title={
                  frame.loaded
                    ? `Jump to ${formatLeadHours(frame.leadHours)}`
                    : `${formatLeadHours(frame.leadHours)} (no frame returned)`
                }
                className={`px-1 font-mono text-[9px] transition disabled:opacity-35 ${
                  Math.abs(position - frame.index) < 0.5
                    ? 'font-bold text-sky-300'
                    : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {formatLeadHours(frame.leadHours)}
              </button>
            ))}
          </div>
        </div>

        {isSynthetic && (
          <span
            className="pill shrink-0 border-amber-400/45 bg-amber-400/12 text-amber-300"
            title="Frames are model output from the synthetic demo generator, not observations."
          >
            <span aria-hidden="true">◆</span>
            SYNTHETIC DEMO
          </span>
        )}
        {reducedMotion && playing && (
          <span className="shrink-0 text-[9px] text-slate-500">reduced-motion</span>
        )}
      </div>

      <p className="mt-2 text-[9px] leading-relaxed text-slate-500">
        Animation interpolates between the model&rsquo;s own forecast lead times. It is a
        deterministic playback of synthetic demo output — not observed radar motion.
      </p>
    </div>
  )
}
