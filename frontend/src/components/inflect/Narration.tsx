/**
 * The narration: one sentence at a time, with the numbers in it live.
 *
 * There is no panel of sliders anywhere in this version. A setting the reader
 * can change appears as the number it already is, inside the sentence that
 * explains what it does — "the curve has 4 regimes" — and is altered by
 * dragging it or typing over it. A control is easiest to understand when it
 * sits inside its own explanation.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { Beat, Slot } from '../../api/inflect'

const PIXELS_PER_STEP = 14

function format(slot: Slot): string {
  if (!isFinite(slot.value)) return '—'
  return slot.value.toFixed(slot.decimals)
}

/** A number in the prose. Draggable and typable when it drives a setting. */
export function LiveNumber({ slot, onChange }: { slot: Slot; onChange?: (control: string, value: number) => void }) {
  const editable = Boolean(slot.control && onChange)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const drag = useRef<{ x: number; from: number } | null>(null)

  const clamp = useCallback(
    (v: number) => {
      const lo = slot.minimum ?? -Infinity
      const hi = slot.maximum ?? Infinity
      return Math.min(hi, Math.max(lo, v))
    },
    [slot.minimum, slot.maximum],
  )

  const commit = useCallback(
    (v: number) => {
      if (!slot.control || !onChange) return
      const step = slot.step ?? 1
      const snapped = clamp(Math.round(v / step) * step)
      if (snapped !== slot.value) onChange(slot.control, snapped)
    },
    [slot, onChange, clamp],
  )

  if (!editable) {
    return <span className="num live live--static">{format(slot)}</span>
  }

  const onPointerDown = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (editing) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, from: slot.value }
  }
  const onPointerMove = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (!drag.current) return
    const step = slot.step ?? 1
    const delta = Math.round((e.clientX - drag.current.x) / PIXELS_PER_STEP) * step
    if (delta !== 0) commit(drag.current.from + delta)
  }
  const onPointerUp = (e: React.PointerEvent<HTMLSpanElement>) => {
    if (drag.current && Math.abs(e.clientX - drag.current.x) < 3) {
      setDraft(format(slot))
      setEditing(true)
    }
    drag.current = null
  }

  if (editing) {
    return (
      <input
        className="num live live__input"
        autoFocus
        value={draft}
        size={Math.max(2, draft.length)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const v = Number(draft)
          if (isFinite(v)) commit(v)
          setEditing(false)
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
          if (e.key === 'Escape') setEditing(false)
        }}
      />
    )
  }

  return (
    <span
      className="num live live--set"
      role="button"
      tabIndex={0}
      title={`drag to change · ${slot.minimum ?? ''}–${slot.maximum ?? ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onKeyDown={(e) => {
        const step = slot.step ?? 1
        if (e.key === 'ArrowUp' || e.key === 'ArrowRight') commit(slot.value + step)
        if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') commit(slot.value - step)
      }}
    >
      {format(slot)}
    </span>
  )
}

/** A beat's sentence, with its slots rendered in place. */
export function Sentence({ beat, onChange }: { beat: Beat; onChange?: (control: string, value: number) => void }) {
  const by = new Map(beat.slots.map((s) => [s.name, s]))
  const parts = beat.sentence.split(/(\{[a-z0-9_]+\})/i)
  return (
    <p className="narration__sentence">
      {parts.map((part, i) => {
        const m = /^\{([a-z0-9_]+)\}$/i.exec(part)
        const slot = m ? by.get(m[1]) : undefined
        if (!slot) return <span key={i}>{part}</span>
        return <LiveNumber key={i} slot={slot} onChange={onChange} />
      })}
    </p>
  )
}

/** The dots: where you are in the reading, and a way to move through it. */
export function BeatRail({
  beats,
  index,
  onSelect,
}: {
  beats: Beat[]
  index: number
  onSelect: (i: number) => void
}) {
  return (
    <div className="rail" role="tablist" aria-label="reading">
      {beats.map((b, i) => (
        <button
          key={b.id}
          role="tab"
          aria-selected={i === index}
          className={`rail__dot rail__dot--${b.kind} ${i === index ? 'is-on' : ''} ${i < index ? 'is-past' : ''}`}
          onClick={() => onSelect(i)}
          title={b.title}
        >
          <span className="rail__label">{b.title}</span>
        </button>
      ))}
    </div>
  )
}

/** Arrow keys and the space bar move through the reading. */
export function useBeatKeys(count: number, index: number, set: (i: number) => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      // A focused live number takes the arrow keys for itself; they would
      // otherwise both change the setting and move to the next beat.
      if (t && typeof t.closest === 'function' && t.closest('.live--set')) return
      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault()
        set(Math.min(count - 1, index + 1))
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        set(Math.max(0, index - 1))
      }
      if (e.key === 'Home') set(0)
      if (e.key === 'End') set(count - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [count, index, set])
}

/** The numbers behind the current sentence, for a reader who wants them. */
export function BeatStats({ beat }: { beat: Beat }) {
  if (!beat.stats.length) return null
  return (
    <dl className="beat-stats">
      {beat.stats.map((s, i) => (
        <div key={i} className="beat-stats__item">
          <dt>{s.label}</dt>
          <dd className="num">
            {typeof s.value === 'number' ? s.value.toLocaleString(undefined, { maximumSignificantDigits: 6 }) : s.value}
            {s.unit ? <small> {s.unit}</small> : null}
          </dd>
        </div>
      ))}
    </dl>
  )
}
