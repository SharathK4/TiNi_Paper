/**
 * Every curve in the set, on one pair of axes.
 *
 * The switch that matters is the second arrangement. Shifting each curve so
 * that its own transition sits at zero removes the scatter between repeats,
 * and what survives is the shape they all share — the thing a reader is
 * usually trying to see and which a hundred overlaid traces normally hide.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import type { Arrangement, ChorusResult } from '../../api/inflect'

const PAD = { top: 26, right: 26, bottom: 40, left: 68 }

function niceTicks(lo: number, hi: number, count = 6): number[] {
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return []
  const raw = (hi - lo) / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const n = raw / mag
  const step = (n >= 7.5 ? 10 : n >= 3.5 ? 5 : n >= 1.5 ? 2 : 1) * mag
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  }
  return out
}

function fmt(v: number): string {
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  return v.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3).replace(/\.?0+$/, '') || '0'
}

export function Chorus({ data, height = 400 }: { data: ChorusResult; height?: number }) {
  const [mode, setMode] = useState<'measured' | 'aligned'>(data.aligned ? 'aligned' : 'measured')
  const [width, setWidth] = useState(980)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const arr: Arrangement | null = mode === 'aligned' ? data.aligned : data.measured
  const geometry = useMemo(() => {
    if (!arr) return null
    const all: number[] = []
    for (const row of arr.lines) {
      for (const v of row) if (v !== null && isFinite(v)) all.push(v)
    }
    if (all.length < 2) return null
    all.sort((a, b) => a - b)
    // Scaled to hold almost all of the data rather than all of it. A handful
    // of curves from another population would otherwise compress the rest
    // into a fifth of the frame — and those curves are named on their own
    // further down the page, where they can be read properly.
    const at = (q: number) => all[Math.min(all.length - 1, Math.max(0, Math.round(q * (all.length - 1))))]
    const lo = at(0.004)
    const hi = at(0.996)
    const pad = (hi - lo) * 0.08 || 1
    return { lo: lo - pad, hi: hi + pad, x0: arr.grid[0], x1: arr.grid[arr.grid.length - 1] }
  }, [arr])

  if (!arr || !geometry) {
    return <p className="ens__none">Not enough curves to draw them together.</p>
  }

  const iw = Math.max(10, width - PAD.left - PAD.right)
  const ih = Math.max(10, height - PAD.top - PAD.bottom)
  const sx = (v: number) => PAD.left + ((v - geometry.x0) / (geometry.x1 - geometry.x0 || 1)) * iw
  const sy = (v: number) => PAD.top + ih - ((v - geometry.lo) / (geometry.hi - geometry.lo || 1)) * ih

  const pathOf = (row: (number | null)[]) => {
    let d = ''
    let pen = false
    for (let i = 0; i < row.length; i++) {
      const v = row[i]
      if (v === null || !isFinite(v)) {
        pen = false
        continue
      }
      d += `${pen ? 'L' : 'M'}${sx(arr.grid[i]).toFixed(1)} ${sy(v).toFixed(1)}`
      pen = true
    }
    return d
  }

  const ribbon = (() => {
    const top: string[] = []
    const bottom: string[] = []
    for (let i = 0; i < arr.grid.length; i++) {
      const a = arr.high[i]
      const b = arr.low[i]
      if (a === null || b === null || !isFinite(a) || !isFinite(b)) continue
      top.push(`${sx(arr.grid[i]).toFixed(1)} ${sy(a).toFixed(1)}`)
      bottom.push(`${sx(arr.grid[i]).toFixed(1)} ${sy(b).toFixed(1)}`)
    }
    if (!top.length) return ''
    return `M${top.join('L')}L${bottom.reverse().join('L')}Z`
  })()

  const xLabel = mode === 'aligned'
    ? `${data.x_label} from each curve's own transition${data.x_unit ? ` [${data.x_unit}]` : ''}`
    : data.x_unit ? `${data.x_label} [${data.x_unit}]` : data.x_label
  const yLabel = data.y_unit ? `${data.y_label} [${data.y_unit}]` : data.y_label

  return (
    <div className="chorus" ref={box}>
      <div className="chorus__head">
        <div className="chorus__switch" role="tablist" aria-label="arrangement">
          <button role="tab" aria-selected={mode === 'measured'}
                  className={mode === 'measured' ? 'is-on' : ''}
                  onClick={() => setMode('measured')}>as measured</button>
          <button role="tab" aria-selected={mode === 'aligned'}
                  className={mode === 'aligned' ? 'is-on' : ''}
                  disabled={!data.aligned}
                  onClick={() => setMode('aligned')}>aligned on the transition</button>
        </div>
        <p className="chorus__count num">
          {arr.n_curves} curves · {arr.n_drawn} drawn
        </p>
      </div>

      <svg width={width} height={height} className="chorus__svg" key={mode}
           role="img" aria-label={`${arr.n_curves} curves, ${arr.origin}`}>
        <defs>
          <clipPath id="chorus-area">
            <rect x={PAD.left} y={PAD.top - 4} width={iw} height={ih + 8} />
          </clipPath>
        </defs>
        <g className="chorus__axis">
          <line x1={PAD.left} x2={PAD.left + iw} y1={PAD.top + ih} y2={PAD.top + ih} />
          {niceTicks(geometry.x0, geometry.x1).map((t) => (
            <text key={`x${t}`} x={sx(t)} y={PAD.top + ih + 19} textAnchor="middle">{fmt(t)}</text>
          ))}
          {niceTicks(geometry.lo, geometry.hi, 5).map((t) => (
            <text key={`y${t}`} x={PAD.left - 10} y={sy(t) + 4} textAnchor="end">{fmt(t)}</text>
          ))}
          <text className="chorus__title" x={PAD.left + iw / 2} y={height - 6} textAnchor="middle">{xLabel}</text>
          <text className="chorus__title" transform={`rotate(-90 14 ${PAD.top + ih / 2})`} x={14} y={PAD.top + ih / 2} textAnchor="middle">{yLabel}</text>
        </g>

        {mode === 'aligned' && (
          <line className="chorus__origin" x1={sx(0)} x2={sx(0)} y1={PAD.top} y2={PAD.top + ih} />
        )}

        <g clipPath="url(#chorus-area)">
          <g className="chorus__lines">
            {arr.lines.map((row, i) => (
              <path key={i} d={pathOf(row)} style={{ animationDelay: `${Math.min(i, 60) * 7}ms` }} />
            ))}
          </g>
          {ribbon && <path className="chorus__ribbon" d={ribbon} />}
          <path className="chorus__median" d={pathOf(arr.median)} pathLength={1} />
        </g>
      </svg>

      <p className="chorus__note">
        {mode === 'aligned'
          ? 'Each curve shifted so its own transition sits at zero. What survives the shift is common to every run; what does not was the scatter between them.'
          : 'Every curve on the axis it was measured on. Slate: the median, with the middle half of the curves shaded.'}
      </p>
    </div>
  )
}
