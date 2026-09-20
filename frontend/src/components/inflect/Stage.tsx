/**
 * The stage: one chart that stays put while the analysis is drawn onto it.
 *
 * Not a dashboard and not a charting library. Every mark here is placed in
 * data coordinates and has to land exactly where the reader is being told to
 * look, which is the one thing a general-purpose chart component makes hard.
 * It is plain SVG with two linear scales.
 *
 * The camera follows the narration: when a beat names a stretch of the curve,
 * the visible range eases to it and the vertical range eases with it, so the
 * detail being discussed fills the frame. Everything is skipped outright when
 * the reader has asked for reduced motion.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import type { Annotation, CurveResult } from '../../api/inflect'

type Range = [number, number]

const PAD = { top: 34, right: 30, bottom: 46, left: 74 }
const TWEEN_MS = 620

function reducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/** Ease a pair of numbers towards a target, in the browser's own frames. */
function useTween(target: Range, ms = TWEEN_MS): Range {
  const [value, setValue] = useState<Range>(target)
  const from = useRef<Range>(target)
  const start = useRef(0)
  const raf = useRef(0)

  useEffect(() => {
    if (reducedMotion()) {
      setValue(target)
      return
    }
    from.current = value
    start.current = performance.now()
    const step = (now: number) => {
      const t = Math.min(1, (now - start.current) / ms)
      const e = 1 - Math.pow(1 - t, 3)
      setValue([
        from.current[0] + (target[0] - from.current[0]) * e,
        from.current[1] + (target[1] - from.current[1]) * e,
      ])
      if (t < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
    // value is deliberately not a dependency: it is the tween's own output
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target[0], target[1], ms])

  return value
}

function niceTicks(lo: number, hi: number, count = 6): number[] {
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return []
  const raw = (hi - lo) / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const first = Math.ceil(lo / step) * step
  const out: number[] = []
  for (let v = first; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

function fmt(v: number): string {
  if (!isFinite(v)) return ''
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  const decimals = a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3
  return v.toFixed(decimals).replace(/\.?0+$/, '') || '0'
}

export function Stage({
  curve,
  annotations,
  focus,
  height = 520,
  showFits = true,
  beatKey,
}: {
  curve: CurveResult
  annotations: Annotation[]
  focus?: [number, number] | null
  height?: number
  showFits?: boolean
  beatKey: string
}) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(960)

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(320, entry.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const full: Range = [curve.domain[0], curve.domain[1]]

  // What the camera should be looking at, and the vertical range that suits it.
  const targetX = useMemo<Range>(() => {
    if (!focus) return full
    const [a, b] = focus
    const pad = Math.max((b - a) * 0.12, (full[1] - full[0]) * 0.01)
    return [Math.max(full[0], a - pad), Math.min(full[1], b + pad)]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focus?.[0], focus?.[1], full[0], full[1]])

  const targetY = useMemo<Range>(() => {
    const { x, y } = curve.curve
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < x.length; i++) {
      if (x[i] < targetX[0] || x[i] > targetX[1]) continue
      if (y[i] < lo) lo = y[i]
      if (y[i] > hi) hi = y[i]
    }
    if (!isFinite(lo) || !isFinite(hi)) [lo, hi] = curve.range
    const span = hi - lo || Math.abs(hi) || 1
    return [lo - span * 0.12, hi + span * 0.12]
  }, [curve, targetX])

  const xd = useTween(targetX)
  const yd = useTween(targetY)

  const iw = Math.max(10, width - PAD.left - PAD.right)
  const ih = Math.max(10, height - PAD.top - PAD.bottom)
  const sx = (v: number) => PAD.left + ((v - xd[0]) / (xd[1] - xd[0] || 1)) * iw
  const sy = (v: number) => PAD.top + ih - ((v - yd[0]) / (yd[1] - yd[0] || 1)) * ih
  const clampY = (v: number) => Math.max(PAD.top - 40, Math.min(PAD.top + ih + 40, v))

  const path = useMemo(() => {
    const { x, y } = curve.curve
    let d = ''
    for (let i = 0; i < x.length; i++) d += `${i ? 'L' : 'M'}${sx(x[i]).toFixed(1)} ${sy(y[i]).toFixed(1)}`
    return d
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [curve, xd[0], xd[1], yd[0], yd[1], width, height])

  const xTicks = niceTicks(xd[0], xd[1])
  const yTicks = niceTicks(yd[0], yd[1], 5)
  const xLabel = curve.x_unit ? `${curve.x_label} [${curve.x_unit}]` : curve.x_label
  const yLabel = curve.y_unit ? `${curve.y_label} [${curve.y_unit}]` : curve.y_label

  const highlights = annotations.filter((a) => a.kind === 'curve' && a.x0 !== undefined)

  return (
    <div className="stage" ref={box}>
      <svg width={width} height={height} className="stage__svg" role="img"
           aria-label={`${yLabel} against ${xLabel}`}>
        <defs>
          <clipPath id="plot-area">
            <rect x={PAD.left} y={PAD.top - 20} width={iw} height={ih + 40} />
          </clipPath>
        </defs>

        {/* grid and axes ------------------------------------------------ */}
        <g className="stage__grid">
          {yTicks.map((t) => (
            <line key={`gy${t}`} x1={PAD.left} x2={PAD.left + iw} y1={sy(t)} y2={sy(t)} />
          ))}
        </g>
        <g className="stage__axis">
          <line x1={PAD.left} x2={PAD.left + iw} y1={PAD.top + ih} y2={PAD.top + ih} />
          {xTicks.map((t) => (
            <text key={`tx${t}`} x={sx(t)} y={PAD.top + ih + 20} textAnchor="middle">{fmt(t)}</text>
          ))}
          {yTicks.map((t) => (
            <text key={`ty${t}`} x={PAD.left - 12} y={sy(t) + 4} textAnchor="end">{fmt(t)}</text>
          ))}
          <text className="stage__axis-title" x={PAD.left + iw / 2} y={height - 8} textAnchor="middle">{xLabel}</text>
          <text className="stage__axis-title" transform={`rotate(-90 16 ${PAD.top + ih / 2})`} x={16} y={PAD.top + ih / 2} textAnchor="middle">{yLabel}</text>
        </g>

        <g clipPath="url(#plot-area)">
          {/* regions sit behind everything they describe */}
          {annotations.filter((a) => a.kind === 'region').map((a, i) => {
            const x0 = sx(a.x0!)
            const w = Math.max(1, sx(a.x1!) - x0)
            // A label needs room, and eight narrow regimes would otherwise
            // print their names on top of each other.
            const room = a.text ? w > a.text.length * 6.2 + 12 : false
            return (
              <g key={`r${i}`} className={`ann ann--region ann--${a.tone}`}>
                <rect x={x0} y={PAD.top - 12} width={w} height={ih + 24} />
                {room && (
                  <text x={x0 + w / 2} y={PAD.top - 2} textAnchor="middle" className="ann__tag">{a.text}</text>
                )}
              </g>
            )
          })}

          {/* the data */}
          <path key={`c-${curve.file}`} className="stage__curve" d={path} pathLength={1} />

          {highlights.map((a, i) => (
            <path key={`h${i}`} className={`stage__curve stage__curve--lit ann--${a.tone}`} d={path}
                  clipPath={`url(#hl${i})`} />
          ))}
          <defs>
            {highlights.map((a, i) => (
              <clipPath key={`hc${i}`} id={`hl${i}`}>
                <rect x={sx(a.x0!)} y={PAD.top - 20} width={Math.max(1, sx(a.x1!) - sx(a.x0!))} height={ih + 40} />
              </clipPath>
            ))}
          </defs>

          {/* fitted regimes */}
          {showFits && curve.fits.map((f, i) => (
            <path key={`f${i}`} className="stage__fit"
                  d={f.points.map(([px, py], k) => `${k ? 'L' : 'M'}${sx(px).toFixed(1)} ${sy(py).toFixed(1)}`).join('')} />
          ))}

          {/* everything the narration points at */}
          <g key={beatKey} className="ann-layer">
            {annotations.map((a, i) => {
              const tone = `ann--${a.tone}`
              if (a.kind === 'hline' && a.y0 !== undefined) {
                return (
                  <line key={i} className={`ann ann--hline ${tone}`}
                        x1={sx(a.x0 ?? xd[0])} x2={sx(a.x1 ?? xd[1])} y1={sy(a.y0)} y2={sy(a.y0)} />
                )
              }
              if (a.kind === 'vline' && a.x0 !== undefined) {
                return (
                  <g key={i} className={`ann ann--vline ${tone}`}>
                    <line x1={sx(a.x0)} x2={sx(a.x0)} y1={PAD.top} y2={PAD.top + ih} />
                    {a.text && <text x={sx(a.x0)} y={PAD.top - 6} textAnchor="middle" className="ann__tag">{a.text}</text>}
                  </g>
                )
              }
              if (a.kind === 'circle' && a.x0 !== undefined && a.y0 !== undefined) {
                const cx = sx(a.x0)
                const cy = clampY(sy(a.y0))
                const up = cy > PAD.top + 70
                return (
                  <g key={i} className={`ann ann--circle ${tone}`} style={{ animationDelay: `${i * 90}ms` }}>
                    <ellipse cx={cx} cy={cy} rx={17} ry={13} pathLength={1} />
                    <line x1={cx} x2={cx} y1={up ? cy - 13 : cy + 13} y2={up ? cy - 30 : cy + 30} className="ann__leader" />
                    {a.text && (
                      <text x={cx} y={up ? cy - 38 : cy + 46} textAnchor="middle" className="ann__label">
                        {a.text}
                        {a.detail && <tspan className="ann__detail" x={cx} dy="1.15em">{a.detail}</tspan>}
                      </text>
                    )}
                  </g>
                )
              }
              if (a.kind === 'bracket' && a.x0 !== undefined && a.y0 !== undefined && a.y1 !== undefined) {
                const bx = sx(a.x1 ?? a.x0) + 26
                const y0 = clampY(sy(a.y0))
                const y1 = clampY(sy(a.y1))
                return (
                  <g key={i} className={`ann ann--bracket ${tone}`} style={{ animationDelay: `${i * 90}ms` }}>
                    <path d={`M${bx - 7} ${y0} L${bx} ${y0} L${bx} ${y1} L${bx - 7} ${y1}`} />
                    {a.text && (
                      <text x={bx + 8} y={(y0 + y1) / 2 + 4} className="ann__label ann__label--left">{a.text}</text>
                    )}
                  </g>
                )
              }
              if (a.kind === 'slope' && a.x0 !== undefined && a.x1 !== undefined && a.y0 !== undefined && a.y1 !== undefined) {
                const x0 = sx(a.x0)
                const x1 = sx(a.x1)
                const y0 = clampY(sy(a.y0))
                const y1 = clampY(sy(a.y1))
                return (
                  <g key={i} className={`ann ann--slope ${tone}`} style={{ animationDelay: `${i * 90}ms` }}>
                    <path d={`M${x0} ${y0} L${x1} ${y0} L${x1} ${y1}`} className="ann__tri" />
                    {a.text && <text x={x1 + 8} y={(y0 + y1) / 2 + 4} className="ann__label ann__label--left">{a.text}</text>}
                  </g>
                )
              }
              return null
            })}
          </g>
        </g>
      </svg>
      {curve.curve.decimated && (
        <p className="stage__note">
          {curve.n_points.toLocaleString()} points, thinned for drawing — every peak and trough is kept. The analysis uses all of them.
        </p>
      )}
    </div>
  )
}
