/**
 * Choosing what to plot.
 *
 * The tool's own choice of columns is a good default and a bad monopoly. A
 * reader may want any pair of the twenty-one columns in a simulation log, and
 * the only honest way to decide is to see them — so every column is listed
 * with its own shape drawn beside it, and the pair currently chosen is drawn
 * large, from data already in the browser, so picking is instant.
 *
 * The tool's suggestion is kept visible and kept first, with the reason it was
 * made. It is a recommendation the reader can overrule, not a decision taken
 * on their behalf.
 */

import { useMemo, useState } from 'react'
import type { Preview, PreviewColumn } from '../../api/inflect'

function fmt(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  return v.toPrecision(digits).replace(/\.?0+$/, '') || '0'
}

/** A column's own shape, drawn against its row order. */
function Thumb({ col }: { col: PreviewColumn }) {
  const d = useMemo(() => {
    const v = col.values
    const finite = v.filter((n): n is number => n !== null && isFinite(n))
    if (finite.length < 2) return ''
    const lo = Math.min(...finite)
    const hi = Math.max(...finite)
    const span = hi - lo || 1
    let path = ''
    let pen = false
    for (let i = 0; i < v.length; i++) {
      const n = v[i]
      if (n === null || !isFinite(n)) {
        pen = false
        continue
      }
      const x = (i / (v.length - 1)) * 56
      const y = 16 - ((n - lo) / span) * 14
      path += `${pen ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`
      pen = true
    }
    return path
  }, [col])
  return (
    <svg viewBox="0 0 56 18" preserveAspectRatio="none" className="colthumb" aria-hidden="true">
      {d ? <path d={d} /> : null}
    </svg>
  )
}

/** The pair currently chosen, drawn large. */
function Pair({ x, y, xUnit, yUnit }: { x: PreviewColumn; y: PreviewColumn; xUnit: string | null; yUnit: string | null }) {
  const W = 1000
  const H = 300
  const PAD = { l: 78, r: 26, t: 22, b: 52 }
  const geom = useMemo(() => {
    const pts: [number, number][] = []
    for (let i = 0; i < Math.min(x.values.length, y.values.length); i++) {
      const a = x.values[i]
      const b = y.values[i]
      if (a === null || b === null || !isFinite(a) || !isFinite(b)) continue
      pts.push([a, b])
    }
    if (pts.length < 2) return null
    const xs = pts.map((p) => p[0])
    const ys = pts.map((p) => p[1])
    const x0 = Math.min(...xs)
    const x1 = Math.max(...xs)
    const y0 = Math.min(...ys)
    const y1 = Math.max(...ys)
    const padY = (y1 - y0) * 0.08 || 1
    return { pts, x0, x1, y0: y0 - padY, y1: y1 + padY }
  }, [x, y])

  if (!geom) {
    return <p className="pair__empty">These two columns have no pairs of readable numbers.</p>
  }

  const iw = W - PAD.l - PAD.r
  const ih = H - PAD.t - PAD.b
  const sx = (v: number) => PAD.l + ((v - geom.x0) / (geom.x1 - geom.x0 || 1)) * iw
  const sy = (v: number) => PAD.t + ih - ((v - geom.y0) / (geom.y1 - geom.y0 || 1)) * ih
  // Sorted by x so the line reads left to right even when the chosen x is not
  // the row order; an unsorted x is exactly the case a reader needs to see.
  const ordered = [...geom.pts].sort((a, b) => a[0] - b[0])
  const path = ordered.map(([a, b], i) => `${i ? 'L' : 'M'}${sx(a).toFixed(1)} ${sy(b).toFixed(1)}`).join('')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="pair__svg" role="img"
         aria-label={`${y.label} against ${x.label}`}>
      <line className="pair__axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
      <line className="pair__axis" x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ih} />
      <text className="pair__tick" x={PAD.l} y={H - 24} textAnchor="start">{fmt(geom.x0)}</text>
      <text className="pair__tick" x={W - PAD.r} y={H - 24} textAnchor="end">{fmt(geom.x1)}</text>
      <text className="pair__tick" x={PAD.l - 10} y={PAD.t + 10} textAnchor="end">{fmt(geom.y1)}</text>
      <text className="pair__tick" x={PAD.l - 10} y={PAD.t + ih} textAnchor="end">{fmt(geom.y0)}</text>
      <path className="pair__line" d={path} />
      <text className="pair__label" x={PAD.l + iw / 2} y={H - 6} textAnchor="middle">
        {xUnit ? `${x.name} [${xUnit}]` : x.label}
      </text>
      <text className="pair__label" transform={`rotate(-90 16 ${PAD.t + ih / 2})`} x={16} y={PAD.t + ih / 2} textAnchor="middle">
        {yUnit ? `${y.name} [${yUnit}]` : y.label}
      </text>
    </svg>
  )
}

function Option({
  col,
  role,
  chosen,
  suggested,
  onPick,
}: {
  col: PreviewColumn
  role: 'x' | 'y'
  chosen: boolean
  suggested: boolean
  onPick: () => void
}) {
  const reasons = role === 'x' ? col.x_reasons : col.y_reasons
  const why = reasons[0] ?? col.kind_reason
  return (
    <button
      className={`colopt ${chosen ? 'is-on' : ''} ${col.constant ? 'is-flat' : ''}`}
      onClick={onPick}
      aria-pressed={chosen}
    >
      <Thumb col={col} />
      <span className="colopt__body">
        <span className="colopt__name">
          {col.name}
          {col.unit && <span className="colopt__unit"> [{col.unit}]</span>}
          {suggested && <span className="colopt__tag">suggested</span>}
          {col.duplicate_of && <span className="colopt__dup">same as {col.duplicate_of}</span>}
        </span>
        <span className="colopt__why">{why}</span>
      </span>
      <span className="colopt__range num">{fmt(col.minimum)} – {fmt(col.maximum)}</span>
    </button>
  )
}

export function Columns({
  preview,
  onConfirm,
  onBack,
  busy,
}: {
  preview: Preview
  onConfirm: (x: number, y: number) => void
  onBack?: () => void
  busy: boolean
}) {
  const [x, setX] = useState<number>(preview.chosen_x ?? preview.recommended_x ?? 0)
  const [y, setY] = useState<number>(
    preview.chosen_y ?? preview.recommended_y ?? Math.min(1, preview.columns.length - 1),
  )
  const [showAll, setShowAll] = useState(false)

  const byIndex = useMemo(() => new Map(preview.columns.map((c) => [c.index, c])), [preview])
  const colX = byIndex.get(x)
  const colY = byIndex.get(y)

  // Ranked first, then everything the ranking left out, so nothing is hidden.
  const order = (ranking: number[]) => {
    const rest = preview.columns.map((c) => c.index).filter((i) => !ranking.includes(i))
    return [...ranking, ...rest]
  }
  const xOrder = useMemo(() => order(preview.x_ranking), [preview])
  const yOrder = useMemo(() => order(preview.y_ranking), [preview])
  const cut = showAll ? Infinity : 6
  const hiddenCount = Math.max(xOrder.length - cut, yOrder.length - cut, 0)

  const swap = () => {
    setX(y)
    setY(x)
  }

  return (
    <section className="choose">
      <div className="choose__head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="choose__dot" aria-hidden="true" />
          <h2 className="intake__eyebrow" style={{ marginBottom: 0 }}>What should we plot?</h2>
        </div>
        <p className="choose__file num">{preview.file} · {preview.n_rows.toLocaleString()} rows · {preview.columns.length} columns</p>
      </div>

      <p className="choose__sentence">
        Reading <b>{colY?.name ?? '—'}</b> against <b>{colX?.name ?? '—'}</b>.{' '}
        <span className="choose__hint">
          {x === preview.recommended_x && y === preview.recommended_y
            ? 'This is the pair the tool suggests — change either if you want a different view.'
            : 'Your choice, not the suggestion.'}
        </span>
      </p>

      <div className="choose__pair">
        {colX && colY ? <Pair x={colX} y={colY} xUnit={preview.x_unit} yUnit={preview.y_unit} /> : null}
      </div>

      <div className="choose__lists">
        <div className="choose__col">
          <h3>Horizontal axis</h3>
          {xOrder.slice(0, cut).map((i) => {
            const c = byIndex.get(i)
            return c ? (
              <Option key={`x${i}`} col={c} role="x" chosen={x === i}
                      suggested={i === preview.recommended_x} onPick={() => setX(i)} />
            ) : null
          })}
        </div>
        <div className="choose__col">
          <h3>What to read</h3>
          {yOrder.slice(0, cut).map((i) => {
            const c = byIndex.get(i)
            return c ? (
              <Option key={`y${i}`} col={c} role="y" chosen={y === i}
                      suggested={i === preview.recommended_y} onPick={() => setY(i)} />
            ) : null
          })}
        </div>
      </div>

      <div className="choose__actions">
        <button className="btn btn--primary" disabled={busy || colX === undefined || colY === undefined || x === y}
                onClick={() => onConfirm(x, y)}>
          {busy ? 'reading…' : 'Read this curve'}
        </button>
        <button className="btn btn--quiet" onClick={swap} disabled={busy}>swap the axes</button>
        {hiddenCount > 0 && (
          <button className="btn btn--quiet" onClick={() => setShowAll(!showAll)}>
            {showAll ? 'show fewer' : `show all ${preview.columns.length} columns`}
          </button>
        )}
        {onBack && <button className="btn btn--quiet" onClick={onBack} disabled={busy}>use different data</button>}
        {x === y && <span className="choose__warn">Pick two different columns.</span>}
      </div>
    </section>
  )
}
