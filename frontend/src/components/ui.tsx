import type { ReactNode } from 'react'
import { CartesianGrid, XAxis, YAxis } from 'recharts'

/* Shared page furniture. Panels are the numbered sections; figures are the
 * chart slots inside them; readouts are the quiet number rows underneath. */

export function Panel({ index, title, lead, children, id }: { index: string; title: string; lead?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section className="panel" id={id}>
      <div className="panel__index">{index}</div>
      <h2 className="panel__title">{title}</h2>
      {lead && <p className="panel__lead">{lead}</p>}
      <div className="panel__body">{children}</div>
    </section>
  )
}

export function Figure({ title, note, children, className }: { title?: ReactNode; note?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <figure className={`figure ${className ?? ''}`}>
      {title && <div className="figure__title">{title}</div>}
      {children}
      {note && <figcaption className="figure__note">{note}</figcaption>}
    </figure>
  )
}

export function Readout({ label, value, unit, tone }: { label: ReactNode; value: ReactNode; unit?: string; tone?: 'model' | 'accent' }) {
  return (
    <div className={`readout ${tone ? `readout--${tone}` : ''}`}>
      <div className="readout__label">{label}</div>
      <div className="readout__value">
        {value}
        {unit && <small>{unit}</small>}
      </div>
    </div>
  )
}

export function Seg<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} className={`seg__btn ${o.value === value ? 'is-on' : ''}`} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Slider({ label, value, min, max, step, onChange, format }: { label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string }) {
  return (
    <label className="control">
      <span className="control__label">
        <span>{label}</span>
        <b>{format ? format(value) : value}</b>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

/* ------------------------------------------------------------ chart theme */
export const chart = {
  ink: '#15161d',
  ink2: '#6e7180',
  ink3: '#9698a8',
  ink4: '#babcca',
  grid: '#e9eaf5',
  accent: '#b5462a',
  accent2: '#d4703f',
  model: '#2f3a63',
  model2: '#6b77a8',
  series: ['#7c8199', '#a5a9bd', '#8a8fa6', '#b3b6c8', '#9a9eb3', '#c1c3d3', '#858a9f', '#adb0c2', '#767b92', '#b9bccc', '#9094aa', '#c8cad8'],
  margin: { top: 12, right: 18, bottom: 8, left: 4 },
}

const axisStyle = { fontSize: 11, fill: chart.ink3, fontFamily: 'inherit' }
/** Axis ticks: drop float noise (24.000000001 → 24) but keep up to 4 significant digits. */
export const tick = (v: number) => (Number.isFinite(v) ? String(parseFloat(v.toPrecision(4))) : '')

/** Round tick positions covering [lo, hi] with about `n` steps (1-2-5 series). */
export function niceTicks(lo: number, hi: number, n = 6): number[] {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / n
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const r = raw / mag
  const step = (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(parseFloat(v.toPrecision(10)))
  return out
}
const labelStyle = { fontSize: 11, fill: chart.ink2 }

export const Grid = () => <CartesianGrid stroke={chart.grid} strokeWidth={1} vertical={false} />

export const XNum = ({ dataKey, label, domain, ticks, range }: { dataKey: string; label: string; domain?: [number | string, number | string]; ticks?: number[]; range?: [number, number] }) => (
  <XAxis
    type="number"
    dataKey={dataKey}
    domain={domain ?? (range ? range : ['dataMin', 'dataMax'])}
    ticks={ticks ?? (range ? niceTicks(range[0], range[1]) : undefined)}
    tick={axisStyle}
    tickFormatter={tick}
    tickLine={false}
    axisLine={{ stroke: '#c6c9d9' }}
    label={{ value: label, position: 'insideBottomRight', offset: -4, ...labelStyle }}
    height={36}
    allowDecimals
  />
)

export const YNum = ({ label, domain, width = 52, range }: { label: string; domain?: [number | string, number | string]; width?: number; range?: [number, number] }) => (
  <YAxis
    type="number"
    domain={domain ?? (range ? range : ['auto', 'auto'])}
    ticks={range ? niceTicks(range[0], range[1], 5) : undefined}
    tick={axisStyle}
    tickFormatter={tick}
    tickLine={false}
    axisLine={false}
    width={width}
    label={{ value: label, angle: -90, position: 'insideLeft', offset: 14, ...labelStyle }}
  />
)

export function Tip({ active, payload, label, unitX, fmtX, rows }: { active?: boolean; payload?: { name?: string; value?: number; color?: string; dataKey?: string }[]; label?: number; unitX: string; fmtX?: (v: number) => string; rows?: (p: { name?: string; value?: number; color?: string; dataKey?: string }) => ReactNode }) {
  if (!active || !payload?.length || label === undefined) return null
  return (
    <div className="tooltip num">
      <div>
        <b>{fmtX ? fmtX(label) : label}</b> {unitX}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color ?? chart.ink2 }}>
          {rows ? rows(p) : `${p.name}: ${typeof p.value === 'number' ? p.value.toFixed(2) : p.value}`}
        </div>
      ))}
    </div>
  )
}
