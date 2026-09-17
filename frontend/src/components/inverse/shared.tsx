import type { ReactNode } from 'react'
import { XAxis } from 'recharts'
import type { InvParams } from '../../api/inverse'
import { fixed, sci } from '../../lib/format'
import { M } from '../Math'

/* Shared pieces for the inverse-problem page. */

export const PARAM_LABEL: Record<keyof InvParams, ReactNode> = {
  U_barrier: <M tex="U^\ddagger" />,
  xi_barrier: <M tex="\xi^\ddagger" />,
  gamma0: <M tex="\Gamma_0" />,
}
export const PARAM_UNIT: Record<keyof InvParams, string> = { U_barrier: 'pN·Å', xi_barrier: 'Å', gamma0: 'ps⁻¹' }
export const PARAM_KEYS: (keyof InvParams)[] = ['U_barrier', 'xi_barrier', 'gamma0']

export const fmtParam = (k: keyof InvParams, v: number | null | undefined, digits?: number): string => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (k === 'gamma0') return sci(v, digits ?? 2)
  return fixed(v, digits ?? (k === 'U_barrier' ? 0 : 1))
}

/** "a – b" with open ends rendered honestly. */
export const interval = (k: keyof InvParams, lo: number | null | undefined, hi: number | null | undefined, open?: [boolean, boolean], support?: [boolean, boolean]): string => {
  const l = lo === null || lo === undefined ? (support?.[0] ? 'support' : open?.[0] ? 'unbounded' : '—') : fmtParam(k, lo)
  const h = hi === null || hi === undefined ? (support?.[1] ? 'support' : open?.[1] ? 'unbounded' : '—') : fmtParam(k, hi)
  return `${l} – ${h}`
}

export function Tag({ kind, children }: { kind: 'measured' | 'inferred' | 'paper' | 'prototype' | 'verified' | 'derived'; children?: ReactNode }) {
  const cls = kind === 'inferred' ? 'tag--inferred' : kind === 'measured' ? 'tag--measured' : kind === 'prototype' ? 'tag--prototype' : kind === 'verified' || kind === 'derived' ? 'tag--verified' : 'tag--paper'
  return <span className={`tag ${cls}`}>{children ?? kind}</span>
}

/** Log-scaled numeric X axis with 1-2-5 ticks. */
export function logTicks(lo: number, hi: number): number[] {
  const out: number[] = []
  const e0 = Math.floor(Math.log10(lo))
  const e1 = Math.ceil(Math.log10(hi))
  for (let e = e0; e <= e1; e++) for (const m of [1, 2, 5]) {
    const v = m * Math.pow(10, e)
    if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v)
  }
  return out
}

export const XLog = ({ dataKey, label, lo, hi, fmt }: { dataKey: string; label: string; lo: number; hi: number; fmt?: (v: number) => string }) => (
  <XAxis
    type="number"
    dataKey={dataKey}
    scale="log"
    domain={[lo, hi]}
    ticks={logTicks(lo, hi)}
    tickFormatter={fmt ?? ((v: number) => sci(v, 0))}
    tick={{ fontSize: 11, fill: '#9698a8' }}
    tickLine={false}
    axisLine={{ stroke: '#c6c9d9' }}
    label={{ value: label, position: 'insideBottomRight', offset: -4, fontSize: 11, fill: '#6e7180' }}
    height={36}
  />
)

export function Table({ head, rows, className }: { head: ReactNode[]; rows: ReactNode[][]; className?: string }) {
  return (
    <div className={`table-wrap table-wrap--auto ${className ?? ''}`}>
      <table className="data data--static">
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
