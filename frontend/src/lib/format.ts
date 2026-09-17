export const fmt = (v: number | null | undefined, digits = 3): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(digits - 1)
  return v.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

export const fixed = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits)

export const secs = (s: number | null | undefined): string => {
  if (s === null || s === undefined) return '—'
  if (s < 60) return `${s.toFixed(1)} s`
  const m = Math.floor(s / 60)
  return `${m} min ${Math.round(s - m * 60)} s`
}

export const mb = (bytes: number): string => `${(bytes / 1e6).toFixed(2)} MB`

/** 2.5e-6 → "2.5×10⁻⁶" (unicode superscripts), digits = significant digits after the first. */
export const sci = (v: number | null | undefined, digits = 1): string => {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (v === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(v)))
  if (e >= -2 && e <= 3) return v.toLocaleString('en-US', { maximumSignificantDigits: digits + 2 })
  const m = v / Math.pow(10, e)
  const sup = String(e).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(d)])
  return `${m.toFixed(digits)}×10${sup}`
}

export const pval = (p: number | null | undefined): string => {
  if (p === null || p === undefined || !Number.isFinite(p)) return '—'
  if (p < 1e-4) return '< 10⁻⁴'
  if (p < 0.01) return p.toExponential(1).replace('e-', '×10⁻')
  return p.toFixed(2)
}

export const pct = (v: number | null | undefined, digits = 0): string => (v === null || v === undefined ? '—' : `${(100 * v).toFixed(digits)} %`)
