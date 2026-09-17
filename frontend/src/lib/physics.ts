/**
 * The cubic landscape, evaluated client-side so the sliders answer instantly.
 * This mirrors backend/tini/potential.py exactly; the backend value is still
 * fetched (debounced) and the two are compared in the UI as a live check.
 *
 *   U0(ξ) = U‡/2 + (3U‡/2ξ‡)(ξ − ξ‡/2) − (2U‡/ξ‡³)(ξ − ξ‡/2)³      (paper)
 *   F(ξ)  = −dU0/dξ = −3U‡/2ξ‡ + (6U‡/ξ‡³)(ξ − ξ‡/2)²            (derived)
 *   f_c   = 3U‡/2ξ‡                                                (paper)
 *   U‡(f) = U‡ (1 − f/f_c)^{3/2}                                   (paper Eq. 4)
 */

export const KBT = 1.380649e-23 * 300 * 1e22 // pN·Å at 300 K = 41.42

export type Landscape = { U: number; xi: number }

export const fc = ({ U, xi }: Landscape) => (3 * U) / (2 * xi)

export const U0 = ({ U, xi }: Landscape, x: number) => {
  const s = x - xi / 2
  return U / 2 + ((3 * U) / (2 * xi)) * s - ((2 * U) / xi ** 3) * s ** 3
}

export const force = ({ U, xi }: Landscape, x: number) => {
  const s = x - xi / 2
  return -(3 * U) / (2 * xi) + ((6 * U) / xi ** 3) * s ** 2
}

export const barrier = (l: Landscape, f: number) => {
  const r = 1 - f / fc(l)
  return r <= 0 ? 0 : l.U * Math.pow(r, 1.5)
}

export const stationary = (l: Landscape, f: number) => {
  const r = 1 - f / fc(l)
  if (r < 0) return null
  const h = (l.xi / 2) * Math.sqrt(r)
  return { xiMin: l.xi / 2 - h, xiTs: l.xi / 2 + h }
}

export function grid(l: Landscape, n = 241, pad = 0.35) {
  const lo = -pad * l.xi
  const hi = (1 + pad) * l.xi
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
}

export function curves(l: Landscape, f: number, n = 241) {
  const xs = grid(l, n)
  return xs.map((x) => {
    const u0 = U0(l, x)
    return { xi: x, U0: u0, Uf: u0 - f * x, F: force(l, x), Ff: force(l, x) + f }
  })
}
