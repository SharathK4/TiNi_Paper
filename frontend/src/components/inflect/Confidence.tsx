/**
 * The beat that answers the two questions a referee asks first.
 *
 * *What is the error bar?* — the bootstrap interval, printed with the value
 * and drawn on the chart as a band around the transition.
 *
 * *Did you tune the settings until it said that?* — the stability sweep. Each
 * mark is the transition as re-derived under one setting; a column of marks
 * at the same height is a result that belongs to the data rather than to the
 * knobs, and is far more convincing than any single number.
 */

import type { Interval, Stability, Uncertainty } from '../../api/inflect'

function show(v: number | null | undefined, unit?: string | null, digits = 4): string {
  if (v === null || v === undefined || !isFinite(v)) return '—'
  const a = Math.abs(v)
  const text = a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : v.toPrecision(digits).replace(/\.?0+$/, '')
  return unit ? `${text} ${unit}` : text
}

function Pill({ label, interval, unit }: { label: string; interval: Interval; unit?: string | null }) {
  return (
    <div className="conf__pill">
      <div className="conf__label">{label}</div>
      <div className="conf__value num">{show(interval.value, unit)}</div>
      <div className="conf__ci num">
        {interval.ci ? `95 % ${show(interval.ci[0], null)} – ${show(interval.ci[1], null)}` : 'no interval'}
      </div>
    </div>
  )
}

/** The sweep as a strip: setting across, where the transition landed up. */
function Sweep({ stability, unit }: { stability: Stability; unit?: string | null }) {
  const found = stability.points.filter((p) => p.found && p.position !== null)
  if (!found.length) return null
  const values = found.map((p) => p.position as number)
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo
  // A flat result must look flat, not be stretched until its noise fills the
  // frame; the band is at least a tenth of the range either way.
  const pad = Math.max(span * 0.6, Math.abs(hi) * 0.004, 1e-9)
  const top = hi + pad
  const bottom = lo - pad
  const W = 100
  const H = 34
  const px = (i: number) => (i / Math.max(1, stability.points.length - 1)) * W
  const py = (v: number) => H - ((v - bottom) / (top - bottom || 1)) * H

  const groups: Record<string, string> = { threshold: 'detector', penalty: 'detail', regimes: 'regimes' }
  const seen = new Set<string>()

  return (
    <figure className="sweep">
      <svg viewBox={`-2 -6 ${W + 4} ${H + 14}`} className="sweep__svg" preserveAspectRatio="none">
        {stability.reference !== null && (
          <line className="sweep__ref" x1={-2} x2={W + 2} y1={py(stability.reference)} y2={py(stability.reference)} />
        )}
        {stability.points.map((p, i) =>
          p.found && p.position !== null ? (
            <circle key={i} cx={px(i)} cy={py(p.position)} r={p.is_default ? 2.1 : 1.4}
                    className={p.is_default ? 'sweep__dot is-default' : 'sweep__dot'}>
              <title>{`${p.label}: ${show(p.position, unit)}`}</title>
            </circle>
          ) : (
            <line key={i} className="sweep__miss" x1={px(i)} x2={px(i)} y1={H - 3} y2={H + 1}>
              <title>{`${p.label}: no transition found`}</title>
            </line>
          ),
        )}
      </svg>
      <figcaption className="sweep__legend">
        {stability.points.map((p, i) => {
          if (seen.has(p.setting)) return null
          seen.add(p.setting)
          return <span key={p.setting} style={{ left: `${px(i)}%` }}>{groups[p.setting]}</span>
        })}
      </figcaption>
    </figure>
  )
}

export function Confidence({
  uncertainty,
  stability,
  xUnit,
  yUnit,
  loading,
}: {
  uncertainty: Uncertainty | null
  stability: Stability | null
  xUnit: string | null
  yUnit: string | null
  loading: boolean
}) {
  if (loading && !uncertainty) {
    return <p className="conf__waiting">Resampling the measurement…</p>
  }
  if (!uncertainty) return null

  const t = uncertainty.transition
  const rate = uncertainty.detection_rate

  return (
    <div className="conf">
      {t ? (
        <>
          <p className="narration__sentence">
            Resampling the measurement {uncertainty.n_used} times puts the transition at{' '}
            <span className="num live live--static">{show(t.position.value, null)}</span>
            {xUnit ? ` ${xUnit}` : ''}, with 95 % of the resamples between{' '}
            <span className="num live live--static">{show(t.position.ci?.[0], null)}</span> and{' '}
            <span className="num live live--static">{show(t.position.ci?.[1], null)}</span>.
          </p>
          <div className="conf__pills">
            <Pill label="position" interval={t.position} unit={xUnit} />
            <Pill label="value at onset" interval={t.onset} unit={yUnit} />
            <Pill label="size of the step" interval={t.step} unit={yUnit} />
          </div>
        </>
      ) : (
        <p className="narration__sentence">
          There is no transition to put an interval on. The slopes below carry theirs.
        </p>
      )}

      <div className="conf__rows">
        {rate < 1 && (
          <p className="conf__row">
            The transition survives resampling {Math.round(rate * 100)} % of the time
            {rate < 0.9 ? ' — treat it as tentative.' : '.'}
          </p>
        )}
        {uncertainty.note && <p className="conf__row">{uncertainty.note}.</p>}
        {stability && <p className="conf__row conf__row--lead">{stability.sentence}</p>}
      </div>

      {stability && <Sweep stability={stability} unit={xUnit} />}

      {uncertainty.segments.length > 0 && (
        <table className="conf__table">
          <thead>
            <tr>
              <th>regime</th>
              <th>slope</th>
              <th>95 % interval</th>
            </tr>
          </thead>
          <tbody>
            {uncertainty.segments.map((s, i) => (
              <tr key={i}>
                <td>
                  {s.model} · {show(s.x0, null, 3)}–{show(s.x1, null, 3)}
                  {xUnit ? ` ${xUnit}` : ''}
                </td>
                <td className="num">{show(s.slope.value, null)}</td>
                <td className="num conf__ci">
                  {s.slope.ci ? `${show(s.slope.ci[0], null)} – ${show(s.slope.ci[1], null)}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
