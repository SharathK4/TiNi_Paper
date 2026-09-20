/**
 * The same reading, across every file in the set.
 *
 * Two questions: where does the transition sit and how much does it move, and
 * is anything here not like the others. The second is the one worth the
 * screen space — a file that differs is either a discovery or a mistake, and
 * each is named with the measurement that sets it apart.
 */

import { useMemo } from 'react'
import type { ChorusResult, EnsembleProgress, EnsembleResult } from '../../api/inflect'
import { Chorus } from './Chorus'
import { SparkGrid } from './Export'

function histogram(values: number[], bins = 28) {
  const v = values.filter((n) => isFinite(n))
  if (v.length < 2) return { bars: [], lo: 0, hi: 1 }
  const lo = Math.min(...v)
  const hi = Math.max(...v)
  const span = hi - lo || 1
  const counts = new Array(bins).fill(0)
  for (const n of v) counts[Math.min(bins - 1, Math.floor(((n - lo) / span) * bins))] += 1
  const peak = Math.max(...counts, 1)
  return { bars: counts.map((c, i) => ({ c, f: c / peak, x: lo + ((i + 0.5) / bins) * span })), lo, hi }
}

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2)
  return v.toPrecision(digits).replace(/\.?0+$/, '')
}

export function Ensemble({
  result,
  chorus,
  progress,
  unitX,
  unitY,
  onRun,
  onOpenFile,
  current,
}: {
  result: EnsembleResult | null
  chorus: ChorusResult | null
  progress: EnsembleProgress | null
  unitX: string | null
  unitY: string | null
  onRun: () => void
  onOpenFile: (name: string) => void
  current: string | null
}) {
  const running = Boolean(progress?.running)
  const positions = useMemo(
    () => (result ? result.curves.filter((c) => c.ok && c.transition_x !== null).map((c) => c.transition_x as number) : []),
    [result],
  )
  const hist = useMemo(() => histogram(positions), [positions])
  const outlierNames = useMemo(() => new Set((result?.outliers ?? []).map((o) => o.name)), [result])

  if (!result) {
    return (
      <section className="ens">
        <h2 className="ens__title">Every file</h2>
        <p className="ens__lead">
          Read all of them the same way, and see where the transition sits — and whether anything in the set does not belong.
        </p>
        <button className="btn btn--primary" onClick={onRun} disabled={running}>
          {running ? `Reading ${progress?.done ?? 0} of ${progress?.total ?? 0}…` : 'Read every file'}
        </button>
        {running && (
          <div className="ens__bar"><span style={{ width: `${progress?.percent ?? 0}%` }} /></div>
        )}
        {progress?.status === 'failed' && <p className="intake__error">{progress.message}</p>}
      </section>
    )
  }

  const tx = result.stats.transition_x
  const onset = result.stats.transition_onset
  const step = result.stats.transition_step

  return (
    <section className="ens">
      <h2 className="ens__title">Every file</h2>
      <p className="ens__headline">{result.headline}</p>

      {chorus && <Chorus data={chorus} />}

      <div className="ens__figures">
        <figure className="ens__fig">
          <figcaption>where the transition happens</figcaption>
          <svg viewBox="0 0 320 96" className="ens__hist" preserveAspectRatio="none">
            {hist.bars.map((b, i) => (
              <rect
                key={i}
                x={(i / hist.bars.length) * 320}
                width={320 / hist.bars.length - 1.2}
                y={80 - b.f * 72}
                height={Math.max(0.6, b.f * 72)}
              />
            ))}
            <line x1={0} x2={320} y1={80} y2={80} className="ens__axis" />
          </svg>
          <div className="ens__scale num">
            <span>{fmt(hist.lo)}</span>
            <span>{fmt(hist.hi)}</span>
          </div>
          <p className="ens__stat num">
            {fmt(tx?.median)} <small>± {fmt(tx?.spread, 2)} {unitX ?? ''}</small>
          </p>
        </figure>

        <div className="ens__numbers">
          <div className="ens__row">
            <span>files read</span>
            <b className="num">{result.n_ok} of {result.n_files}</b>
          </div>
          <div className="ens__row">
            <span>with a transition</span>
            <b className="num">{tx?.n ?? 0}</b>
          </div>
          <div className="ens__row">
            <span>value at the transition</span>
            <b className="num">{fmt(onset?.median)} <small>± {fmt(onset?.spread, 2)} {unitY ?? ''}</small></b>
          </div>
          <div className="ens__row">
            <span>size of the step</span>
            <b className="num">{fmt(step?.median)} <small>± {fmt(step?.spread, 2)} {unitY ?? ''}</small></b>
          </div>
          {result.groups.map((g) => (
            <div className="ens__row" key={g.signature}>
              <span>{g.signature}</span>
              <b className="num">{g.count}</b>
            </div>
          ))}
          {result.n_failed > 0 && (
            <div className="ens__row ens__row--warn">
              <span>could not be read</span>
              <b className="num">{result.n_failed}</b>
            </div>
          )}
        </div>
      </div>

      {result.outliers.length > 0 ? (
        <div className="ens__outliers">
          <h3>Not like the others</h3>
          {result.outliers.map((o) => (
            <button key={o.name} className="outlier" onClick={() => onOpenFile(o.name)}>
              <span className="outlier__name">{o.name}</span>
              <span className="outlier__why">{o.sentence}</span>
            </button>
          ))}
        </div>
      ) : (
        <p className="ens__none">Nothing in this set stands apart from the rest.</p>
      )}

      <div className="ens__files">
        <h3>Open a file</h3>
        <SparkGrid
          sparklines={result.sparklines ?? {}}
          names={result.curves.slice(0, 400).map((c) => c.name)}
          current={current}
          odd={outlierNames}
          bad={new Set(result.curves.filter((c) => !c.ok).map((c) => c.name))}
          onOpen={onOpenFile}
        />
      </div>
    </section>
  )
}
