/**
 * Version 4 — Inflect.
 *
 * One page with two states: before there is data, and after. After, the chart
 * is the stage and the analysis plays over it as a sequence of beats. The
 * reader moves with the dots, the arrow keys, or by changing a number in the
 * sentence they are reading.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  inflect,
  type CurveQuery,
  type CurveResult,
  type EnsembleProgress,
  type EnsembleResult,
  type SessionDetail,
} from '../../api/inflect'
import { BeatRail, BeatStats, Sentence, useBeatKeys } from './Narration'
import { Ensemble } from './Ensemble'
import { Intake } from './Intake'
import { Stage } from './Stage'

type Controls = { n_segments?: number; penalty_scale: number; detect_transitions: boolean }

export function InflectPage() {
  const [sampleAvailable, setSampleAvailable] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [file, setFile] = useState<string | null>(null)
  const [curve, setCurve] = useState<CurveResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [index, setIndex] = useState(0)
  const [controls, setControls] = useState<Controls>({ penalty_scale: 1, detect_transitions: true })
  const [ensemble, setEnsemble] = useState<EnsembleResult | null>(null)
  const [ensProgress, setEnsProgress] = useState<EnsembleProgress | null>(null)
  const poll = useRef<number | null>(null)

  useEffect(() => {
    inflect.spec().then((s) => setSampleAvailable(Boolean(s.sample_available))).catch(() => setSampleAvailable(false))
  }, [])

  // --- loading a curve ----------------------------------------------------
  const load = useCallback(
    async (id: string, q: CurveQuery) => {
      setLoading(true)
      setError(null)
      try {
        const c = await inflect.curve(id, q)
        setCurve(c)
        setFile(c.file)
        setIndex((i) => Math.min(i, Math.max(0, c.reading.beats.length - 1)))
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  const open = useCallback(
    async (id: string, d: SessionDetail) => {
      setSessionId(id)
      setDetail(d)
      setIndex(0)
      // A strain column in absolute units reads better as a percentage, and
      // the units are worth carrying into every sentence.
      const cols = d.columns
      const x = cols.recommended_x !== null ? cols.columns[cols.recommended_x] : null
      if (x && x.kind === 'strain' && !x.unit && Math.abs(x.maximum) <= 1.5) {
        await inflect.setConfig(id, { x_scale: 100, x_unit: '%' })
      }
      await load(id, {})
      // The session already knows whether a comparison exists; asking for one
      // that does not would answer 404 and leave an error in the console for
      // a case that is entirely ordinary.
      setEnsemble(d.has_ensemble ? await inflect.ensemble(id).catch(() => null) : null)
    },
    [load],
  )

  const reload = useCallback(
    (next: Partial<Controls> & { file?: string }) => {
      if (!sessionId) return
      const merged = { ...controls, ...next }
      setControls({ n_segments: merged.n_segments, penalty_scale: merged.penalty_scale, detect_transitions: merged.detect_transitions })
      load(sessionId, {
        file: next.file ?? file ?? undefined,
        n_segments: merged.n_segments,
        penalty_scale: merged.penalty_scale,
        detect_transitions: merged.detect_transitions,
      })
    },
    [sessionId, controls, file, load],
  )

  const onControl = useCallback(
    (control: string, value: number) => {
      if (control === 'n_segments') reload({ n_segments: Math.round(value) })
    },
    [reload],
  )

  // --- the ensemble run ---------------------------------------------------
  const runEnsemble = useCallback(async () => {
    if (!sessionId) return
    setEnsProgress(await inflect.runEnsemble(sessionId))
    const tick = async () => {
      const p = await inflect.ensembleProgress()
      setEnsProgress(p)
      if (p.running) {
        poll.current = window.setTimeout(tick, 400)
      } else if (p.status === 'done') {
        try {
          setEnsemble(await inflect.ensemble(sessionId))
        } catch {
          /* the report may be missing if the run was cancelled */
        }
      }
    }
    poll.current = window.setTimeout(tick, 400)
  }, [sessionId])

  useEffect(() => () => { if (poll.current) window.clearTimeout(poll.current) }, [])

  const beats = curve?.reading.beats ?? []
  const beat = beats[Math.min(index, beats.length - 1)]
  useBeatKeys(beats.length, index, setIndex)

  const annotations = useMemo(() => beat?.annotations ?? [], [beat])

  if (!sessionId || !detail) {
    return (
      <>
        <Masthead />
        <Intake onOpen={open} sampleAvailable={sampleAvailable} />
        <Footnote />
      </>
    )
  }

  return (
    <>
      <Masthead headline={curve?.reading.headline} />

      {error && <p className="intake__error">{error}</p>}

      {curve && beat && (
        <section className={`reading ${loading ? 'is-loading' : ''}`}>
          <header className="reading__head">
            <div>
              <h2 className="reading__file">{curve.file}</h2>
              <p className="reading__shape">{curve.segmentation.shape.replace(/->/g, '→')}</p>
            </div>
            <div className="reading__tools">
              <button
                className={`btn btn--quiet ${controls.detect_transitions ? 'is-on' : ''}`}
                onClick={() => reload({ detect_transitions: !controls.detect_transitions })}
                title="Find transitions before splitting the curve into regimes"
              >
                {controls.detect_transitions ? 'finding transitions' : 'transitions off'}
              </button>
              {controls.n_segments !== undefined && (
                <button className="btn btn--quiet" onClick={() => reload({ n_segments: undefined })}>
                  let it choose
                </button>
              )}
              <button className="btn btn--quiet" onClick={() => { setSessionId(null); setCurve(null); setEnsemble(null) }}>
                new data
              </button>
            </div>
          </header>

          <Stage
            curve={curve}
            annotations={annotations}
            focus={beat.focus ?? null}
            beatKey={`${curve.file}-${beat.id}-${index}`}
          />

          <div className="narration">
            <BeatRail beats={beats} index={index} onSelect={setIndex} />
            <h3 className="narration__title">{beat.title}</h3>
            <Sentence beat={beat} onChange={onControl} />
            <BeatStats beat={beat} />
            <div className="narration__nav">
              <button className="btn btn--quiet" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
                ← back
              </button>
              <span className="narration__count num">{index + 1} / {beats.length}</span>
              <button
                className="btn btn--primary"
                onClick={() => setIndex(Math.min(beats.length - 1, index + 1))}
                disabled={index >= beats.length - 1}
              >
                next →
              </button>
            </div>
          </div>
        </section>
      )}

      <Ensemble
        result={ensemble}
        progress={ensProgress}
        unitX={curve?.x_unit ?? null}
        unitY={curve?.y_unit ?? null}
        onRun={runEnsemble}
        onOpenFile={(name) => { setIndex(0); reload({ file: name }) }}
        current={file}
      />

      <Footnote />
    </>
  )
}

function Masthead({ headline }: { headline?: string }) {
  return (
    <header className="masthead masthead--inflect">
      <h1 className="wordmark">Inflect</h1>
      <p className="standfirst">
        Drop in a measurement and it is read back to you: where the curve changes its mind, what each stretch of it is
        worth, and whether anything in the set does not belong.
        {headline && <> <b>{headline}.</b></>}
      </p>
      <p className="byline">Version 4 · curve analysis for any table of numbers</p>
    </header>
  )
}

function Footnote() {
  return (
    <p className="footnote">
      Transitions are found before regimes are fitted, and the value reported at a transition is the one at its onset,
      not the largest reached. Analysis: <code>backend/tini/inflect</code>.
    </p>
  )
}
