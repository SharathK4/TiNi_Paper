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
  inflectColumns,
  inflectLedger,
  inflectMore,
  type Annotation,
  type Beat,
  type Finding,
  type Ledger as LedgerDoc,
  type Preview,
  type ChorusResult,
  type CurveQuery,
  type CurveResult,
  type EnsembleProgress,
  type EnsembleResult,
  type SessionDetail,
  type Stability,
  type Uncertainty,
} from '../../api/inflect'
import { BeatRail, BeatStats, Sentence, useBeatKeys } from './Narration'
import { Confidence } from './Confidence'
import { Export } from './Export'
import { Columns } from './Columns'
import { Ledger } from './Ledger'
import { Record } from './Record'
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
  const [chorus, setChorus] = useState<ChorusResult | null>(null)
  const [uncertainty, setUncertainty] = useState<Uncertainty | null>(null)
  const [stability, setStability] = useState<Stability | null>(null)
  const [checking, setChecking] = useState(false)
  const [ledger, setLedger] = useState<LedgerDoc | null>(null)
  const [ledgerLoading, setLedgerLoading] = useState(false)
  const [noteBusy, setNoteBusy] = useState(false)
  const [inspect, setInspect] = useState<Finding | null>(null)
  // Choosing the columns sits between reading the file and reading the curve.
  const [preview, setPreview] = useState<Preview | null>(null)
  const [choosing, setChoosing] = useState(false)
  // The first ledger for a file under the settings it opened with: what a
  // change of setting is compared against.
  const baseline = useRef<{ file: string; ledger: LedgerDoc } | null>(null)
  const poll = useRef<number | null>(null)
  const askedFor = useRef<string>('')
  const ledgerFor = useRef<string>('')

  useEffect(() => {
    inflect.spec().then((s) => setSampleAvailable(Boolean(s.sample_available))).catch(() => setSampleAvailable(false))
  }, [])

  // --- loading a curve ----------------------------------------------------
  const load = useCallback(
    async (id: string, q: CurveQuery) => {
      setLoading(true)
      setError(null)
      setUncertainty(null)
      setStability(null)
      setInspect(null)
      try {
        const c = await inflect.curve(id, q)
        setCurve(c)
        setFile(c.file)
        setIndex((i) => Math.min(i, Math.max(0, c.reading.beats.length - 1)))
        // The ledger follows: it resamples and sweeps, so it lands a second
        // or two after the curve rather than holding the curve back.
        const key = `${id}|${c.file}|${q.n_segments ?? 'auto'}|${q.penalty_scale ?? 1}|${q.detect_transitions ?? true}`
        ledgerFor.current = key
        setLedgerLoading(true)
        inflectLedger
          .ledger(id, { file: c.file, n_segments: q.n_segments, penalty_scale: q.penalty_scale, detect_transitions: q.detect_transitions })
          .then((l) => {
            if (ledgerFor.current !== key) return
            setLedger(l)
            const isDefault = q.n_segments === undefined && (q.penalty_scale ?? 1) === 1 && (q.detect_transitions ?? true)
            if (isDefault || baseline.current?.file !== c.file) baseline.current = { file: c.file, ledger: l }
          })
          .catch(() => { if (ledgerFor.current === key) setLedger(null) })
          .finally(() => { if (ledgerFor.current === key) setLedgerLoading(false) })
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
      setCurve(null)
      setLedger(null)
      setError(null)
      // A strain column in absolute units reads better as a percentage, and
      // the units are worth carrying into every sentence.
      const cols = d.columns
      const x = cols.recommended_x !== null ? cols.columns[cols.recommended_x] : null
      if (x && x.kind === 'strain' && !x.unit && Math.abs(x.maximum) <= 1.5) {
        await inflect.setConfig(id, { x_scale: 100, x_unit: '%' })
      }
      // Stop here and let the reader choose what to plot. The tool's own pick
      // is pre-selected; it is a suggestion, not a decision taken for them.
      setChoosing(true)
      try {
        setPreview(await inflectColumns.preview(id))
      } catch (e) {
        // No preview is no reason to be stuck: fall through to the tool's pick.
        setPreview(null)
        setChoosing(false)
        await load(id, {})
      }
      if (d.has_ensemble) {
        setEnsemble(await inflect.ensemble(id).catch(() => null))
        setChorus(await inflectMore.chorus(id).catch(() => null))
      } else {
        setEnsemble(null)
        setChorus(null)
      }
    },
    [load],
  )

  /** The reader has picked a pair: remember it and read the curve. */
  const chooseColumns = useCallback(
    async (xi: number, yi: number) => {
      if (!sessionId) return
      setLoading(true)
      try {
        // Units belong to the column, so a pick that changes the column
        // clears a unit that was set for a different one.
        const picked = preview?.columns.find((c) => c.index === yi)
        await inflect.setConfig(sessionId, {
          x_column: xi,
          y_column: yi,
          y_unit: picked?.unit ?? null,
        })
        setChoosing(false)
        setIndex(0)
        await load(sessionId, { x_column: xi, y_column: yi })
      } finally {
        setLoading(false)
      }
    },
    [sessionId, preview, load],
  )

  /** Go back to the chooser from the reading. */
  const changeColumns = useCallback(async () => {
    if (!sessionId) return
    // Cleared first: the chooser reads the current pair when it mounts, and
    // mounting it against the previous answer would show the suggestion
    // again rather than what the reader chose last time.
    setPreview(null)
    setChoosing(true)
    try {
      setPreview(await inflectColumns.preview(sessionId, file ?? undefined))
    } catch (e) {
      setError((e as Error).message)
      setChoosing(false)
    }
  }, [sessionId, file])

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
          setChorus(await inflectMore.chorus(sessionId).catch(() => null))
        } catch {
          /* the report may be missing if the run was cancelled */
        }
      }
    }
    poll.current = window.setTimeout(tick, 400)
  }, [sessionId])

  useEffect(() => () => { if (poll.current) window.clearTimeout(poll.current) }, [])

  const addNote = useCallback(
    async (f: Finding, text: string) => {
      if (!sessionId || !curve) return
      setNoteBusy(true)
      try {
        const n = await inflectLedger.addNote(sessionId, curve.file, f.id, text)
        setLedger((l) => (l ? { ...l, notes: [...l.notes, n] } : l))
      } finally {
        setNoteBusy(false)
      }
    },
    [sessionId, curve],
  )
  const removeNote = useCallback(
    async (noteId: string) => {
      if (!sessionId) return
      setNoteBusy(true)
      try {
        await inflectLedger.removeNote(sessionId, noteId)
        setLedger((l) => (l ? { ...l, notes: l.notes.filter((n) => n.id !== noteId) } : l))
      } finally {
        setNoteBusy(false)
      }
    },
    [sessionId],
  )

  // What a change of setting did, finding by finding, against the baseline.
  const delta = useMemo(() => {
    const b = baseline.current
    if (!ledger || !b || b.file !== ledger.file || b.ledger === ledger) return null
    const before = new Map(b.ledger.findings.map((f) => [f.id, f]))
    const moved: string[] = []
    const gone: string[] = []
    const fresh: string[] = []
    for (const f of ledger.findings) {
      const o = before.get(f.id)
      if (!o) { fresh.push(f.label); continue }
      const same = f.value === null || o.value === null
        ? (f.text ?? '') === (o.text ?? '')
        : Math.abs(f.value - o.value) <= 1e-9 * Math.max(1, Math.abs(o.value))
      if (!same) moved.push(f.label)
    }
    for (const f of b.ledger.findings) if (!ledger.findings.some((g) => g.id === f.id)) gone.push(f.label)
    const pos = ledger.findings.find((f) => f.id === 'transition.0.position')
    const posBefore = before.get('transition.0.position')
    const posHeld = pos && posBefore && pos.value !== null && posBefore.value !== null && Math.abs(pos.value - posBefore.value) < 1e-9
    return { moved, gone, fresh, posHeld: Boolean(posHeld), n: ledger.findings.length }
  }, [ledger])

  // The reading comes from the backend; the interval on it is asked for
  // separately, because resampling takes a second and the curve should be on
  // screen immediately. The beat is inserted where it belongs either way, and
  // fills in when the numbers arrive.
  const beats: Beat[] = useMemo(() => {
    const base = curve?.reading.beats ?? []
    if (!base.length) return base
    const t = curve?.reading.transition
    const confidence: Beat = {
      id: 'confidence',
      kind: 'confidence',
      title: 'How sure is this?',
      sentence: '',
      slots: [],
      stats: [],
      annotations:
        t && uncertainty?.transition?.position.ci
          ? [
              {
                kind: 'band',
                tone: 'accent',
                x0: uncertainty.transition.position.ci[0],
                x1: uncertainty.transition.position.ci[1],
                y0: uncertainty.transition.position.value,
                text: '95 % interval',
              },
            ]
          : [],
      focus: t ? [Math.max(curve!.domain[0], t.x_lo - t.width * 12 - 0.05),
                  Math.min(curve!.domain[1], t.x_hi + t.width * 12 + 0.05)] : undefined,
    }
    const last = base.map((b) => b.kind).lastIndexOf('transition')
    const at = last >= 0 ? last + 1 : base.length
    return [...base.slice(0, at), confidence, ...base.slice(at)]
  }, [curve, uncertainty])

  const beat = beats[Math.min(index, beats.length - 1)]
  useBeatKeys(beats.length, index, setIndex)

  // Resampling is asked for once the reader reaches the beat that needs it.
  useEffect(() => {
    if (!sessionId || !curve || beat?.kind !== 'confidence') return
    const key = `${sessionId}|${curve.file}|${controls.n_segments ?? 'auto'}|${controls.penalty_scale}|${controls.detect_transitions}`
    if (askedFor.current === key) return
    askedFor.current = key
    const q = {
      file: curve.file,
      n_segments: controls.n_segments,
      penalty_scale: controls.penalty_scale,
      detect_transitions: controls.detect_transitions,
    }
    setChecking(true)
    Promise.all([
      inflectMore.uncertainty(sessionId, q).catch(() => null),
      inflectMore.stability(sessionId, q).catch(() => null),
    ])
      .then(([u, st]) => {
        setUncertainty(u)
        setStability(st)
      })
      .finally(() => setChecking(false))
  }, [sessionId, curve, beat?.kind, controls])

  const annotations = useMemo<Annotation[]>(() => {
    if (!inspect) return beat?.annotations ?? []
    const a = inspect.anchor
    const tone: Annotation['tone'] = inspect.tier === 'read' ? 'model' : 'accent'
    const out: Annotation[] = []
    if (a.x !== undefined && a.y !== undefined) out.push({ kind: 'circle', tone, x0: a.x, y0: a.y, text: inspect.label })
    else if (a.x0 !== undefined && a.x1 !== undefined && a.y0 !== undefined && a.y1 !== undefined)
      out.push({ kind: 'slope', tone, x0: a.x0, x1: a.x1, y0: a.y0, y1: a.y1, text: inspect.label })
    else if (a.x0 !== undefined && a.x1 !== undefined) out.push({ kind: 'region', tone, x0: a.x0, x1: a.x1, text: inspect.label })
    if (inspect.interval?.ci && inspect.id.endsWith('.position'))
      out.push({ kind: 'band', tone: 'accent', x0: inspect.interval.ci[0], x1: inspect.interval.ci[1], y0: inspect.value ?? undefined, text: '95 % interval' })
    return out
  }, [beat, inspect])

  const inspectFocus = useMemo<[number, number] | null>(() => {
    if (!inspect || !curve) return null
    const a = inspect.anchor
    const span = curve.domain[1] - curve.domain[0]
    const lo = a.x0 ?? a.x ?? curve.domain[0]
    const hi = a.x1 ?? a.x ?? curve.domain[1]
    const pad = Math.max((hi - lo) * 0.3, span * 0.06)
    return [Math.max(curve.domain[0], lo - pad), Math.min(curve.domain[1], hi + pad)]
  }, [inspect, curve])

  useEffect(() => { setInspect(null) }, [index])

  const notesFor = useMemo(() => {
    if (!ledger || !beat) return []
    const ids = new Set(beat.findings ?? [])
    return ledger.notes.filter((n) => ids.has(n.finding_id))
  }, [ledger, beat])

  if (!sessionId || !detail) {
    return (
      <>
        <Masthead />
        <Intake onOpen={open} sampleAvailable={sampleAvailable} />
        <Footnote />
      </>
    )
  }

  if (choosing) {
    return (
      <>
        <Masthead />
        {error && <p className="intake__error">{error}</p>}
        {preview ? (
          <Columns
            key={`${preview.file}:${preview.chosen_x}:${preview.chosen_y}`}
            preview={preview}
            busy={loading}
            onConfirm={chooseColumns}
            onBack={curve ? () => setChoosing(false) : () => { setSessionId(null); setDetail(null); setPreview(null); setChoosing(false) }}
          />
        ) : (
          <section className="choose"><p className="conf__waiting">Looking at every column…</p></section>
        )}
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
              <p className="reading__shape" title={curve.segmentation.shape.replace(/->/g, '→')}>
                {(() => {
                  const parts = curve.segmentation.shape.split(' -> ')
                  if (parts.length <= 9) return curve.segmentation.shape.replace(/->/g, '→')
                  const n = curve.segmentation.n_segments
                  const t = curve.segmentation.boundaries.filter((b) => b.kind === 'transition').length
                  return `${parts[0]} → … → ${parts[parts.length - 1]} · ${n} regimes, ${t} transition${t === 1 ? '' : 's'}`
                })()}
              </p>
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
              <button className="btn btn--quiet" onClick={changeColumns}>
                change columns
              </button>
              <button className="btn btn--quiet" onClick={() => { setSessionId(null); setCurve(null); setEnsemble(null); setPreview(null) }}>
                new data
              </button>
            </div>
          </header>

          {delta && (delta.moved.length || delta.gone.length || delta.fresh.length) ? (
            <p className="delta">
              Against the reading this file opened with: <b className="num">{delta.moved.length}</b> of <b className="num">{delta.n}</b> findings moved
              {delta.gone.length ? <>, <b className="num">{delta.gone.length}</b> no longer found</> : null}
              {delta.fresh.length ? <>, <b className="num">{delta.fresh.length}</b> new</> : null}
              {delta.posHeld ? <>. The transition position did not move.</> : delta.moved.some((m) => m.startsWith('transition position')) ? <>. The transition position moved.</> : '.'}
            </p>
          ) : delta ? (
            <p className="delta">Against the reading this file opened with: nothing moved.</p>
          ) : null}

          <Stage
            curve={curve}
            annotations={annotations}
            focus={inspectFocus ?? beat.focus ?? null}
            beatKey={`${curve.file}-${beat.id}-${index}-${inspect?.id ?? ''}`}
          />
          {inspect && (
            <p className="inspecting">
              Showing <b>{inspect.label}</b>
              {inspect.value !== null ? <> = <span className="num">{inspect.value.toPrecision(4)}{inspect.unit ? ` ${inspect.unit}` : ''}</span></> : null}
              <button className="btn btn--quiet btn--tiny" onClick={() => setInspect(null)}>back to the reading</button>
            </p>
          )}

          <div className="narration">
            <BeatRail beats={beats} index={index} onSelect={setIndex} />
            <h3 className="narration__title">{beat.title}</h3>
            {beat.kind !== 'confidence' && (
              <p className={`tierline tierline--${beat.tier ?? 'measured'}`}>
                {beat.tier === 'read' ? 'the software\u2019s reading, by a stated rule' : 'measured by a stated procedure'}
              </p>
            )}
            {beat.kind === 'confidence' ? (
              <Confidence
                uncertainty={uncertainty}
                stability={stability}
                xUnit={curve.x_unit}
                yUnit={curve.y_unit}
                loading={checking}
              />
            ) : (
              <>
                <Sentence beat={beat} onChange={onControl} />
                <BeatStats beat={beat} />
              </>
            )}
            {notesFor.length > 0 && (
              <div className="conclusions">
                <p className="tierline tierline--concluded">your conclusion</p>
                {notesFor.map((n) => (
                  <blockquote key={n.id} className="conclusion conclusion--inline"><p>{n.text}</p></blockquote>
                ))}
              </div>
            )}
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

      {curve && (
        <Ledger
          ledger={ledger}
          loading={ledgerLoading}
          onShow={(f) => {
            setInspect(f)
            document.querySelector('.stage')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }}
          onNote={addNote}
          onRemoveNote={removeNote}
          busy={noteBusy}
        />
      )}

      <Ensemble
        result={ensemble}
        chorus={chorus}
        progress={ensProgress}
        unitX={curve?.x_unit ?? null}
        unitY={curve?.y_unit ?? null}
        onRun={runEnsemble}
        onOpenFile={(name) => { setIndex(0); reload({ file: name }) }}
        current={file}
      />

      <Export sessionId={sessionId} file={file} />
      <Record sessionId={sessionId} file={file} />

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
