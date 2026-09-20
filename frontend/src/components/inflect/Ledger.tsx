/**
 * The ledger: every finding, inspectable, with who is responsible for it.
 *
 * Three tiers are kept apart on the page as they are in the record:
 *
 *   measured   a number from a stated procedure — the software's arithmetic
 *   read       a judgement by a stated rule — the software's reading, which
 *              the researcher may reject by rejecting the rule
 *   concluded  the researcher's own interpretation, written here, set in a
 *              different voice, and never mixed with the two above
 *
 * Each row opens into the evidence behind it: the method or rule in words,
 * the replicates that made its interval, and the sweep that tested it.
 */

import { useMemo, useState } from 'react'
import type { Finding, FindingInterval, FindingSensitivity, Ledger as LedgerDoc, Note } from '../../api/inflect'

function show(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !isFinite(v)) return '—'
  const a = Math.abs(v)
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : v.toPrecision(digits).replace(/\.?0+$/, '')
}

/** The replicates as a strip: where they fell, and where the estimate is. */
function Replicates({ interval }: { interval: FindingInterval }) {
  const s = interval.samples ?? []
  if (!s.length || !interval.ci) return null
  const lo = Math.min(...s, interval.value)
  const hi = Math.max(...s, interval.value)
  const span = hi - lo || 1
  const px = (v: number) => ((v - lo) / span) * 100
  return (
    <figure className="reps">
      <svg viewBox="0 0 100 18" preserveAspectRatio="none" className="reps__svg">
        <rect className="reps__ci" x={px(interval.ci[0])} width={Math.max(0.4, px(interval.ci[1]) - px(interval.ci[0]))} y={4} height={10} />
        {s.map((v, i) => (
          <line key={i} className="reps__tick" x1={px(v)} x2={px(v)} y1={5} y2={13} />
        ))}
        <line className="reps__point" x1={px(interval.value)} x2={px(interval.value)} y1={1} y2={17} />
      </svg>
      <figcaption className="reps__cap num">
        <span>{show(lo)}</span>
        <span>{interval.n} resamples · 95 % band</span>
        <span>{show(hi)}</span>
      </figcaption>
    </figure>
  )
}

/** The sweep as a strip: one mark per setting. */
function Sweep({ s }: { s: FindingSensitivity }) {
  const vals = s.points.map((p) => p.value).filter((v): v is number => v !== null && isFinite(v))
  if (!vals.length) return null
  const lo = Math.min(...vals, s.value)
  const hi = Math.max(...vals, s.value)
  const pad = Math.max((hi - lo) * 0.5, Math.abs(s.value) * 0.003, 1e-9)
  const top = hi + pad
  const bottom = lo - pad
  const W = 100
  const H = 22
  const px = (i: number) => (i / Math.max(1, s.points.length - 1)) * W
  const py = (v: number) => H - ((v - bottom) / (top - bottom || 1)) * H
  return (
    <figure className="sweepmini">
      <svg viewBox={`-2 -3 ${W + 4} ${H + 6}`} preserveAspectRatio="none" className="sweepmini__svg">
        <line className="sweepmini__ref" x1={-2} x2={W + 2} y1={py(s.value)} y2={py(s.value)} />
        {s.points.map((p, i) =>
          p.value !== null && isFinite(p.value) ? (
            <circle key={i} cx={px(i)} cy={py(p.value)} r={1.5} className="sweepmini__dot">
              <title>{`${p.label}: ${show(p.value)}`}</title>
            </circle>
          ) : (
            <line key={i} className="sweepmini__miss" x1={px(i)} x2={px(i)} y1={H - 2} y2={H + 2}>
              <title>{`${p.label}: not found`}</title>
            </line>
          ),
        )}
      </svg>
      <figcaption className="reps__cap">
        <span>{s.present_in} of {s.of} settings</span>
        <span>{s.holds ? 'holds' : `moves by ${show(s.spread, 3)}`}</span>
      </figcaption>
    </figure>
  )
}

function Tier({ tier }: { tier: Finding['tier'] | 'concluded' }) {
  const text = tier === 'measured' ? 'measured' : tier === 'read' ? 'the software’s reading' : 'your conclusion'
  return <span className={`tier tier--${tier}`}>{text}</span>
}

function Row({
  f,
  notes,
  open,
  onToggle,
  onShow,
  onNote,
  onRemoveNote,
  busy,
}: {
  f: Finding
  notes: Note[]
  open: boolean
  onToggle: () => void
  onShow: () => void
  onNote: (text: string) => Promise<void>
  onRemoveNote: (id: string) => Promise<void>
  busy: boolean
}) {
  const [draft, setDraft] = useState('')
  const ci = f.interval?.ci
  const sens = f.sensitivity
  const value = f.value !== null ? `${show(f.value)}${f.unit ? ` ${f.unit}` : ''}` : (f.text ?? '—')

  return (
    <>
      <tr className={`ledger__row ${open ? 'is-open' : ''}`} onClick={onToggle}>
        <td><Tier tier={f.tier} /></td>
        <td className="ledger__label">
          {f.label}
          {notes.length > 0 && <span className="ledger__hasnote" title={`${notes.length} note(s)`}>{notes.length}</span>}
        </td>
        <td className="num ledger__value">{value}</td>
        <td className="num ledger__ci">
          {ci ? `${show(ci[0])} – ${show(ci[1])}` : f.tier === 'measured' ? (f.id === 'noise' ? 'input to the bootstrap' : '—') : ''}
          {f.interval?.biased && <span className="ledger__flag" title="the replicates sit to one side of the estimate">shifted</span>}
        </td>
        <td className="ledger__sens">
          {sens ? (
            <span className={sens.holds ? 'holds' : 'moves'}>{sens.holds ? 'holds' : `moves ${show(sens.spread, 2)}`}</span>
          ) : ''}
        </td>
        <td className="ledger__act">
          <button className="btn btn--quiet btn--tiny" onClick={(e) => { e.stopPropagation(); onShow() }}>show</button>
        </td>
      </tr>
      {open && (
        <tr className="ledger__detail">
          <td colSpan={6}>
            <div className="detail">
              <div className="detail__col">
                <h4>{f.tier === 'read' ? 'The rule' : 'The method'}</h4>
                <p>{f.tier === 'read' && f.rule ? f.rule : f.method}</p>
                {f.tier === 'read' && <p className="detail__method">Procedure: {f.method}</p>}
                {Object.keys(f.evidence).length > 0 && (
                  <dl className="detail__evidence">
                    {Object.entries(f.evidence).map(([k, v]) => (
                      <div key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd className="num">{Array.isArray(v) ? v.map((n) => show(n as number, 5)).join(' – ') : typeof v === 'number' ? show(v, 5) : String(v)}</dd></div>
                    ))}
                  </dl>
                )}
              </div>
              <div className="detail__col">
                {f.interval?.ci && (
                  <>
                    <h4>How sure</h4>
                    <Replicates interval={f.interval} />
                    {f.interval.biased && (
                      <p className="detail__note">
                        The replicates sit {f.interval.shift! > 0 ? 'above' : 'below'} the estimate by about {show(Math.abs(f.interval.shift!), 2)}. The estimate is at the resolution of the method; added noise can only move it one way.
                      </p>
                    )}
                  </>
                )}
                {sens && (
                  <>
                    <h4>Under other settings</h4>
                    <Sweep s={sens} />
                  </>
                )}
              </div>
              <div className="detail__col detail__col--notes">
                <h4>What you conclude</h4>
                {notes.map((n) => (
                  <blockquote key={n.id} className="conclusion">
                    <p>{n.text}</p>
                    <footer>
                      <span>{n.created_at_iso.replace('T', ' ')}</span>
                      <button className="btn btn--quiet btn--tiny" disabled={busy} onClick={() => onRemoveNote(n.id)}>remove</button>
                    </footer>
                  </blockquote>
                ))}
                <textarea
                  className="conclusion__input"
                  placeholder="Your interpretation of this finding — kept apart from what the software measured, and exported under its own heading."
                  value={draft}
                  rows={3}
                  onChange={(e) => setDraft(e.target.value)}
                />
                <button
                  className="btn btn--primary btn--tiny"
                  disabled={busy || !draft.trim()}
                  onClick={async () => { await onNote(draft); setDraft('') }}
                >
                  keep this
                </button>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export function Ledger({
  ledger,
  loading,
  onShow,
  onNote,
  onRemoveNote,
  busy,
}: {
  ledger: LedgerDoc | null
  loading: boolean
  onShow: (f: Finding) => void
  onNote: (finding: Finding, text: string) => Promise<void>
  onRemoveNote: (noteId: string) => Promise<void>
  busy: boolean
}) {
  const [open, setOpen] = useState<string | null>(null)
  const [tier, setTier] = useState<'all' | 'measured' | 'read' | 'noted'>('all')
  const byFinding = useMemo(() => {
    const m = new Map<string, Note[]>()
    for (const n of ledger?.notes ?? []) m.set(n.finding_id, [...(m.get(n.finding_id) ?? []), n])
    return m
  }, [ledger])

  if (!ledger) {
    return (
      <section className="ledger">
        <h2 className="ens__title">Every finding</h2>
        <p className="ens__lead">{loading ? 'Measuring, resampling and sweeping…' : 'Open a file to see its findings.'}</p>
      </section>
    )
  }

  const s = ledger.summary
  const rows = ledger.findings.filter((f) =>
    tier === 'all' ? true : tier === 'noted' ? byFinding.has(f.id) : f.tier === tier)

  return (
    <section className="ledger">
      <h2 className="ens__title">Every finding</h2>
      <p className="ens__lead">
        <b className="num">{s.n}</b> findings for this file: <b className="num">{s.tiers.measured ?? 0}</b> measured by a stated procedure,{' '}
        <b className="num">{s.tiers.read ?? 0}</b> read by a stated rule. <b className="num">{s.with_interval}</b> carry a bootstrap interval;{' '}
        <b className="num">{s.holds}</b> of the <b className="num">{s.swept}</b> that were swept hold across every setting.
        {ledger.notes.length > 0 && <> You have concluded <b className="num">{ledger.notes.length}</b> thing{ledger.notes.length === 1 ? '' : 's'} from them.</>}
      </p>

      <div className="ledger__filters" role="tablist">
        {(['all', 'measured', 'read', 'noted'] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tier === t} className={tier === t ? 'is-on' : ''} onClick={() => setTier(t)}>
            {t === 'noted' ? 'with your conclusions' : t === 'read' ? 'the software’s reading' : t}
          </button>
        ))}
      </div>

      <div className="ledger__wrap">
      <table className="ledger__table">
        <thead>
          <tr>
            <th>who</th>
            <th>finding</th>
            <th>value</th>
            <th>95 % interval</th>
            <th>across settings</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => (
            <Row
              key={f.id}
              f={f}
              notes={byFinding.get(f.id) ?? []}
              open={open === f.id}
              onToggle={() => setOpen(open === f.id ? null : f.id)}
              onShow={() => onShow(f)}
              onNote={(text) => onNote(f, text)}
              onRemoveNote={onRemoveNote}
              busy={busy}
            />
          ))}
        </tbody>
      </table>
      </div>
    </section>
  )
}
