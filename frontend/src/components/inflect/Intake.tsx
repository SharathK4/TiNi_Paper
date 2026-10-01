/**
 * Getting the data in, and showing the reader what was worked out from it.
 *
 * The findings appear one line at a time rather than all at once. That is not
 * decoration: each line is a judgement the tool has made on the reader's
 * behalf — which column is the axis, which is the response, and why — and
 * they are worth reading rather than skipping past.
 */

import { useCallback, useRef, useState } from 'react'
import { inflect, type ColumnReport, type SessionDetail } from '../../api/inflect'

type Phase = 'idle' | 'sending' | 'reading' | 'ready' | 'failed'

function findings(detail: SessionDetail): { text: string; detail?: string }[] {
  const d = detail.first_file.dialect
  const cols: ColumnReport = detail.columns
  const out: { text: string; detail?: string }[] = []
  const n = detail.files.length

  out.push({
    text: n === 1 ? `1 file, ${detail.first_file.n_rows.toLocaleString()} rows` : `${n} files, ${detail.first_file.n_rows.toLocaleString()} rows each`,
  })
  out.push({
    text: `${d.delimiter}-separated, ${d.n_fields} columns`,
    detail: d.comment_prefix ? `${d.n_comment_lines} comment line(s) beginning ${d.comment_prefix}` : 'no comment lines',
  })
  out.push({
    text: d.header_line
      ? d.header_from_comment
        ? 'column names taken from the header comment'
        : 'column names taken from the first line'
      : 'no header found — columns are numbered',
  })

  const x = cols.recommended_x !== null ? cols.columns[cols.recommended_x] : null
  const y = cols.recommended_y !== null ? cols.columns[cols.recommended_y] : null
  if (x) out.push({ text: `${x.label} is the controlled variable`, detail: x.x_reasons.join(' · ') })
  if (y) out.push({ text: `${y.label} is the response worth reading`, detail: y.y_reasons.join(' · ') })

  const dupes = cols.columns.filter((c) => c.duplicate_of)
  if (dupes.length) {
    out.push({
      text: `${dupes.length} column(s) repeat information already present`,
      detail: dupes.slice(0, 4).map((c) => `${c.name} = ${c.duplicate_of}`).join(' · '),
    })
  }
  const warn = detail.first_file.warnings
  if (warn.length) out.push({ text: warn[0] })
  return out
}

export function Intake({
  onOpen,
  sampleAvailable,
}: {
  onOpen: (sessionId: string, detail: SessionDetail) => void
  sampleAvailable: boolean
}) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [over, setOver] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [problems, setProblems] = useState<{ name: string; error: string }[]>([])
  const picker = useRef<HTMLInputElement>(null)

  const open = useCallback(async (id: string) => {
    setPhase('reading')
    try {
      const d = await inflect.session(id)
      setSessionId(id)
      setDetail(d)
      setPhase('ready')
    } catch (e) {
      setError((e as Error).message)
      setPhase('failed')
    }
  }, [])

  const send = useCallback(
    async (files: File[]) => {
      if (!files.length) return
      setPhase('sending')
      setError(null)
      setProgress(0)
      try {
        const r = await inflect.upload(files, files.length === 1 ? files[0].name : `${files.length} files`, setProgress)
        setProblems(r.problems ?? [])
        await open(r.id)
      } catch (e) {
        setError((e as Error).message)
        setPhase('failed')
      }
    },
    [open],
  )

  const useSample = useCallback(async () => {
    setPhase('sending')
    setError(null)
    try {
      const r = await inflect.sample()
      await open(r.id)
    } catch (e) {
      setError((e as Error).message)
      setPhase('failed')
    }
  }, [open])

  if (phase === 'ready' && detail && sessionId) {
    const lines = findings(detail)
    return (
      <section className="intake intake--ready">
        <h2 className="intake__eyebrow">What this is</h2>
        <ul className="findings">
          {lines.map((f, i) => (
            <li key={i} className="findings__item" style={{ animationDelay: `${i * 130}ms` }}>
              <span className="findings__tick" aria-hidden="true" />
              <span className="findings__text">
                {f.text}
                {f.detail && <span className="findings__why">{f.detail}</span>}
              </span>
            </li>
          ))}
        </ul>
        {problems.length > 0 && (
          <p className="intake__problems">
            {problems.length} file(s) could not be read: {problems.slice(0, 3).map((p) => p.name).join(', ')}
          </p>
        )}
        <div className="intake__actions" style={{ animationDelay: `${lines.length * 130 + 120}ms` }}>
          <button className="btn btn--primary" onClick={() => onOpen(sessionId, detail)}>
            Choose what to plot
          </button>
          <button className="btn btn--quiet" onClick={() => { setPhase('idle'); setDetail(null) }}>
            Use different data
          </button>
        </div>
      </section>
    )
  }

  const busy = phase === 'sending' || phase === 'reading'

  return (
    <section className="intake">
      <div
        className={`drop ${over ? 'is-over' : ''} ${busy ? 'is-busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          send(Array.from(e.dataTransfer.files))
        }}
        onClick={() => !busy && picker.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' && !busy) picker.current?.click() }}
      >
        <input
          ref={picker}
          type="file"
          multiple
          hidden
          onChange={(e) => send(Array.from(e.target.files ?? []))}
        />
        <div className="drop__mark" aria-hidden="true">
          <svg viewBox="0 0 120 64" className="drop__curve">
            <path d="M4 52 C 28 50, 40 44, 52 22 L 56 10 L 60 46 C 74 50, 96 50, 116 48" pathLength={1} />
          </svg>
        </div>
        <p className="drop__title">
          {phase === 'sending' ? 'Sending…' : phase === 'reading' ? 'Working out what this is…' : 'Drop your data here'}
        </p>
        <p className="drop__hint">
          {busy ? ' ' : 'any table of numbers — .dat, .csv, .txt, .tsv — one file or a whole run'}
        </p>
        {phase === 'sending' && progress > 0 && (
          <div className="drop__bar"><span style={{ width: `${Math.round(progress * 100)}%` }} /></div>
        )}
      </div>

      <div className="intake__alt">
        {sampleAvailable && (
          <button className="btn btn--quiet" onClick={useSample} disabled={busy}>
            or read the sample set — 296 compression runs
          </button>
        )}
      </div>

      {error && <p className="intake__error">{error}</p>}
    </section>
  )
}
