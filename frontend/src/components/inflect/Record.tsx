/**
 * The record: what it would take for someone else to get the same numbers.
 *
 * A hash of the data as read, a hash of the code that read it, every setting,
 * the seed, and the findings. It is self-verifying: "verify" recomputes the
 * ledger from the record's own settings and reports, finding by finding,
 * whether the numbers still agree. A record that verifies is a result; one
 * that does not names exactly what moved.
 */

import { useCallback, useState } from 'react'
import { inflectLedger, type RecordDoc, type VerifyResult } from '../../api/inflect'

function shortHash(h: string | undefined): string {
  return h ? `${h.slice(0, 8)}…${h.slice(-4)}` : '—'
}

export function Record({ sessionId, file }: { sessionId: string; file: string | null }) {
  const [record, setRecord] = useState<RecordDoc | null>(null)
  const [verify, setVerify] = useState<VerifyResult | null>(null)
  const [busy, setBusy] = useState<'record' | 'verify' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const build = useCallback(async () => {
    setBusy('record')
    setError(null)
    setVerify(null)
    try {
      setRecord(await inflectLedger.record(sessionId, file ?? undefined))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }, [sessionId, file])

  const check = useCallback(async () => {
    if (!record) return
    setBusy('verify')
    setError(null)
    try {
      setVerify(await inflectLedger.verify(sessionId, record))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }, [sessionId, record])

  return (
    <section className="record">
      <h3>Reproduce it</h3>
      <p className="record__lead">
        A record names the data by its hash, the code by its hash, every setting and the seed, and carries the findings with your
        conclusions under their own heading. Verifying recomputes the findings from the record and says which still agree.
      </p>
      <div className="export__row">
        <button className="btn btn--quiet" onClick={build} disabled={busy !== null}>
          {busy === 'record' ? 'building…' : record ? 'rebuild the record' : 'build the record'}
        </button>
        {record && (
          <>
            <a className="btn btn--quiet" href={inflectLedger.recordUrl(sessionId, file ?? undefined)} download>
              download record
            </a>
            <button className="btn btn--primary" onClick={check} disabled={busy !== null}>
              {busy === 'verify' ? 'recomputing…' : 'verify'}
            </button>
          </>
        )}
      </div>
      {error && <p className="intake__error">{error}</p>}

      {record && (
        <dl className="record__facts">
          <div><dt>data</dt><dd className="num" title={record.data.sha256}>{record.data.file} · sha256 {shortHash(record.data.sha256)} · {record.data.n_rows.toLocaleString()} rows</dd></div>
          <div><dt>code</dt><dd className="num">Inflect {record.software.version} · {record.software.code_hash}</dd></div>
          <div><dt>settings</dt><dd className="num">
            {Object.entries(record.settings).filter(([, v]) => v !== null && v !== undefined && v !== '').map(([k, v]) => `${k} ${String(v)}`).join(' · ')}
          </dd></div>
          <div><dt>seed</dt><dd className="num">{record.seed}</dd></div>
          <div><dt>findings</dt><dd className="num">{record.findings.length} · {record.interpretations.length} of your conclusions</dd></div>
          <div><dt>made</dt><dd className="num">{record.created_at_iso.replace('T', ' ')}</dd></div>
        </dl>
      )}

      {verify && (
        <div className={`verdict ${verify.reproduced ? 'is-good' : 'is-bad'}`}>
          <p className="verdict__line">
            {verify.reproduced
              ? <>Reproduced. All <b className="num">{verify.n_same}</b> findings agree to one part in 10⁹; the data and the code are the ones the record names.</>
              : <><b className="num">{verify.n_same}</b> of <b className="num">{verify.n_recorded}</b> findings agree.</>}
          </p>
          {!verify.data_matches && <p className="verdict__row">The data on disk is not the data the record was made from.</p>}
          {!verify.code_matches && <p className="verdict__row">The analysis code has changed since the record was made.</p>}
          {verify.moved.length > 0 && (
            <ul className="verdict__list">
              {verify.moved.map((m) => (
                <li key={m.id} className="num">
                  {m.id}: recorded {String(m.recorded)}, now {String(m.now)}
                </li>
              ))}
            </ul>
          )}
          {verify.missing.length > 0 && <p className="verdict__row">No longer found: {verify.missing.join(', ')}</p>}
          {verify.new.length > 0 && <p className="verdict__row">Newly found: {verify.new.join(', ')}</p>}
        </div>
      )}
    </section>
  )
}
