import type { Progress } from '../api/client'
import { secs } from '../lib/format'

/* One quiet line under the masthead: where the pipeline is, and the two
 * actions that matter (resume, or redo from scratch). */
export function StatusStrip({ progress, offline, onRun, onCancel }: { progress: Progress | null; offline: boolean; onRun: (force: boolean) => void; onCancel: () => void }) {
  if (offline) {
    return (
      <div className="status">
        <span className="status__item">
          Backend not reachable — start it with <code>python -m tini serve</code> in <code>backend/</code>.
        </span>
      </div>
    )
  }
  if (!progress) return <div className="status">Connecting…</div>

  const running = progress.running
  const dotClass = (s: string) => {
    const st = progress.stage_status[s] ?? 'pending'
    if (st === 'done' || st.startsWith('skipped')) return 'is-done'
    if (st === 'running') return 'is-running'
    if (st === 'failed') return 'is-failed'
    return ''
  }
  const nothingYet = Object.values(progress.stage_status).every((s) => s === 'pending') && progress.status === 'idle'

  return (
    <div className="status" aria-live="polite">
      <span className="status__stages" title={progress.stages.join(' → ')}>
        {progress.stages.map((s) => (
          <span key={s} className={`status__dot ${dotClass(s)}`} title={`${s}: ${progress.stage_status[s]}`} />
        ))}
      </span>
      <span className="status__item">
        {running ? (
          <>
            <b>{progress.stage}</b>
            {progress.files_total > 0 && (
              <>
                {' '}
                · <b>{progress.files_done}</b> / {progress.files_total} files
              </>
            )}
          </>
        ) : nothingYet ? (
          'Not processed yet'
        ) : (
          <>
            <b>{progress.status}</b> · {progress.message}
          </>
        )}
      </span>
      {running && (
        <span className="status__bar">
          <span className="status__fill" style={{ width: `${progress.percent}%` }} />
        </span>
      )}
      <span className="status__item">
        <b>{progress.percent.toFixed(0)}%</b>
      </span>
      <span className="status__item">
        elapsed <b>{secs(progress.elapsed_s)}</b>
      </span>
      {running && progress.eta_s !== null && (
        <span className="status__item">
          remaining <b>{secs(progress.eta_s)}</b>
        </span>
      )}
      {progress.last_checkpoint && (
        <span className="status__item" title={progress.last_checkpoint.path}>
          checkpoint <b>{progress.last_checkpoint.stage}</b> · {progress.last_checkpoint.time_iso.slice(11, 19)}
        </span>
      )}
      <span className="status__actions">
        {running ? (
          <button className="btn btn--quiet" onClick={onCancel}>
            Cancel
          </button>
        ) : (
          <>
            <button className="btn btn--primary" onClick={() => onRun(false)}>
              {nothingYet ? 'Process dataset' : progress.status === 'done' ? 'Re-check' : 'Resume'}
            </button>
            {!nothingYet && (
              <button className="btn btn--quiet" onClick={() => onRun(true)} title="Ignore checkpoints and redo every stage">
                Redo all
              </button>
            )}
          </>
        )}
      </span>
    </div>
  )
}
