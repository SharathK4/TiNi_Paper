import { useState } from 'react'
import type { InvProgress, InvRunList, InvSpec } from '../../api/inverse'
import { secs, sci } from '../../lib/format'
import { Panel, Seg } from '../ui'
import { Tag } from './shared'
import { M } from '../Math'

/* D. Where a run is configured, started, watched and re-opened. */
export function WorkspacePanel({
  spec,
  progress,
  runs,
  selectedRun,
  onSelectRun,
  onRun,
  onResume,
  onCancel,
}: {
  spec: InvSpec | null
  progress: InvProgress | null
  runs: InvRunList | null
  selectedRun: string | null
  onSelectRun: (id: string) => void
  onRun: (config: Record<string, unknown>) => void
  onResume: (id: string) => void
  onCancel: () => void
}) {
  const d = spec?.defaults
  const [source, setSource] = useState<'rate_x_time' | 'smoothed_peak'>('rate_x_time')
  const [reference, setReference] = useState<string>('pdf_mle')
  const [bootstrapN, setBootstrapN] = useState(d?.bootstrap_n ?? 200)
  const [profilePoints, setProfilePoints] = useState(d?.profile_points ?? 21)
  const [replicates, setReplicates] = useState(d?.synthetic_replicates ?? 10)
  const [seed, setSeed] = useState(d?.seed ?? 0)
  const running = !!progress?.running

  const stageCls = (s: string) => {
    const st = progress?.stage_status?.[s] ?? 'pending'
    return st === 'done' || st.startsWith('skipped') ? 'is-done' : st === 'running' ? 'is-running' : st === 'failed' ? 'is-failed' : ''
  }

  return (
    <Panel
      index="2.4"
      title="Running the inference"
      lead={<>One run is one complete answer — fit, uncertainty, diagnostics, sensitivity, synthetic recovery — stored under its own identifier.</>}
    >
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <div className="figure__title">configuration</div>
          <div className="controls" style={{ marginTop: 8 }}>
            <div className="control">
              <span className="control__label">
                <span>definition of f‡</span>
              </span>
              <Seg
                value={source}
                onChange={setSource}
                options={[
                  { value: 'rate_x_time', label: <><M tex="\dot f\,t^\ddagger" /> (paper)</> },
                  { value: 'smoothed_peak', label: 'smoothed peak' },
                ]}
              />
            </div>
            <div className="control">
              <span className="control__label">
                <span>starts and comparison from</span>
              </span>
              <Seg
                value={reference}
                onChange={setReference}
                options={[
                  { value: 'pdf_mle', label: <>Table 1, <M tex="p(f\mid\dot f)" /></> },
                  { value: 'mean_lsq', label: <>Table 1, <M tex="\langle f\rangle(\dot f)" /></> },
                ]}
              />
            </div>
          </div>
          <div className="controls">
            <NumberField label="bootstrap resamples" value={bootstrapN} min={20} max={2000} step={20} onChange={setBootstrapN} />
            <NumberField label="profile points / parameter" value={profilePoints} min={5} max={61} step={2} onChange={setProfilePoints} />
            <NumberField label="synthetic replicates / design" value={replicates} min={0} max={50} step={1} onChange={setReplicates} />
            <NumberField label="random seed" value={seed} min={0} max={99999} step={1} onChange={setSeed} />
          </div>
          <p className="figure__note">
            Nelder–Mead from 7 starts, L-BFGS-B polish, bounds U‡ {d?.bounds.U_barrier.join('–')} pN·Å, ξ‡ {d?.bounds.xi_barrier.join('–')} Å, Γ₀{' '}
            {d ? `${sci(d.bounds.gamma0[0], 0)}–${sci(d.bounds.gamma0[1], 0)}` : ''} ps⁻¹. About 20 s, CPU only.
          </p>
          <div style={{ display: 'flex', gap: 10, marginTop: 24, alignItems: 'center' }}>
            {running ? (
              <button className="btn btn--quiet" onClick={onCancel}>
                Cancel
              </button>
            ) : (
              <button
                className="btn btn--primary"
                onClick={() =>
                  onRun({ source, reference_set: reference, bootstrap_n: bootstrapN, profile_points: profilePoints, synthetic_replicates: replicates, seed })
                }
              >
                Run inference
              </button>
            )}
            {!running && progress?.status === 'cancelled' && progress.run_id && (
              <button className="btn" onClick={() => onResume(progress.run_id!)}>
                Resume {progress.run_id}
              </button>
            )}
            {!running && progress?.status === 'failed' && <span className="figure__note">Last run failed: {progress.message}</span>}
          </div>
        </div>

        <div>
          <div className="figure__title">progress</div>
          <ul className="stage-list" style={{ marginTop: 8 }}>
            {(spec?.phases ?? []).map((ph) => {
              const st = progress?.stage_status?.[ph] ?? 'pending'
              const isCur = progress?.stage === ph && running
              return (
                <li key={ph}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 12.5 }}>{ph}</span>
                  <span className="st" style={{ color: 'var(--ink-2)' }}>
                    {isCur && progress && progress.files_total > 0 ? `${progress.files_done} / ${progress.files_total}` : isCur ? progress?.message : ''}
                  </span>
                  <span className={`st ${stageCls(ph)}`}>{st}</span>
                </li>
              )
            })}
          </ul>
          {progress && (
            <div className="readouts">
              <div className="readout">
                <div className="readout__label">status</div>
                <div className="readout__value">{progress.status}</div>
              </div>
              <div className="readout">
                <div className="readout__label">percent</div>
                <div className="readout__value num">{progress.percent.toFixed(0)} %</div>
              </div>
              <div className="readout">
                <div className="readout__label">elapsed</div>
                <div className="readout__value num">{secs(progress.elapsed_s)}</div>
              </div>
              {running && progress.eta_s !== null && (
                <div className="readout">
                  <div className="readout__label">remaining</div>
                  <div className="readout__value num">{secs(progress.eta_s)}</div>
                </div>
              )}
              {progress.run_id && (
                <div className="readout">
                  <div className="readout__label">run</div>
                  <div className="readout__value" style={{ fontFamily: 'var(--mono)', fontSize: 14 }}>
                    {progress.run_id}
                  </div>
                </div>
              )}
            </div>
          )}
          {running && (
            <div className="status__bar" style={{ marginTop: 12 }}>
              <span className="status__fill" style={{ width: `${progress?.percent ?? 0}%` }} />
            </div>
          )}

          <div className="figure__title" style={{ marginTop: 26 }}>
            previous runs · select to reopen
          </div>
          {runs && runs.runs.length > 0 ? (
            <div className="table-wrap table-wrap--auto" style={{ maxHeight: 220 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>run</th>
                    <th>finished</th>
                    <th>f‡ from</th>
                    <th>n</th>
                    <th><M tex="U^\ddagger" /></th>
                    <th><M tex="\xi^\ddagger" /></th>
                    <th><M tex="\Gamma_0" /></th>
                    <th>−ln L</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.runs.map((r) => (
                    <tr key={r.run_id} className={r.run_id === selectedRun ? 'is-selected' : ''} onClick={() => onSelectRun(r.run_id)}>
                      <td style={{ fontFamily: 'var(--mono)' }}>
                        {r.run_id}
                        {runs.latest?.run_id === r.run_id && <Tag kind="measured">latest</Tag>}
                      </td>
                      <td>{r.completed_at_iso?.replace('T', ' ')}</td>
                      <td>{r.source === 'rate_x_time' ? <M tex="\dot f\,t^\ddagger" /> : 'peak'}</td>
                      <td>{r.n}</td>
                      <td>{r.params?.U_barrier?.toFixed(0)}</td>
                      <td>{r.params?.xi_barrier?.toFixed(1)}</td>
                      <td>{sci(r.params?.gamma0, 2)}</td>
                      <td>{r.nll?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="figure__note">No completed run yet.</p>
          )}
        </div>
      </div>
    </Panel>
  )
}

function NumberField({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="control" style={{ minWidth: 160 }}>
      <span className="control__label">
        <span>{label}</span>
      </span>
      <input className="numfield num" type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Math.min(max, Math.max(min, Number(e.target.value))))} />
    </label>
  )
}
