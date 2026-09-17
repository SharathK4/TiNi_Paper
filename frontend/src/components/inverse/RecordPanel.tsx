import { useState } from 'react'
import type { InvFit, InvResult, InvSpec } from '../../api/inverse'
import { fixed, sci } from '../../lib/format'
import { Empty, Panel } from '../ui'
import { Table } from './shared'

/* G. Everything needed to reproduce the run, and the optimiser's own record. */
export function RecordPanel({ result, spec, fit }: { result: InvResult | null; spec: InvSpec | null; fit: InvFit | null }) {
  const [showConfig, setShowConfig] = useState(false)
  const lead = <>Every number above traces to a dataset, a code version, a model equation, a configuration and a method.</>
  if (!result) {
    return (
      <Panel index="G" title="The record" lead={lead}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const cfg = result.config
  const stale = spec && spec.code_hash !== result.code_hash
  return (
    <Panel index="G" title="The record" lead={lead}>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <table className="facts">
            <tbody>
              <tr>
                <td>Run</td>
                <td style={{ fontFamily: 'var(--mono)' }}>{result.run_id}</td>
              </tr>
              <tr>
                <td>Created · completed</td>
                <td>
                  {result.created_at_iso?.replace('T', ' ')} · {result.completed_at_iso?.replace('T', ' ')}
                </td>
              </tr>
              <tr>
                <td>Dataset fingerprint</td>
                <td style={{ fontFamily: 'var(--mono)' }}>{result.dataset_fingerprint}</td>
              </tr>
              <tr>
                <td>Sample fingerprint</td>
                <td style={{ fontFamily: 'var(--mono)' }}>{result.sample_fingerprint}</td>
              </tr>
              <tr>
                <td>Model</td>
                <td>
                  <span style={{ fontFamily: 'var(--mono)' }}>{result.model_version}</span> — paper Eq. (6) likelihood, Eq. (1); code hash{' '}
                  <span style={{ fontFamily: 'var(--mono)' }}>{result.code_hash}</span>
                  {stale && <span className="tag tag--prototype">code has changed since this run</span>}
                </td>
              </tr>
              <tr>
                <td>Numerical method</td>
                <td>
                  Nelder–Mead (adaptive), {fit?.starts.length ?? 7} starts, maxiter {cfg.maxiter}, xatol {sci(cfg.xatol, 0)}, fatol {sci(cfg.fatol, 0)}; L-BFGS-B polish{' '}
                  {result.fit.polish?.improved ? 'improved the optimum' : 'confirmed the optimum'}; Hessian by central differences (h = 10⁻³ in ln w); profiles on{' '}
                  {cfg.profile_points} log-spaced points; bootstrap {cfg.bootstrap_n} resamples, seed {cfg.seed}
                </td>
              </tr>
              <tr>
                <td>Fit cost</td>
                <td>
                  {result.fit.n_eval_total} likelihood evaluations in {fixed(result.fit.duration_s, 2)} s
                </td>
              </tr>
              <tr>
                <td>Files</td>
                <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
                  {Object.values(result.phase_files).map((f) => (
                    <div key={f}>artifacts/{f}</div>
                  ))}
                </td>
              </tr>
            </tbody>
          </table>
          <button className="btn btn--quiet" style={{ marginTop: 16 }} onClick={() => setShowConfig((s) => !s)}>
            {showConfig ? 'hide' : 'show'} configuration and starts
          </button>
          {showConfig && <pre className="code-block">{JSON.stringify(cfg, null, 2)}</pre>}
        </div>
        <div>
          {fit && showConfig && (
            <>
              <div className="figure__title">multi-start record</div>
              <Table
                head={['start U‡', 'start ξ‡', 'start Γ₀', '→ U‡', 'ξ‡', 'Γ₀', '−ln L', 'evals', 'ok']}
                rows={fit.starts.map((s) => [
                  <span className="num">{s.start.U_barrier.toFixed(0)}</span>,
                  <span className="num">{s.start.xi_barrier.toFixed(1)}</span>,
                  <span className="num">{sci(s.start.gamma0, 1)}</span>,
                  <span className="num">{s.params.U_barrier.toFixed(1)}</span>,
                  <span className="num">{s.params.xi_barrier.toFixed(2)}</span>,
                  <span className="num">{sci(s.params.gamma0, 2)}</span>,
                  <span className="num">{s.nll.toFixed(3)}</span>,
                  s.n_eval,
                  s.converged ? 'yes' : 'no',
                ])}
              />
              <p className="figure__note">Converged: the best start reports success and at least two starts agree to 10⁻³ in −ln L.</p>
            </>
          )}
        </div>
      </div>
    </Panel>
  )
}
