import type { Progress, Summary } from '../api/client'
import { fixed, secs } from '../lib/format'
import { Fdot, M, MeanT } from './Math'
import { Panel } from './ui'

/* The experiment on paper, what this version verified against it, and the
 * processing record. Deliberately a reading panel, not a chart. */
export function SummaryPanel({ summary, progress }: { summary: Summary | null; progress: Progress | null }) {
  const s = summary
  return (
    <Panel index="1.6" title="The record" lead="What the simulation was, what this version established from the files, and where every number came from.">
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <div className="figure__title">The experiment, from the paper</div>
          {s && (
            <table className="facts">
              <tbody>
                <tr>
                  <td>Paper</td>
                  <td>
                    {s.paper.authors}, <i>{s.paper.journal}</i> · doi {s.paper.doi}
                  </td>
                </tr>
                <tr>
                  <td>System</td>
                  <td>{s.paper.system}</td>
                </tr>
                <tr>
                  <td>Ensemble</td>
                  <td>{s.paper.ensemble}</td>
                </tr>
                <tr>
                  <td>Loading</td>
                  <td>{s.paper.loading}</td>
                </tr>
                <tr>
                  <td>Force</td>
                  <td>
                    <M tex="f(t) = (A_{xx}/n)\,\sigma_{xx}(t)" />, force per atom, in pN
                  </td>
                </tr>
                <tr>
                  <td>Thermal energy</td>
                  <td>
                    <M tex="k_BT" /> = {fixed(s.constants.kBT_pN_A, 2)} pN·Å at {s.constants.temperature_K} K
                  </td>
                </tr>
                {s.inspection?.paper_rate_match && (
                  <tr>
                    <td>This dataset</td>
                    <td>
                      rate ({s.inspection.paper_rate_match.index}) of the paper's seven; {s.inspection.n_files} of the {s.inspection.paper_rate_match.n_traces} traces the paper used at this rate
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}

          <div className="figure__title" style={{ marginTop: 28 }}>
            Verified against the paper vs. prototype choices
          </div>
          <table className="facts">
            <tbody>
              <tr>
                <td>
                  <span className="tag tag--verified" style={{ marginLeft: 0 }}>
                    verified
                  </span>
                </td>
                <td>
                  Column units (ps, Å, pN) — confirmed by matching the file's extension rate and the mean-trace slope to the paper's rate (v) values, not assumed.
                </td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--verified" style={{ marginLeft: 0 }}>
                    verified
                  </span>
                </td>
                <td>
                  <M tex="\langle f(t)\rangle \approx \dot f t" /> (Eq. 2); cubic <M tex="U_0(\xi)" />; tilt <M tex="U_0 - f\xi" />; <M tex="f_c = 3U^\ddagger/2\xi^\ddagger" />; barrier
                  reduction (Eq. 4); Table 1 parameters.
                </td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--verified" style={{ marginLeft: 0 }}>
                    derived
                  </span>
                </td>
                <td>
                  <M tex="F(\xi) = -\mathrm{d}U_0/\mathrm{d}\xi" /> and the stationary points under force — one derivative of the paper's cubic, tested numerically.
                </td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--prototype" style={{ marginLeft: 0 }}>
                    prototype
                  </span>
                </td>
                <td>The fit window for <Fdot /> (paper: "a small time interval near t = 0"; here 100 ps by default, with the sensitivity shown).</td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--prototype" style={{ marginLeft: 0 }}>
                    prototype
                  </span>
                </td>
                <td>The peak picker for t‡ (moving average then argmax; the paper does not describe its method).</td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--inferred" style={{ marginLeft: 0 }}>
                    version 2
                  </span>
                </td>
                <td>
                  Fitting <M tex="U^\ddagger, \xi^\ddagger, \Gamma_0" /> to these events by maximum likelihood (paper Eq. 6) — <a href="#inverse">the inverse problem</a>.
                </td>
              </tr>
              <tr>
                <td>
                  <span className="tag tag--inferred" style={{ marginLeft: 0 }}>
                    version 3
                  </span>
                </td>
                <td>
                  Integrating the paper's Smoluchowski equation on this landscape — <a href="#forward">the forward model</a>.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <div className="figure__title">Established from the files</div>
          {s && (
            <table className="facts">
              <tbody>
                <tr>
                  <td>Traces</td>
                  <td>
                    {s.validation ? `${s.validation.n_ok} valid of ${s.validation.n_files}` : '—'}
                    {s.validation && `, ${s.validation.n_with_warnings} with warnings`}
                  </td>
                </tr>
                <tr>
                  <td>Extension rate</td>
                  <td>{s.inspection ? `${s.inspection.extension_rate_A_per_ps?.toExponential(4)} Å/ps` : '—'}</td>
                </tr>
                <tr>
                  <td>Force rate <Fdot /></td>
                  <td>
                    {s.force_rate ? `${fixed(s.force_rate.force_rate_pN_per_ps, 4)} ± ${fixed(s.force_rate.force_rate_se, 4)} pN/ps (t ≤ ${s.force_rate.window_ps} ps, R² ${fixed(s.force_rate.r2, 3)})` : '—'}
                  </td>
                </tr>
                <tr>
                  <td>Mean trace peak</td>
                  <td>{s.force_rate ? `${fixed(s.force_rate.mean_trace_max_pN, 2)} pN at ${fixed(s.force_rate.mean_trace_argmax_ps, 1)} ps` : '—'}</td>
                </tr>
                <tr>
                  <td><MeanT /></td>
                  <td>{s.events ? `${fixed(s.events.t_peak_ps.mean, 1)} ± ${fixed(s.events.t_peak_ps.std, 1)} ps (n = ${s.events.n})` : '—'}</td>
                </tr>
                <tr>
                  <td><M tex="\langle f^\ddagger\rangle = \dot f\,\langle t^\ddagger\rangle" /></td>
                  <td>{s.events ? `${fixed(s.events.f_crit_from_rate_pN.mean, 2)} ± ${fixed(s.events.f_crit_from_rate_pN.std, 2)} pN` : '—'}</td>
                </tr>
                {s.potential &&
                  Object.entries(s.potential).map(([k, v]) => (
                    <tr key={k}>
                      <td><M tex="f_c" /> · Table 1, {k === 'pdf_mle' ? <M tex="p(f\mid\dot f)" /> : <M tex="\langle f\rangle(\dot f)" />} fit</td>
                      <td>
                        {fixed(v.parameters.f_c_pN, 2)} pN · barrier {fixed(v.parameters.U_barrier_kBT, 1)} k<sub>B</sub>T
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}

          <div className="figure__title" style={{ marginTop: 28 }}>
            Processing record
          </div>
          {progress ? (
            <>
              <ul className="stage-list">
                {progress.stages.map((st) => {
                  const cp = progress.checkpoints?.[st]
                  const status = progress.stage_status[st]
                  const cls = status === 'done' || status?.startsWith('skipped') ? 'is-done' : status === 'running' ? 'is-running' : status === 'failed' ? 'is-failed' : ''
                  return (
                    <li key={st}>
                      <span style={{ fontFamily: 'var(--mono)', fontSize: 12.5 }}>{st}</span>
                      <span className="st" style={{ color: 'var(--ink-2)' }}>
                        {cp ? `${cp.outputs.length} file${cp.outputs.length === 1 ? '' : 's'} · ${secs(cp.duration_s)} · ${cp.completed_at_iso.replace('T', ' ')}` : '—'}
                      </span>
                      <span className={`st ${cls}`}>{status}</span>
                    </li>
                  )
                })}
              </ul>
              {progress.errors.length > 0 && (
                <div className="errors">{progress.errors.map((e) => `${e.time} [${e.stage}] ${e.type}: ${e.message}`).join('\n')}</div>
              )}
              <p className="figure__note" style={{ marginTop: 12 }}>
                Every stage writes its outputs atomically to <code>artifacts/</code> and records a fingerprint of the settings, the dataset and its own source. A
                stage is only redone when one of those changes; the previous copy of each file is kept as <code>.prev</code>. Static PNGs of every chart are in{' '}
                <code>artifacts/charts/</code>.
              </p>
            </>
          ) : (
            <p className="figure__note">No processing record yet.</p>
          )}
        </div>
      </div>
    </Panel>
  )
}
