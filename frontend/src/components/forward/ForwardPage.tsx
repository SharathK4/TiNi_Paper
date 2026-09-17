/**
 * Version 3 — the forward Smoluchowski model, one scrolling page:
 *
 *   A the equation and the headline prediction · B configuration and run ·
 *   C the density in motion · D the predicted observable against Eq. 6 and
 *   the sample · E the force spectrum · F numerical validation · G record
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Bar, ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip } from 'recharts'
import { useResource } from '../../api/hooks'
import { forward, type FwdObservables, type FwdProgress, type FwdResolved, type FwdResult, type FwdRunList, type FwdSpec, type Level } from '../../api/forward'
import { fixed, sci, secs } from '../../lib/format'
import { Eq, Fdot, M, MeanF } from '../Math'
import { Empty, Figure, Grid, Panel, Readout, Seg, Tip, XNum, YNum, chart } from '../ui'
import { Table, XLog } from '../inverse/shared'

function useForwardProgress() {
  const [progress, setProgress] = useState<FwdProgress | null>(null)
  const timer = useRef<number | null>(null)
  const tick = useCallback(async () => {
    try {
      const p = await forward.progress()
      setProgress(p)
      return p.running
    } catch {
      return false
    }
  }, [])
  useEffect(() => {
    let alive = true
    const loop = async () => {
      const running = await tick()
      if (!alive) return
      timer.current = window.setTimeout(loop, running ? 700 : 5000)
    }
    loop()
    return () => {
      alive = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [tick])
  return { progress, refresh: tick }
}

export function ForwardPage() {
  const { progress, refresh } = useForwardProgress()
  const [version, setVersion] = useState(0)
  const wasRunning = useRef(false)
  useEffect(() => {
    if (progress?.running) wasRunning.current = true
    else if (wasRunning.current) {
      wasRunning.current = false
      setVersion((v) => v + 1)
    }
  }, [progress?.running])

  const spec = useResource<FwdSpec>(() => forward.spec(), [])
  const runs = useResource<FwdRunList>(() => forward.runs(), [], version)
  const [selected, setSelected] = useState<string | null>(null)
  const runId = selected ?? runs.data?.latest?.run_id ?? null
  const result = useResource<FwdResult>(() => (runId ? forward.result(runId) : Promise.reject(new Error('none'))), [runId], version)
  const obs = useResource<FwdObservables>(() => (runId ? forward.observables(runId) : Promise.reject(new Error('none'))), [runId], version)
  const res = result.data

  const run = useCallback(
    async (cfg: Record<string, unknown>) => {
      try {
        await forward.run(cfg)
        await refresh()
      } catch (e) {
        window.alert(`Could not start: ${(e as Error).message}`)
      }
    },
    [refresh],
  )
  const cancel = useCallback(async () => {
    await forward.cancel()
    await refresh()
  }, [refresh])

  return (
    <>
      <header className="masthead">
        <h1 className="wordmark">TiNi</h1>
        <p className="standfirst">
          The forward model: the paper's Smoluchowski equation integrated in time on the cubic landscape, so that the critical-force density is computed from
          the dynamics rather than from the Kramers rate law.
          {res && (
            <>
              {' '}
              Predicted <b><MeanF /> = {fixed(res.observables.mean_pN, 2)} pN</b> against Eq. (6)'s {fixed(res.observables.eq6.mean_pN, 2)} with the same parameters.
            </>
          )}
        </p>
        <p className="byline">Version 3 · forward Smoluchowski model · after Maitra &amp; Gururajan, Sci. Rep. 15:3008 (2025)</p>
      </header>

      <ModelPanel res={res} />
      <RunPanel spec={spec.data} progress={progress} runs={runs.data} selectedRun={runId} onSelectRun={setSelected} onRun={run} onCancel={cancel} />
      <DensityPanel res={res} obs={obs.data} />
      <ObservablePanel res={res} obs={obs.data} />
      <SpectrumPanel res={res} spec={spec.data} />
      <ValidationPanel res={res} />
      <RecordPanel res={res} spec={spec.data} />

      <p className="footnote">
        Solver: finite volume, Scharfetter–Gummel fluxes, backward Euler; <code>backend/tini/forward</code>. Runs: <code>artifacts/forward/runs/</code>. Specification and
        validation: <code>docs/FORWARD_MODEL.md</code>.
      </p>
    </>
  )
}

/* ---------------------------------------------------------------- A */
function ModelPanel({ res }: { res: FwdResult | null }) {
  const p = res?.parameters
  return (
    <Panel index="3.1" title="The forward model" lead={<>The probability of the order parameter, pushed by thermal noise and pulled by a force that grows in time, until it leaves the well.</>}>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div className="prose">
          <Eq tag="paper · Einstein–Smoluchowski · verified" tex="\partial_t P(\xi,t) = \partial_\xi\, D\left[\partial_\xi P + \frac{1}{k_BT}\,\frac{\partial U}{\partial\xi}\,P\right],\qquad U(\xi,t) = U_0(\xi) - \dot f\,t\,\xi,\qquad D = \frac{k_BT}{\eta}" />
          <Eq tag="paper · the friction fixed by Γ₀ (Kramers prefactor) · verified" tex="\Gamma_0 = \frac{6U^\ddagger/(\xi^\ddagger)^2}{2\pi\eta}\,e^{-U^\ddagger/k_BT}\ \Rightarrow\ \eta\ [\mathrm{pN\,ps/\AA}]" />
          <p>
            Reflecting wall on the left of the well; absorbing boundary at or beyond the barrier. The survival probability <M tex="\Psi(t)=\int P\,d\xi" /> falls as
            density is absorbed; the absorbed flux is the first-passage density, and <M tex="p(f\mid\dot f) = -\dot f^{-1}\,d\Psi/dt" /> at <M tex="f=\dot f t" /> is
            the same observable Version 2 fits with Eq. (6) — here without the Kramers and steady-flux approximations.
          </p>
        </div>
        <div>
          {p ? (
            <table className="facts">
              <tbody>
                <tr>
                  <td>Parameters</td>
                  <td>{res?.parameter_provenance.label}</td>
                </tr>
                <tr>
                  <td>
                    <M tex="U^\ddagger,\ \xi^\ddagger,\ \Gamma_0" />
                  </td>
                  <td className="num">
                    {fixed(p.U_barrier_pN_A, 1)} pN·Å ({fixed(p.U_barrier_kBT, 1)} <M tex="k_BT" />) · {fixed(p.xi_barrier_A, 2)} Å · {sci(p.gamma0_per_ps, 2)} ps⁻¹
                  </td>
                </tr>
                <tr>
                  <td>
                    <M tex="\eta,\ D" />
                  </td>
                  <td className="num">
                    {fixed(p.eta_pN_ps_per_A, 4)} pN·ps/Å · {fixed(p.D_A2_per_ps, 1)} Å²/ps
                  </td>
                </tr>
                <tr>
                  <td>Time scales</td>
                  <td className="num">
                    well relaxation {fixed(p.tau_well_ps, 3)} ps · diffusion over <M tex="\xi^\ddagger" /> {fixed(p.tau_diffusion_ps, 1)} ps · ramp to <M tex="f_c" /> {fixed(res?.t_end_ps, 0)} ps
                  </td>
                </tr>
                <tr>
                  <td>Loading</td>
                  <td className="num">
                    <Fdot /> = {fixed(res?.force_rate_pN_per_ps, 4)} pN/ps ({res?.force_rate_provenance.source === 'v1' ? 'measured in Version 1' : 'user-specified'})
                  </td>
                </tr>
                <tr>
                  <td>Boundary</td>
                  <td className="num">
                    {res?.boundary.boundary === 'barrier_top' ? <>absorbing at <M tex="\xi^\ddagger" /></> : <>absorbing beyond the barrier, {res?.boundary.margin_kBT} <M tex="k_BT" /> below the top</>} · <M tex="\xi_A" /> = {fixed(res?.boundary.xi_A_A, 2)} Å ·
                    reflecting at <M tex="\xi_L" /> = {fixed(res?.boundary.xi_L_A, 1)} Å ({fixed(res?.boundary.U0_at_xi_L_kBT, 0)} <M tex="k_BT" /> up)
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <Empty>No forward run yet — section 3.2.</Empty>
          )}
        </div>
      </div>
    </Panel>
  )
}

/* ---------------------------------------------------------------- B */
function RunPanel({ spec, progress, runs, selectedRun, onSelectRun, onRun, onCancel }: { spec: FwdSpec | null; progress: FwdProgress | null; runs: FwdRunList | null; selectedRun: string | null; onSelectRun: (id: string) => void; onRun: (cfg: Record<string, unknown>) => void; onCancel: () => void }) {
  const [source, setSource] = useState('inverse:latest')
  const [boundary, setBoundary] = useState('beyond_barrier')
  const [cells, setCells] = useState(250)
  const [steps, setSteps] = useState(20000)
  const [customU, setCustomU] = useState(600)
  const [customXi, setCustomXi] = useState(30)
  const [customLogG, setCustomLogG] = useState(-6.5)
  const [rate, setRate] = useState<string>('')
  const [temp, setTemp] = useState(300)
  const running = !!progress?.running
  const resolved = useResource<FwdResolved>(() => (source === 'custom' ? Promise.reject(new Error('custom')) : forward.parameters(source, undefined, temp)), [source, temp])

  const config = () => ({
    parameter_source: source,
    boundary,
    n_cells: cells,
    n_steps: steps,
    temperature_K: temp,
    ...(rate ? { force_rate: Number(rate) } : {}),
    ...(source === 'custom' ? { custom: { U_barrier: customU, xi_barrier: customXi, gamma0: Math.pow(10, customLogG) } } : {}),
  })
  const stageCls = (s: string) => {
    const st = progress?.stage_status?.[s] ?? 'pending'
    return st === 'done' || st.startsWith('skipped') ? 'is-done' : st === 'running' ? 'is-running' : st === 'failed' ? 'is-failed' : ''
  }

  return (
    <Panel index="3.2" title="Running the model" lead={<>Choose where the three parameters come from, where the density is absorbed, and how finely to resolve ξ and t.</>}>
      <div className="grid-2" style={{ alignItems: 'start' }}>
        <div>
          <div className="controls">
            <div className="control">
              <span className="control__label">
                <span>parameters from</span>
              </span>
              <Seg
                value={source}
                onChange={setSource}
                options={[
                  { value: 'inverse:latest', label: 'Version 2 fit' },
                  { value: 'paper:pdf_mle', label: <>paper, <M tex="p(f\mid\dot f)" /></> },
                  { value: 'paper:mean_lsq', label: <>paper, <M tex="\langle f\rangle(\dot f)" /></> },
                  { value: 'custom', label: 'custom' },
                ]}
              />
            </div>
            <div className="control">
              <span className="control__label">
                <span>absorbing boundary</span>
              </span>
              <Seg
                value={boundary}
                onChange={setBoundary}
                options={[
                  { value: 'beyond_barrier', label: 'beyond the barrier' },
                  { value: 'barrier_top', label: <>at <M tex="\xi^\ddagger" /> (paper, literal)</> },
                ]}
              />
            </div>
          </div>
          {source === 'custom' && (
            <div className="controls">
              <NumberField label="U‡ [pN·Å]" value={customU} min={100} max={3000} step={1} onChange={setCustomU} />
              <NumberField label="ξ‡ [Å]" value={customXi} min={2} max={200} step={0.1} onChange={setCustomXi} />
              <NumberField label="log₁₀ Γ₀ [ps⁻¹]" value={customLogG} min={-14} max={2} step={0.1} onChange={setCustomLogG} />
            </div>
          )}
          <div className="controls">
            <NumberField label="cells in ξ" value={cells} min={50} max={4000} step={50} onChange={setCells} />
            <NumberField label="time steps" value={steps} min={500} max={400000} step={500} onChange={setSteps} />
            <label className="control" style={{ minWidth: 160 }}>
              <span className="control__label">
                <span>force rate [pN/ps] · blank = Version 1</span>
              </span>
              <input className="numfield num" type="text" value={rate} placeholder={resolved.data ? fixed(resolved.data.force_rate_pN_per_ps, 4) : ''} onChange={(e) => setRate(e.target.value)} />
            </label>
            <NumberField label="temperature [K]" value={temp} min={100} max={1000} step={10} onChange={setTemp} />
          </div>
          {resolved.data && (
            <p className="figure__note">
              {resolved.data.provenance.label}: U‡ {fixed(resolved.data.parameters.U_barrier_pN_A, 1)} pN·Å, ξ‡ {fixed(resolved.data.parameters.xi_barrier_A, 2)} Å, Γ₀ {sci(resolved.data.parameters.gamma0_per_ps, 2)} ps⁻¹ →
              η {fixed(resolved.data.parameters.eta_pN_ps_per_A, 4)} pN·ps/Å, D {fixed(resolved.data.parameters.D_A2_per_ps, 1)} Å²/ps. Ramp to f_c in {fixed(resolved.data.t_end_ps, 0)} ps, Δt ={' '}
              {sci(resolved.data.t_end_ps / steps, 2)} ps. {temp !== 300 && 'η is held at its 300 K value; Γ₀ is rescaled by Kramers — an assumption beyond the paper.'}
            </p>
          )}
          {resolved.missing && source.startsWith('inverse') && <p className="figure__note">No Version 2 run exists yet; choose paper or custom parameters.</p>}
          <p className="figure__note">
            Validation, grid/time refinement and the seven-rate spectrum are included (about 12 s in total). Backward Euler, unconditionally stable; the time step
            is chosen for accuracy, and its effect is measured in section 3.6.
          </p>
          <div style={{ display: 'flex', gap: 10, marginTop: 24, alignItems: 'center' }}>
            {running ? (
              <button className="btn btn--quiet" onClick={onCancel}>
                Cancel
              </button>
            ) : (
              <button className="btn btn--primary" onClick={() => onRun(config())}>
                Run forward model
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
                  <span className="st" style={{ color: 'var(--ink-2)' }}>{isCur && progress && progress.files_total > 0 ? `${progress.files_done} / ${progress.files_total}` : isCur ? progress?.message : ''}</span>
                  <span className={`st ${stageCls(ph)}`}>{st}</span>
                </li>
              )
            })}
          </ul>
          {progress && (
            <div className="readouts">
              <Readout label="status" value={progress.status} />
              <Readout label="percent" value={`${progress.percent.toFixed(0)} %`} />
              <Readout label="elapsed" value={secs(progress.elapsed_s)} />
              {running && progress.eta_s !== null && <Readout label="remaining" value={secs(progress.eta_s)} />}
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
                    <th>parameters</th>
                    <th>boundary</th>
                    <th><Fdot /></th>
                    <th><MeanF /></th>
                    <th>cells × steps</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.runs.map((r) => (
                    <tr key={r.run_id} className={r.run_id === selectedRun ? 'is-selected' : ''} onClick={() => onSelectRun(r.run_id)}>
                      <td style={{ fontFamily: 'var(--mono)' }}>{r.run_id}</td>
                      <td>{r.completed_at_iso?.replace('T', ' ')}</td>
                      <td>{r.parameter_source}</td>
                      <td>{r.boundary === 'barrier_top' ? 'at ξ‡' : 'beyond'}</td>
                      <td>{fixed(r.force_rate, 4)}</td>
                      <td>{fixed(r.mean_pN, 2)}</td>
                      <td>
                        {r.n_cells} × {r.n_steps}
                      </td>
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

/* ---------------------------------------------------------------- C */
function DensityPanel({ res, obs }: { res: FwdResult | null; obs: FwdObservables | null }) {
  const [frac, setFrac] = useState(0.6)
  const snaps = obs?.snapshots
  const pick = useMemo(() => {
    if (!snaps || !res || !snaps.P?.length || snaps.P.length !== snaps.f_pN.length) return []
    const fc = res.parameters.f_c_pN
    const want = [0, 0.3, 0.5, frac, 0.8].map((x) => x * fc)
    return want.map((f) => {
      const i = snaps.f_pN.reduce((best, v, k) => (Math.abs(v - f) < Math.abs(snaps.f_pN[best] - f) ? k : best), 0)
      return { f: snaps.f_pN[i], t: snaps.t_ps[i], P: snaps.P[i] }
    })
  }, [snaps, res, frac])
  const data = useMemo(() => {
    if (!snaps || !pick.length) return []
    return snaps.xi_A.map((x, i) => {
      const row: Record<string, number> = { xi: x }
      pick.forEach((s, k) => (row[`p${k}`] = s.P?.[i] ?? 0))
      return row
    })
  }, [snaps, pick])
  const surv = useMemo(() => (obs ? obs.t_ps.map((t, i) => ({ t, f: t * obs.force_rate_pN_per_ps, s: obs.survival_t[i] })) : []), [obs])
  const lead = <>Five moments of the same density as the force rises: it leans toward the barrier, thins, and is gone by the time the barrier is.</>
  if (!res || !obs) {
    return (
      <Panel index="3.3" title="The density in motion" lead={lead}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const shades = ['#2f3a63', '#4a5680', '#6b77a8', '#b5462a', '#a5a9bd']
  return (
    <Panel index="3.3" title="The density in motion" lead={lead}>
      <div className="grid-2">
        <Figure title={<><M tex="P(\xi,t)" /> at five forces · terracotta is adjustable</>} note={<>Dashed: the absorbing boundary at <M tex="\xi_A" /> = {fixed(res.boundary.xi_A_A, 1)} Å. Density is not renormalised; what is missing has been absorbed.</>}>
          <div className="controls" style={{ marginBottom: 12 }}>
            <label className="control">
              <span className="control__label">
                <span>highlighted snapshot</span>
                <b>{fixed(frac * res.parameters.f_c_pN, 1)} pN · {(100 * frac).toFixed(0)} % of fc</b>
              </span>
              <input type="range" min={0} max={1} step={0.01} value={frac} onChange={(e) => setFrac(Number(e.target.value))} />
            </label>
          </div>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={chart.margin}>
                <Grid />
                <XNum dataKey="xi" label="ξ [Å]" range={[Math.floor(res.grid.xi_L_A / 10) * 10, Math.ceil(res.grid.xi_A_A / 10) * 10]} />
                <YNum label="P [1/Å]" />
                <Tooltip content={<Tip unitX="Å" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.name}: ${typeof p.value === 'number' ? p.value.toFixed(4) : ''}`} />} />
                {pick.map((s, k) => (
                  <Line key={k} dataKey={`p${k}`} name={`f = ${s.f.toFixed(1)} pN`} type="monotone" stroke={shades[k]} strokeWidth={k === 3 ? 1.8 : 1.1} dot={false} isAnimationActive={false} />
                ))}
                <ReferenceLine x={res.boundary.xi_A_A} stroke={chart.ink3} strokeDasharray="4 3" />
                <ReferenceLine x={0} stroke={chart.grid} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure title={<>survival <M tex="\Psi(t)" /> against the applied force</>} note={<>Reaches {sci(res.observables.residual_mass, 1)} at <M tex="f_c" />: the residual mass that would transition deterministically when the barrier vanishes.</>}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={surv} margin={chart.margin}>
                <Grid />
                <XNum dataKey="f" label="applied force f [pN]" range={[0, Math.ceil(res.parameters.f_c_pN / 5) * 5]} />
                <YNum label="Ψ" range={[0, 1]} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `Ψ = ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`} />} />
                <Line dataKey="s" type="monotone" stroke={chart.model} strokeWidth={1.6} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>
    </Panel>
  )
}

/* ---------------------------------------------------------------- D */
function ObservablePanel({ res, obs }: { res: FwdResult | null; obs: FwdObservables | null }) {
  const dc = res?.observables.data_comparison
  const hasData = dc && !('unavailable' in dc)
  const pdf = useMemo(() => (obs ? obs.f_pN.map((f, i) => ({ f, x: f, p: obs.pdf[i], q: obs.pdf_eq6[i], c: obs.cdf[i], cq: obs.cdf_eq6[i] })) : []), [obs])
  const hist = useMemo(() => (hasData ? dc.histogram.centers.map((x, i) => ({ x, d: dc.histogram.density[i], err: dc.histogram.density_err[i] })) : []), [dc, hasData])
  const lead = <>The critical-force density from the dynamics, beside the paper's closed form for the same parameters, and beside the hundred measured forces.</>
  if (!res || !obs) {
    return (
      <Panel index="3.4" title="The predicted observable" lead={lead}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const o = res.observables
  const xr: [number, number] = [0, Math.ceil(res.parameters.f_c_pN / 5) * 5]
  return (
    <Panel index="3.4" title="The predicted observable" lead={lead}>
      <div className="grid-2">
        <Figure title={<>density <M tex="p(f\mid\dot f)" /> · PDE (slate), Eq. 6 (dashed), sample (terracotta)</>} note="Same U‡, ξ‡, Γ₀ in both curves. The sample histogram is Version 1's; bars ± √n.">
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={(hasData ? hist : pdf) as Record<string, number>[]} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xr} />
                <YNum label="density [1/pN]" />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.dataKey === 'p' ? 'PDE' : p.dataKey === 'q' ? 'Eq. 6' : 'sample'} ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`} />} />
                {hasData && (
                  <Bar dataKey="d" fill={chart.accent} fillOpacity={0.45} isAnimationActive={false} radius={[2, 2, 0, 0]}>
                    <ErrorBar dataKey="err" width={3} stroke={chart.accent} strokeWidth={1} />
                  </Bar>
                )}
                <Line data={pdf} dataKey="q" type="monotone" stroke={chart.model2} strokeWidth={1.2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line data={pdf} dataKey="p" type="monotone" stroke={chart.model} strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure title="cumulative" note={`Largest gap between PDE and Eq. 6: ${fixed(o.max_cdf_diff_pde_vs_eq6, 3)}; L1 distance between the densities ${fixed(o.l1_pde_vs_eq6, 3)}.`}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={pdf} margin={chart.margin}>
                <Grid />
                <XNum dataKey="f" label="f‡ [pN]" range={xr} />
                <YNum label="P(f ≤ f‡)" range={[0, 1]} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.dataKey === 'c' ? 'PDE' : 'Eq. 6'} ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`} />} />
                <Line dataKey="cq" type="monotone" stroke={chart.model2} strokeWidth={1.2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line dataKey="c" type="monotone" stroke={chart.model} strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>
      <div className="readouts">
        <Readout label={<>PDE <MeanF />, s.d.</>} value={`${fixed(o.mean_pN, 2)}, ${fixed(o.sd_pN, 2)}`} unit="pN" tone="model" />
        <Readout label={<>Eq. 6 <MeanF />, s.d.</>} value={`${fixed(o.eq6.mean_pN, 2)}, ${fixed(o.eq6.sd_pN, 2)}`} unit="pN" />
        <Readout label={<>Eq. 8 <MeanF /></>} value={fixed(o.eq6.mean_eq8_pN, 2)} unit="pN" />
        {hasData && <Readout label={<>sample <MeanF />, s.d.</>} value={`${fixed(dc.sample_mean_pN, 2)}, ${fixed(dc.sample_sd_pN, 2)}`} unit={`pN · n = ${dc.n}`} tone="accent" />}
        {hasData && <Readout label="ln L of sample · PDE / Eq. 6" value={`${fixed(dc.log_likelihood_pde, 1)} / ${fixed(dc.log_likelihood_eq6, 1)}`} />}
        {hasData && <Readout label="KS D, sample vs PDE" value={fixed(dc.ks_D_vs_pde, 3)} />}
        <Readout label={<M tex="\psi(f_c)" />} value={sci(o.residual_mass, 1)} />
      </div>
      {hasData && (
        <p className="figure__note" style={{ marginTop: 10 }}>
          {dc.ks_note}. {res.parameter_provenance.source.startsWith('inverse') ? 'Agreement with the sample is expected here: these parameters were fitted to it (Version 2) — this run tests the Kramers approximation, not the physics.' : 'These parameters were not fitted to this sample; with n = 100 the 5 % critical value of D is 0.136.'}
        </p>
      )}
    </Panel>
  )
}

/* ---------------------------------------------------------------- E */
function SpectrumPanel({ res, spec }: { res: FwdResult | null; spec: FwdSpec | null }) {
  const rows = res && !('skipped' in res.spectrum) ? res.spectrum : null
  const data = useMemo(() => (rows ? rows.map((r) => ({ x: r.force_rate, pde: r.mean_pde_pN, eq8: r.mean_eq8_pN, eq6: r.mean_eq6_exact_pN, sd: r.sd_pde_pN })) : []), [rows])
  const fig3 = useMemo(() => (spec ? spec.fig3_readings_by_eye.map((r) => ({ x: r.force_rate_pN_per_ps, y: r.mean_f_pN, e: r.uncertainty_pN })) : []), [spec])
  const dc = res?.observables.data_comparison
  const ours = dc && !('unavailable' in dc) ? [{ x: dc.sample_force_rate_pN_per_ps, y: dc.sample_mean_pN, e: dc.sample_sd_pN }] : []
  const lead = <>The same parameters at the paper's seven loading rates: the PDE's mean critical force beside Eq. (8), Eq. (6) and the paper's figure.</>
  if (!rows) {
    return (
      <Panel index="3.5" title="The force spectrum" lead={lead}>
        <Empty>No spectrum in this run.</Empty>
      </Panel>
    )
  }
  return (
    <Panel index="3.5" title="The force spectrum" lead={lead}>
      <Figure title={<><M tex="\langle f^\ddagger\rangle(\dot f)" /> · PDE (slate points), Eq. 8 (line), Eq. 6 exact (dotted)</>} note="Grey: the paper's Fig. 3 symbols read by eye, ± 1 pN. Terracotta: this dataset. The PDE sits above the closed forms by an amount that grows with rate — the transit past the barrier that the steady-flux approximation neglects.">
        <div className="chart chart--tall">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={chart.margin}>
              <Grid />
              <XLog dataKey="x" label="force rate [pN/ps]" lo={2e-3} hi={2} />
              <YNum label="mean f‡ [pN]" range={[5, 35]} />
              <Tooltip content={<Tip unitX="pN/ps" fmtX={(v) => sci(v, 2)} rows={(p) => `${p.name ?? p.dataKey}: ${typeof p.value === 'number' ? p.value.toFixed(2) : ''} pN`} />} />
              <Line dataKey="eq8" name="Eq. 8" type="monotone" stroke={chart.model2} strokeWidth={1.4} dot={false} isAnimationActive={false} />
              <Line dataKey="eq6" name="Eq. 6 exact" type="monotone" stroke={chart.model2} strokeWidth={1} strokeDasharray="2 3" dot={false} isAnimationActive={false} />
              <Scatter data={fig3} dataKey="y" name="Fig. 3" fill={chart.ink3} isAnimationActive={false}>
                <ErrorBar dataKey="e" width={3} stroke={chart.ink3} direction="y" />
              </Scatter>
              <Scatter data={ours} dataKey="y" name="this dataset" fill={chart.accent} isAnimationActive={false}>
                <ErrorBar dataKey="e" width={4} stroke={chart.accent} direction="y" />
              </Scatter>
              <Scatter dataKey="pde" name="PDE" fill={chart.model} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Figure>
      <Table
        head={['rate', <><Fdot /> [pN/ps]</>, <>PDE <MeanF /></>, 'PDE s.d.', 'Eq. 6', 'Eq. 8', 'PDE − Eq. 6', <M tex="\psi(f_c)" />]}
        rows={rows.map((r) => [
          r.index,
          <span className="num">{sci(r.force_rate, 3)}</span>,
          <span className="num">{fixed(r.mean_pde_pN, 2)}</span>,
          <span className="num">{fixed(r.sd_pde_pN, 2)}</span>,
          <span className="num">{fixed(r.mean_eq6_exact_pN, 2)}</span>,
          <span className="num">{fixed(r.mean_eq8_pN, 2)}</span>,
          <span className="num">{fixed(r.mean_pde_pN - r.mean_eq6_exact_pN, 2)}</span>,
          <span className="num">{sci(r.residual_mass_pde, 1)}</span>,
        ])}
      />
    </Panel>
  )
}

/* ---------------------------------------------------------------- F */
function ValidationPanel({ res }: { res: FwdResult | null }) {
  const v = res && !('skipped' in res.validation) ? res.validation : null
  const c = res && !('skipped' in res.convergence) ? res.convergence : null
  const lead = <>What the solver is known to get exactly right, what it gets right to a stated tolerance, and how the answer moves with resolution.</>
  if (!res) {
    return (
      <Panel index="3.6" title="Numerical validation" lead={lead}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const levelRows = (levels: Level[]) => levels.map((l) => [<span className="num">{l.n_cells}</span>, <span className="num">{l.n_steps}</span>, <span className="num">{fixed(l.h_A, 3)}</span>, <span className="num">{sci(l.dt_ps, 2)}</span>, <span className="num">{fixed(l.mean_pN, 4)}</span>, <span className="num">{l.l1_vs_previous !== undefined ? sci(l.l1_vs_previous, 2) : '—'}</span>, <span className="num">{l.mean_change_pN !== undefined ? sci(l.mean_change_pN, 2) : '—'}</span>])
  return (
    <Panel index="3.6" title="Numerical validation" lead={lead}>
      {v && (
        <Table
          head={['test', 'statement', 'benchmark', 'result']}
          rows={[
            ['A conservation', 'Ψ + absorbed = 1 at every step', 'identity of the scheme', <span className="num">max deviation {sci(v.A_B_conservation_positivity.conservation_error_max, 1)}</span>],
            ['B positivity', 'P ≥ 0 everywhere, no clipping', 'M-matrix property', <span className="num">min P = {sci(v.A_B_conservation_positivity.min_density, 1)} /Å</span>],
            ['C1 equilibrium', 'Boltzmann density stationary (both walls reflecting)', 'exact', <span className="num">L1 drift {sci(v.C1_reflecting_equilibrium.l1_drift, 1)} over {v.C1_reflecting_equilibrium.t_end_ps} ps</span>],
            ...v.C2_mfpt_exact.map((m, i) => [`C2 first passage${i ? ' (at ξ‡)' : ''}`, <>mean first-passage time at fixed f = {fixed(m.force_pN, 1)} pN ({fixed(m.barrier_remaining_kBT, 1)} <M tex="k_BT" /> barrier)</>, 'exact 1-D quadrature formula', <span className="num">{fixed(m.tau_pde_ps, 2)} vs {fixed(m.tau_exact_ps, 2)} ps · {(100 * m.relative_error).toFixed(2)} %</span>]),
            ['C3 diffusion', 'U ≡ 0: variance grows as 2Dt', 'analytic', <span className="num">{(100 * v.C3_free_diffusion.relative_error).toExponential(1)} %</span>],
            ...v.F_kramers.placements.map((p) => [`F Kramers · ${p.placement.replace('_', ' ')}`, <>escape rate at fixed f = {fixed(v.F_kramers.force_pN, 1)} pN ({fixed(v.F_kramers.barrier_remaining_kBT, 1)} <M tex="k_BT" />), absorbing at ξ = {fixed(p.xi_A_A, 1)} Å</>, "paper's Γ(f), Eq. 4", <span className="num"><M tex="\Gamma_{\mathrm{PDE}}/\Gamma_{\mathrm{Eq.\,4}}" /> = {fixed(p.ratio_pde_over_eq4, 3)}</span>]),
            ['G reproducibility', 'identical arrays on rerun', 'bitwise', v.G_reproducibility.identical_survival && v.G_reproducibility.identical_snapshots ? 'identical' : `differs by ${sci(v.G_reproducibility.max_abs_diff, 1)}`],
          ]}
        />
      )}
      <p className="figure__note">
        Kramers' rate is recovered when the density is absorbed beyond the barrier; absorbing on the transition state itself escapes faster by the recrossing factor
        (→ 2 for high barriers). The paper's Eq. (4) therefore corresponds to the "beyond" placement.
      </p>
      {c && (
        <div className="grid-2" style={{ marginTop: 36, alignItems: 'start' }}>
          <div>
            <div className="figure__title">grid refinement · observed order {fixed(c.grid_order, 2)} · Richardson error {sci(c.richardson_grid_pN, 1)} pN</div>
            <Table head={['cells', 'steps', 'h [Å]', 'Δt [ps]', <><MeanF /> [pN]</>, 'L1 vs coarser', <>Δ<MeanF /></>]} rows={levelRows(c.grid_levels)} />
          </div>
          <div>
            <div className="figure__title">time refinement · observed order {fixed(c.time_order, 2)} · Richardson error {sci(c.richardson_time_pN, 1)} pN</div>
            <Table head={['cells', 'steps', 'h [Å]', 'Δt [ps]', <><MeanF /> [pN]</>, 'L1 vs coarser', <>Δ<MeanF /></>]} rows={levelRows(c.time_levels)} />
          </div>
        </div>
      )}
      <p className="figure__note">
        Second order in ξ (Scharfetter–Gummel), first order in t (backward Euler); the Richardson estimates bound the discretisation error of <MeanF /> for this run,
        which is far below the 0.2–0.3 pN by which the PDE differs from Eq. (6). There is no closed-form solution for the ramped problem; C2 is the exact benchmark.
      </p>
    </Panel>
  )
}

/* ---------------------------------------------------------------- G */
function RecordPanel({ res, spec }: { res: FwdResult | null; spec: FwdSpec | null }) {
  const [show, setShow] = useState(false)
  if (!res) {
    return (
      <Panel index="3.7" title="The record" lead={<>Provenance of every number above.</>}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const stale = spec && spec.code_hash !== res.code_hash
  return (
    <Panel index="3.7" title="The record" lead={<>Provenance of every number above.</>}>
      <table className="facts">
        <tbody>
          <tr>
            <td>Run</td>
            <td style={{ fontFamily: 'var(--mono)' }}>
              {res.run_id} · {res.created_at_iso?.replace('T', ' ')} → {res.completed_at_iso?.replace('T', ' ')}
            </td>
          </tr>
          <tr>
            <td>Model</td>
            <td>
              <span style={{ fontFamily: 'var(--mono)' }}>{res.model_version}</span> — paper's Smoluchowski equation, cubic U₀, η from Γ₀ · code <span style={{ fontFamily: 'var(--mono)' }}>{res.code_hash}</span>
              {stale && <span className="tag tag--prototype">code has changed since this run</span>}
            </td>
          </tr>
          <tr>
            <td>Parameters</td>
            <td>
              {res.parameter_provenance.label}
              {res.parameter_provenance.inverse_run_id && <> (inverse run <span style={{ fontFamily: 'var(--mono)' }}>{res.parameter_provenance.inverse_run_id}</span>)</>}
              {res.parameter_provenance.temperature_note && <> · {res.parameter_provenance.temperature_note}</>}
            </td>
          </tr>
          <tr>
            <td>Numerics</td>
            <td className="num">
              {res.grid.n_cells} cells, h = {fixed(res.grid.h_A, 3)} Å · {res.config.n_steps} steps, Δt = {sci(res.dt_ps, 2)} ps · θ = {res.config.theta} · solve {fixed(res.solver_duration_s, 1)} s
            </td>
          </tr>
          <tr>
            <td>Files</td>
            <td style={{ fontFamily: 'var(--mono)', fontSize: 12 }}>
              {Object.values(res.phase_files).map((f) => (
                <div key={f}>artifacts/{f}</div>
              ))}
            </td>
          </tr>
        </tbody>
      </table>
      <button className="btn btn--quiet" style={{ marginTop: 16 }} onClick={() => setShow((s) => !s)}>
        {show ? 'hide' : 'show'} configuration
      </button>
      {show && <pre className="code-block">{JSON.stringify(res.config, null, 2)}</pre>}
    </Panel>
  )
}
