import { useEffect, useMemo, useState } from 'react'
import { ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, YAxis } from 'recharts'
import { inverse, type InvModel, type InvSample, type InvSpec } from '../../api/inverse'
import { useDebounced, useResource } from '../../api/hooks'
import { fixed, sci } from '../../lib/format'
import { Eq, M } from '../Math'
import { Figure, Grid, Panel, Readout, Seg, Slider, Tip, XNum, YNum, chart } from '../ui'
import { XLog } from './shared'

/* C. The paper's density, explorable; the force spectrum as a forward check. */
export function ModelPanel({ spec, sample, fitted }: { spec: InvSpec | null; sample: InvSample | null; fitted: { U_barrier: number; xi_barrier: number; gamma0: number } | null }) {
  const sets = spec?.paper_parameter_sets ?? {}
  const [preset, setPreset] = useState<string>('pdf_mle')
  const [U, setU] = useState(638.14)
  const [xi, setXi] = useState(29.2)
  const [logG, setLogG] = useState(Math.log10(3.96e-7))

  useEffect(() => {
    if (preset === 'fitted' && fitted) {
      setU(fitted.U_barrier)
      setXi(fitted.xi_barrier)
      setLogG(Math.log10(fitted.gamma0))
    } else if (sets[preset]) {
      setU(sets[preset].U_barrier_pN_A)
      setXi(sets[preset].xi_barrier_A)
      setLogG(Math.log10(sets[preset].gamma0_per_ps))
    }
  }, [preset, sets, fitted])

  const dU = useDebounced(U, 200)
  const dXi = useDebounced(xi, 200)
  const dG = useDebounced(logG, 200)
  const fdot = sample?.force_rate_pN_per_ps
  const model = useResource<InvModel>(() => inverse.model(dU, dXi, Math.pow(10, dG), fdot), [dU, dXi, dG, fdot])
  const m = model.data

  const curves = useMemo(() => (m ? m.f_pN.map((f, i) => ({ f, p: m.pdf[i], psi: m.survival[i] })) : []), [m])
  const spectrum = useMemo(() => (m ? m.spectrum.map((r) => ({ x: r.force_rate, eq8: r.mean_eq8, exact: r.mean_exact })) : []), [m])
  const fig3 = useMemo(() => (spec ? spec.fig3_readings_by_eye.map((r) => ({ x: r.force_rate_pN_per_ps, y: r.mean_f_pN, e: r.uncertainty_pN })) : []), [spec])
  const ours = useMemo(() => (sample ? [{ x: sample.force_rate_pN_per_ps, y: sample.mean_pN, e: sample.sd_pN }] : []), [sample])
  const custom = (setter: (v: number) => void) => (v: number) => {
    setPreset('custom')
    setter(v)
  }

  return (
    <Panel
      index="C"
      title="The model"
      lead={<>A crystal waits in the austenite well while the applied force lowers the barrier; the paper solves that race in closed form.</>}
    >
      <div className="grid-2" style={{ alignItems: 'center' }}>
        <div className="prose">
          <Eq tag="Eq. (4) · rate under force" tex="\Gamma(f) = \Gamma_0\left(1-\tfrac{f}{f_c}\right)^{1/2} e^{\,U^\ddagger\Delta(f)/k_BT},\qquad \Delta(f) = 1-\left(1-\tfrac{f}{f_c}\right)^{3/2}" />
          <Eq tag="Eq. (5) · survival" tex="\psi(f) = \exp\!\left\{\mu_0\left[1-e^{U^\ddagger\Delta(f)/k_BT}\right]\right\},\qquad \mu_0 = \frac{\Gamma_0\,k_BT}{\dot f\,\xi^\ddagger}" />
          <Eq tag="Eq. (6) · the density that is fitted" tex="p(f\mid\dot f) = \frac{\Gamma(f)}{\dot f}\,\psi(f)" />
          <p style={{ marginTop: 18 }}>
            <M tex="f_c" /> sets where the density ends, <M tex="U^\ddagger/k_BT" /> how steeply the rate climbs, <M tex="\mu_0" /> how early escape happens. Only{' '}
            <M tex="\mu_0" /> depends on the loading rate — which is why one rate cannot separate all three.
          </p>
        </div>
        <Figure title={<><M tex="p(f\mid\dot f)" /> and <M tex="\psi(f)" /> at ḟ = {m ? fixed(m.force_rate_pN_per_ps, 4) : '—'} pN/ps</>}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={curves} margin={chart.margin}>
                <Grid />
                <XNum dataKey="f" label="f [pN]" range={[0, m ? Math.ceil(m.parameters.f_c_pN / 5) * 5 : 40]} />
                <YNum label="p [1/pN]" />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => (p.dataKey === 'p' ? `p = ${typeof p.value === 'number' ? p.value.toFixed(4) : ''}` : `ψ = ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`)} />} />
                <Line dataKey="p" type="monotone" stroke={chart.model} strokeWidth={1.6} dot={false} isAnimationActive={false} />
                <Line dataKey="psi" type="monotone" stroke={chart.ink3} strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} yAxisId="r" />
                <YAxis yAxisId="r" orientation="right" type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} width={36} tick={{ fontSize: 11, fill: '#9698a8' }} tickLine={false} axisLine={false} label={{ value: 'ψ', angle: 90, position: 'insideRight', fontSize: 11, fill: '#6e7180' }} />
                {m && <ReferenceLine x={m.mean_exact_pN} stroke={chart.model2} strokeDasharray="2 3" label={{ value: '⟨f⟩', position: 'top', fontSize: 11, fill: chart.model2 }} />}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <div className="controls" style={{ marginTop: 44 }}>
        <div className="control">
          <span className="control__label">
            <span>parameters</span>
          </span>
          <Seg
            value={preset}
            onChange={setPreset}
            options={[
              { value: 'pdf_mle', label: 'paper, p(f|ḟ)' },
              { value: 'mean_lsq', label: 'paper, ⟨f⟩(ḟ)' },
              ...(fitted ? [{ value: 'fitted', label: 'this fit' }] : []),
            ]}
          />
        </div>
        <Slider label="U‡" value={U} min={200} max={1500} step={1} onChange={custom(setU)} format={(v) => `${v.toFixed(0)} pN·Å`} />
        <Slider label="ξ‡" value={xi} min={5} max={80} step={0.1} onChange={custom(setXi)} format={(v) => `${v.toFixed(1)} Å`} />
        <Slider label="Γ₀" value={logG} min={-12} max={-2} step={0.05} onChange={custom(setLogG)} format={(v) => `${sci(Math.pow(10, v), 1)} ps⁻¹`} />
      </div>
      {m && (
        <div className="readouts" style={{ marginTop: 0 }}>
          <Readout label={<M tex="f_c" />} value={fixed(m.parameters.f_c_pN, 1)} unit="pN" tone="model" />
          <Readout label={<M tex="U^\ddagger/k_BT" />} value={fixed(m.parameters.U_barrier_kBT, 1)} tone="model" />
          <Readout label={<M tex="\mu_0" />} value={sci(m.parameters.mu0, 1)} tone="model" />
          <Readout label="⟨f⟩ · Eq. 8 / exact" value={`${fixed(m.mean_eq8_pN, 1)} / ${fixed(m.mean_exact_pN, 1)}`} unit="pN" />
          <Readout label={<M tex="\psi(f_c)" />} value={sci(m.residual_mass, 0)} />
        </div>
      )}

      <div style={{ marginTop: 64 }}>
        <Figure
          title={<>force spectrum <M tex="\langle f^\ddagger\rangle(\dot f)" /> · Eq. (8), evaluated forward</>}
          note={<>Grey: the paper's Fig. 3 symbols read by eye, ± 1 pN. Terracotta: this dataset, mean ± s.d. Dotted: the exact mean where Eq. (8)'s series drifts.</>}
        >
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={spectrum} margin={chart.margin}>
                <Grid />
                <XLog dataKey="x" label="ḟ [pN/ps]" lo={2e-3} hi={2} />
                <YNum label="⟨f‡⟩ [pN]" range={[5, 35]} />
                <Tooltip content={<Tip unitX="pN/ps" fmtX={(v) => sci(v, 2)} rows={(p) => `${p.name ?? p.dataKey}: ${typeof p.value === 'number' ? p.value.toFixed(2) : ''} pN`} />} />
                <Line dataKey="eq8" name="Eq. 8" type="monotone" stroke={chart.model} strokeWidth={1.6} dot={false} isAnimationActive={false} />
                <Line dataKey="exact" name="exact" type="monotone" stroke={chart.model} strokeWidth={1} strokeDasharray="2 3" dot={false} isAnimationActive={false} />
                <Scatter data={fig3} dataKey="y" name="Fig. 3" fill={chart.ink3} isAnimationActive={false}>
                  <ErrorBar dataKey="e" width={3} stroke={chart.ink3} direction="y" />
                </Scatter>
                <Scatter data={ours} dataKey="y" name="this dataset" fill={chart.accent} isAnimationActive={false}>
                  <ErrorBar dataKey="e" width={4} stroke={chart.accent} direction="y" />
                </Scatter>
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>
    </Panel>
  )
}
