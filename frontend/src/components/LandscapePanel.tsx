import { useEffect, useMemo, useState } from 'react'
import { ComposedChart, Line, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts'
import { api, type ParamSet, type Potential } from '../api/client'
import { useDebounced, useResource } from '../api/hooks'
import { fixed } from '../lib/format'
import { KBT, U0, barrier, curves, fc as fcOf, stationary } from '../lib/physics'
import { Eq, M } from './Math'
import { Figure, Grid, Panel, Readout, Seg, Slider, Tip, XNum, YNum, chart } from './ui'

/* The cubic landscape explorer. The curves are computed in the browser so the
 * sliders answer immediately; the backend evaluates the same formulas
 * (debounced) and the maximum discrepancy is shown as a live check. */
export function LandscapePanel({ paramSets, defaultSet, forceRate, meanCritForce, onFc }: { paramSets: Record<string, ParamSet>; defaultSet: string; forceRate: number | null; meanCritForce: number | null; onFc: (fc: number) => void }) {
  const [setKey, setSetKey] = useState(defaultSet)
  const ps = paramSets[setKey]
  const [U, setU] = useState(ps?.U_barrier_pN_A ?? 638.14)
  const [xi, setXi] = useState(ps?.xi_barrier_A ?? 29.2)
  const [fFrac, setFFrac] = useState(0)
  const [energyUnit, setEnergyUnit] = useState<'pNA' | 'kBT'>('pNA')

  useEffect(() => {
    if (ps) {
      setU(ps.U_barrier_pN_A)
      setXi(ps.xi_barrier_A)
    }
  }, [ps])

  const land = useMemo(() => ({ U, xi }), [U, xi])
  const fc = fcOf(land)
  const f = fFrac * fc
  useEffect(() => onFc(fc), [fc, onFc])

  const data = useMemo(() => curves(land, f), [land, f])
  const sp = stationary(land, f)
  const uf = barrier(land, f)
  const scale = energyUnit === 'kBT' ? 1 / KBT : 1
  const unit = energyUnit === 'kBT' ? 'k_BT' : 'pN·Å'

  const dU = useDebounced(U, 250)
  const dXi = useDebounced(xi, 250)
  const dF = useDebounced(f, 250)
  const remote = useResource<Potential>(() => api.potential(dU, dXi, dF), [dU, dXi, dF])
  const crossCheck = useMemo(() => {
    const r = remote.data
    if (!r || r.parameters.U_barrier_pN_A !== dU || r.parameters.xi_barrier_A !== dXi) return null
    const local = curves({ U: dU, xi: dXi }, dF, r.xi_A.length)
    let maxU = 0
    for (let i = 0; i < r.xi_A.length; i++) maxU = Math.max(maxU, Math.abs(local[i].U0 - r.U0_pN_A[i]))
    return { maxU, forceCheck: r.force_check_max_abs_pN }
  }, [remote.data, dU, dXi, dF])

  const scaled = useMemo(() => data.map((d) => ({ ...d, U0s: d.U0 * scale, Ufs: d.Uf * scale })), [data, scale])
  const xiRange: [number, number] = [data[0].xi, data[data.length - 1].xi]
  const tAtF = forceRate ? f / forceRate : null
  const isPaper = ps && U === ps.U_barrier_pN_A && xi === ps.xi_barrier_A

  return (
    <Panel
      index="05"
      title="The landscape"
      lead={
        <>
          The paper's model: the crystal sits in a well (austenite), separated by a barrier of height <M tex="U^\ddagger" /> at distance <M tex="\xi^\ddagger" /> from
          the twinned state. Pulling on the box tilts the landscape by <M tex="-f\,\xi" /> until, at <M tex="f_c" />, the barrier is gone.
        </>
      }
    >
      <div className="controls">
        <div className="control">
          <span className="control__label">
            <span>parameters</span>
          </span>
          <Seg
            value={setKey}
            onChange={setSetKey}
            options={Object.entries(paramSets).map(([k, v]) => ({ value: k, label: v.label.split('·')[1]?.split(',')[0].trim() ?? k }))}
          />
        </div>
        <Slider label="barrier U‡" value={U} min={200} max={1200} step={1} onChange={setU} format={(v) => `${v.toFixed(0)} pN·Å · ${(v / KBT).toFixed(1)} kBT`} />
        <Slider label="barrier position ξ‡" value={xi} min={5} max={60} step={0.1} onChange={setXi} format={(v) => `${v.toFixed(1)} Å`} />
        <Slider label="applied force f" value={fFrac} min={0} max={1.1} step={0.005} onChange={setFFrac} format={(v) => `${(v * fc).toFixed(1)} pN · ${(100 * v).toFixed(0)} % of f_c`} />
        <div className="control">
          <span className="control__label">
            <span>energy unit</span>
          </span>
          <Seg
            value={energyUnit}
            onChange={setEnergyUnit}
            options={[
              { value: 'pNA', label: 'pN·Å' },
              { value: 'kBT', label: 'kBT' },
            ]}
          />
        </div>
        {ps && !isPaper && (
          <button
            className="btn btn--quiet"
            onClick={() => {
              setU(ps.U_barrier_pN_A)
              setXi(ps.xi_barrier_A)
            }}
          >
            reset to Table 1
          </button>
        )}
      </div>

      <div className="grid-2">
        <Figure
          title={
            <>
              free energy · <M tex="U_0(\xi)" /> and the tilted <M tex="U(\xi,t) = U_0(\xi) - f\,\xi" />
            </>
          }
          note={
            <>
              Slate: the equilibrium profile. Terracotta: tilted by the current force{tAtF !== null ? `, which the mean ramp reaches at t = ${fixed(tAtF, 1)} ps (with the ḟ chosen in section 03)` : ''}.
              Dots mark the moving minimum and transition state; they meet at <M tex="f = f_c" />.
            </>
          }
        >
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={scaled} margin={chart.margin}>
                <Grid />
                <XNum dataKey="xi" label="ξ [Å]" range={xiRange} />
                <YNum label={`U [${energyUnit === 'kBT' ? 'kBT' : 'pN·Å'}]`} width={60} />
                <Tooltip
                  content={
                    <Tip
                      unitX="Å"
                      fmtX={(v) => v.toFixed(2)}
                      rows={(p) => `${p.dataKey === 'U0s' ? 'U₀' : 'U − fξ'} = ${typeof p.value === 'number' ? p.value.toFixed(energyUnit === 'kBT' ? 2 : 1) : '—'} ${energyUnit === 'kBT' ? 'kBT' : 'pN·Å'}`}
                    />
                  }
                />
                <ReferenceLine x={0} stroke={chart.grid} />
                <ReferenceLine y={0} stroke={chart.grid} />
                <Line type="monotone" dataKey="U0s" stroke={chart.model} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                {f > 0 && <Line type="monotone" dataKey="Ufs" stroke={chart.accent} strokeWidth={1.5} dot={false} isAnimationActive={false} />}
                {sp && (
                  <>
                    <ReferenceDot x={sp.xiMin} y={(U0(land, sp.xiMin) - f * sp.xiMin) * scale} r={4} fill={chart.accent} stroke="#fff" strokeWidth={1.5} />
                    <ReferenceDot x={sp.xiTs} y={(U0(land, sp.xiTs) - f * sp.xiTs) * scale} r={4} fill={chart.accent} stroke="#fff" strokeWidth={1.5} />
                  </>
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>

        <Figure
          title={
            <>
              landscape force · <M tex="F(\xi) = -\mathrm{d}U_0/\mathrm{d}\xi" />
            </>
          }
          note={
            <>
              The force the landscape exerts on the order parameter. It is zero at the well and at the barrier top, and most negative, <M tex="-f_c" />, at the
              inflection midway. The tilted curve is <M tex="F + f" />; where it crosses zero are the dots on the left.
            </>
          }
        >
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={chart.margin}>
                <Grid />
                <XNum dataKey="xi" label="ξ [Å]" range={xiRange} />
                <YNum label="F [pN]" />
                <Tooltip content={<Tip unitX="Å" fmtX={(v) => v.toFixed(2)} rows={(p) => `${p.dataKey === 'F' ? 'F' : 'F + f'} = ${typeof p.value === 'number' ? p.value.toFixed(2) : '—'} pN`} />} />
                <ReferenceLine y={0} stroke={chart.ink4} />
                <ReferenceLine y={-fc} stroke={chart.model2} strokeDasharray="4 3" label={{ value: `−f_c = −${fc.toFixed(1)} pN`, position: 'insideBottomRight', fontSize: 11, fill: chart.model2 }} />
                <Line type="monotone" dataKey="F" stroke={chart.model} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                {f > 0 && <Line type="monotone" dataKey="Ff" stroke={chart.accent} strokeWidth={1.5} dot={false} isAnimationActive={false} />}
                {sp && (
                  <>
                    <ReferenceDot x={sp.xiMin} y={0} r={4} fill={chart.accent} stroke="#fff" strokeWidth={1.5} />
                    <ReferenceDot x={sp.xiTs} y={0} r={4} fill={chart.accent} stroke="#fff" strokeWidth={1.5} />
                  </>
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <div className="readouts">
        <Readout label={<M tex="f_c = 3U^\ddagger/2\xi^\ddagger" />} value={fixed(fc, 2)} unit="pN" tone="model" />
        <Readout label="barrier at this f" value={fixed(uf * scale, energyUnit === 'kBT' ? 2 : 1)} unit={unit} tone="model" />
        <Readout label="fraction left" value={`${(100 * uf / U).toFixed(1)} %`} />
        <Readout label={<><M tex="\xi_{\min}" /> minimum</>} value={sp ? fixed(sp.xiMin, 2) : '—'} unit="Å" />
        <Readout label={<><M tex="\xi_{TS}" /> transition state</>} value={sp ? fixed(sp.xiTs, 2) : '—'} unit="Å" />
        <Readout label="curvature at well" value={fixed((6 * U) / xi ** 2, 2)} unit="pN/Å" />
        {meanCritForce !== null && <Readout label="⟨f‡⟩ observed (04)" value={fixed(meanCritForce, 1)} unit={`pN · ${((100 * meanCritForce) / fc).toFixed(0)} % of f_c`} tone="accent" />}
        {crossCheck && (
          <Readout label="backend agreement" value={crossCheck.maxU < 1e-3 ? 'exact' : crossCheck.maxU.toExponential(1)} unit={`· dU/dξ check ${crossCheck.forceCheck.toExponential(1)} pN`} />
        )}
      </div>

      <div className="grid-2" style={{ marginTop: 30, alignItems: 'start' }}>
        <div className="prose">
          <Eq tag="paper · cubic free-energy profile · verified" tex="U_0(\xi) = \frac{U^\ddagger}{2} + \frac{3U^\ddagger}{2\xi^\ddagger}\left(\xi - \frac{\xi^\ddagger}{2}\right) - \frac{2U^\ddagger}{(\xi^\ddagger)^3}\left(\xi - \frac{\xi^\ddagger}{2}\right)^3" />
          <p>
            Built so that <M tex="U_0(0)=0" /> is the austenite minimum and <M tex="U_0(\xi^\ddagger)=U^\ddagger" /> the barrier top; both are stationary points. The
            tests in <code>backend/tests/test_potential.py</code> check exactly this.
          </p>
          <Eq tag="paper · tilt under the imposed force · verified" tex="U(\xi,t) \approx U_0(\xi) - \langle f(t)\rangle\,\xi, \qquad \langle f(t)\rangle \approx \dot f\,t" />
          <Eq tag="paper · maximal turnover force · verified" tex="f_c \equiv \frac{3U^\ddagger}{2\xi^\ddagger}" />
        </div>
        <div className="prose">
          <Eq tag="derived · landscape force" tex="F(\xi) = -\frac{\mathrm{d}U_0}{\mathrm{d}\xi} = -\frac{3U^\ddagger}{2\xi^\ddagger} + \frac{6U^\ddagger}{(\xi^\ddagger)^3}\left(\xi - \frac{\xi^\ddagger}{2}\right)^2" />
          <p>
            One derivative of the paper's cubic, checked against a numerical derivative on every request (the "dU/dξ check" above is the largest disagreement, from
            finite differences only).
          </p>
          <Eq tag="derived · barrier and stationary points under force · checked numerically" tex="U^\ddagger(f) = U^\ddagger\left(1-\frac{f}{f_c}\right)^{3/2}, \qquad \xi_{\min},\,\xi_{TS} = \frac{\xi^\ddagger}{2} \mp \frac{\xi^\ddagger}{2}\sqrt{1-\frac{f}{f_c}}" />
          <p>
            The first expression is the paper's Eq. (4) rearranged; the second follows from setting <M tex="F(\xi) + f = 0" />. The minimum moves right as the force
            grows — the paper's explanation for why faster loading gives larger critical forces.
          </p>
        </div>
      </div>
    </Panel>
  )
}
