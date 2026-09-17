import { useEffect, useMemo, useState } from 'react'
import { Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts'
import { api, type Average } from '../api/client'
import { useDebounced, useResource } from '../api/hooks'
import { fixed } from '../lib/format'
import { Eq, M } from './Math'
import { Empty, Figure, Grid, Panel, Readout, Slider, Tip, XNum, YNum, chart } from './ui'

export function AveragePanel({ version, defaultWindow, paperRate, onRate }: { version: number; defaultWindow: number; paperRate: number | null; onRate: (r: number) => void }) {
  const [window, setWindow] = useState(defaultWindow)
  const w = useDebounced(window, 120)
  const avg = useResource<Average>(() => api.average(w), [w], version)

  const data = useMemo(() => {
    const a = avg.data
    if (!a) return []
    const fitAt = new Map(a.force_rate.t_fit.map((t, i) => [t, a.force_rate.f_fit[i]]))
    return a.time_ps.map((t, i) => ({
      t,
      x: a.mean_x_A[i],
      mean: a.mean_f_pN[i],
      lo: a.mean_f_pN[i] - a.std_f_pN[i],
      hi: a.mean_f_pN[i] + a.std_f_pN[i],
      band: [a.mean_f_pN[i] - a.std_f_pN[i], a.mean_f_pN[i] + a.std_f_pN[i]],
      iqr: [a.p25_f_pN[i], a.p75_f_pN[i]],
      fit: fitAt.get(t),
    }))
  }, [avg.data])

  const tRange: [number, number] = [0, avg.data ? avg.data.time_ps[avg.data.time_ps.length - 1] : 640]
  const fr = avg.data?.force_rate
  const slope = fr?.slope_pN_per_ps
  useEffect(() => {
    if (slope !== undefined) onRate(slope)
  }, [slope, onRate])

  const lead = (
    <>
      Average the hundred traces at each instant and the noise cancels: what remains is the mean force ramp <M tex="\langle f(t)\rangle" />. Its early slope is the
      force rate <M tex="\dot f" />, the one number that links the simulation to the model.
    </>
  )

  if (avg.missing) {
    return (
      <Panel index="03" title="The average" lead={lead}>
        <Empty>Run the pipeline to compute the ensemble average.</Empty>
      </Panel>
    )
  }

  return (
    <Panel index="03" title="The average" lead={lead}>
      <div className="controls">
        <Slider label="fit window for ḟ" value={window} min={10} max={300} step={5} onChange={setWindow} format={(v) => `t ≤ ${v} ps`} />
      </div>

      <div className="grid-2">
        <Figure
          title="mean force with ± 1 s.d. band"
          note="Dashed line: least-squares fit on the window. Past ~150 ps the first traces have already transitioned and the mean bends over — which is why the window has to stop before that."
        >
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={chart.margin}>
                <Grid />
                <XNum dataKey="t" label="t [ps]" range={tRange} />
                <YNum label="⟨f⟩ [pN]" />
                <Tooltip
                  content={
                    <Tip
                      unitX="ps"
                      fmtX={(v) => v.toFixed(2)}
                      rows={(p) => {
                        if (p.dataKey === 'band') return null
                        const v = typeof p.value === 'number' ? p.value.toFixed(2) : '—'
                        return p.dataKey === 'mean' ? `⟨f⟩ = ${v} pN` : p.dataKey === 'fit' ? `fit = ${v} pN` : null
                      }}
                    />
                  }
                />
                <Area dataKey="band" stroke="none" fill={chart.accent} fillOpacity={0.12} isAnimationActive={false} />
                <Line type="linear" dataKey="mean" stroke={chart.accent} strokeWidth={1.4} dot={false} isAnimationActive={false} />
                <Line type="linear" dataKey="fit" stroke={chart.ink} strokeWidth={1.2} strokeDasharray="5 4" dot={false} isAnimationActive={false} connectNulls={false} />
                <ReferenceLine x={w} stroke={chart.ink3} strokeDasharray="2 3" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>

        <Figure title="mean force against mean box length" note={<>The same curve against the imposed deformation rather than time. <M tex="L_x" /> is the box length, not the landscape coordinate <M tex="\xi" /> of section 05.</>}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="⟨Lₓ⟩ [Å]" />
                <YNum label="⟨f⟩ [pN]" />
                <Tooltip content={<Tip unitX="Å" fmtX={(v) => v.toFixed(3)} rows={(p) => (p.dataKey === 'mean' ? `⟨f⟩ = ${typeof p.value === 'number' ? p.value.toFixed(2) : '—'} pN` : null)} />} />
                <Area dataKey="iqr" stroke="none" fill={chart.accent} fillOpacity={0.12} isAnimationActive={false} />
                <Line type="linear" dataKey="mean" stroke={chart.accent} strokeWidth={1.4} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      {fr && (
        <div className="readouts">
          <Readout label={<>force rate <M tex="\dot f" /></>} value={fixed(fr.slope_pN_per_ps, 4)} unit="pN/ps" tone="accent" />
          <Readout label="± s.e." value={fixed(fr.slope_se, 4)} unit="pN/ps" />
          <Readout label="R²" value={fixed(fr.r2, 3)} />
          <Readout label="points in fit" value={fr.n_points} />
          {paperRate !== null && <Readout label="paper, rate (v)" value={fixed(paperRate, 4)} unit="pN/ps" />}
          {paperRate !== null && <Readout label="ratio" value={fixed(fr.slope_pN_per_ps / paperRate, 4)} />}
          <Readout label="traces averaged" value={avg.data?.n_traces ?? '—'} />
        </div>
      )}

      <div className="grid-2" style={{ marginTop: 28, alignItems: 'start' }}>
        <div className="prose">
          <Eq tag="paper, Eq. (2) · verified" tex="\langle f(t)\rangle \approx \dot f\, t \equiv f(t)" />
          <p>
            The paper computes <M tex="\dot f" /> as the gradient of the averaged trace "in a small time interval near t = 0". How small is not stated, so the
            window is a <span className="tag tag--prototype" style={{ marginLeft: 0 }}>prototype assumption</span>; the table shows how much the answer moves with it.
          </p>
        </div>
        <div>
          <div className="figure__title">sensitivity of ḟ to the window</div>
          <table className="facts">
            <tbody>
              {avg.data?.sensitivity.map((s) => (
                <tr key={s.window_ps}>
                  <td>t ≤ {s.window_ps} ps</td>
                  <td>
                    {fixed(s.slope, 4)} ± {fixed(s.slope_se, 4)} pN/ps · R² {fixed(s.r2, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  )
}
