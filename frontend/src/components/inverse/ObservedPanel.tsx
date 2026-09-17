import { useEffect, useMemo, useState } from 'react'
import { Bar, ComposedChart, ErrorBar, Line, ResponsiveContainer, Tooltip } from 'recharts'
import { inverse, type InvSample } from '../../api/inverse'
import { useDebounced, useResource } from '../../api/hooks'
import { fixed } from '../../lib/format'
import { M } from '../Math'
import { Empty, Figure, Grid, Panel, Readout, Seg, Slider, Tip, XNum, YNum, chart } from '../ui'

/* B. The sample: one critical force per trace, two density estimators. */
export function ObservedPanel({ version, defaults, onSample }: { version: number; defaults: { smoothing: number; window: number }; onSample: (s: InvSample | null) => void }) {
  const [source, setSource] = useState<'rate_x_time' | 'smoothed_peak'>('rate_x_time')
  const [smoothing, setSmoothing] = useState(defaults.smoothing)
  const [window, setWindow] = useState(defaults.window)
  const sm = useDebounced(smoothing, 150)
  const w = useDebounced(window, 150)
  const smp = useResource<InvSample>(() => inverse.sample(source, sm, w), [source, sm, w], version)
  const s = smp.data
  useEffect(() => onSample(s), [s, onSample])

  const hist = useMemo(() => (s ? s.histogram.centers.map((x, i) => ({ x, d: s.histogram.density[i], err: s.histogram.density_err[i] })) : []), [s])
  const kde = useMemo(() => (s ? s.kde.grid.map((x, i) => ({ x, k: s.kde.density[i] })) : []), [s])
  const ecdf = useMemo(() => (s ? s.ecdf.x.map((x, i) => ({ x, F: s.ecdf.F[i] })) : []), [s])
  const xRange: [number, number] = s ? [Math.floor(s.min_pN / 5) * 5 - 5, Math.ceil(s.max_pN / 5) * 5 + 5] : [0, 40]

  const lead = <>One critical force per trace. A histogram and a kernel estimate are drawn together so the shape is not one estimator's artefact.</>
  if (smp.missing) {
    return (
      <Panel index="B" title="What was observed" lead={lead}>
        <Empty>Run the Version 1 pipeline first.</Empty>
      </Panel>
    )
  }

  return (
    <Panel index="B" title="What was observed" lead={lead}>
      <div className="controls">
        <div className="control">
          <span className="control__label">
            <span>f‡ defined as</span>
          </span>
          <Seg
            value={source}
            onChange={setSource}
            options={[
              { value: 'rate_x_time', label: 'ḟ · t‡ (paper)' },
              { value: 'smoothed_peak', label: 'smoothed peak' },
            ]}
          />
        </div>
        <Slider label="peak-picker smoothing" value={smoothing} min={1} max={61} step={2} onChange={setSmoothing} format={(v) => `${v} pts`} />
        <Slider label="ḟ fit window" value={window} min={25} max={200} step={5} onChange={setWindow} format={(v) => `${v} ps`} />
      </div>

      <div className="grid-2">
        <Figure title={<>density of <M tex="f^\ddagger" /></>} note={s ? `Bars ± √n, ${s.histogram.bins} bins; kernel bandwidth ${fixed(s.kde.bandwidth_pN, 2)} pN.` : undefined}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={hist} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xRange} />
                <YNum label="density [1/pN]" />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => (p.dataKey === 'd' ? `histogram ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}` : p.dataKey === 'k' ? `kernel ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}` : null)} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="d" fill={chart.accent} fillOpacity={0.5} isAnimationActive={false} radius={[2, 2, 0, 0]}>
                  <ErrorBar dataKey="err" width={3} stroke={chart.accent} strokeWidth={1} />
                </Bar>
                <Line data={kde} dataKey="k" type="monotone" stroke={chart.ink} strokeWidth={1.4} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure title="cumulative">
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={ecdf} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xRange} />
                <YNum label="P(f ≤ f‡)" range={[0, 1]} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(2)} rows={(p) => `F = ${typeof p.value === 'number' ? p.value.toFixed(2) : ''}`} />} />
                <Line dataKey="F" type="stepAfter" stroke={chart.accent} strokeWidth={1.4} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      {s && (
        <div className="readouts">
          <Readout label="events" value={s.n} />
          <Readout label="mean" value={`${fixed(s.mean_pN, 2)} ± ${fixed(s.sem_pN, 2)}`} unit="pN" tone="accent" />
          <Readout label="s.d." value={fixed(s.sd_pN, 2)} unit="pN" />
          <Readout label="skewness" value={fixed(s.skewness, 2)} />
          <Readout label="ḟ" value={fixed(s.force_rate_pN_per_ps, 4)} unit="pN/ps" />
        </div>
      )}
    </Panel>
  )
}
