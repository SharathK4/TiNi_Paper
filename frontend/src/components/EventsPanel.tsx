import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, ReferenceLine, ResponsiveContainer, Tooltip } from 'recharts'
import { api, type Events } from '../api/client'
import { useDebounced, useResource } from '../api/hooks'
import { fixed } from '../lib/format'
import { M } from './Math'
import { Empty, Figure, Grid, Panel, Readout, Slider, Tip, XNum, YNum, chart } from './ui'

function hist(h: { counts: number[]; edges: number[] } | undefined) {
  if (!h) return []
  return h.counts.map((n, i) => ({ x: (h.edges[i] + h.edges[i + 1]) / 2, n }))
}

export function EventsPanel({ version, defaultSmoothing, fc, onEvents }: { version: number; defaultSmoothing: number; fc: number | null; onEvents: (e: Events | null) => void }) {
  const [smoothing, setSmoothing] = useState(defaultSmoothing)
  const s = useDebounced(smoothing, 150)
  const ev = useResource<Events>(() => api.events(s), [s], version)

  // Hoist the default-smoothing events to the page so section 02 can mark peaks.
  const isDefault = s === defaultSmoothing
  useEffect(() => {
    if (isDefault) onEvents(ev.data)
  }, [ev.data, isDefault, onEvents])

  const tHist = useMemo(() => hist(ev.data?.histograms.t_peak_ps), [ev.data])
  const fHist = useMemo(() => hist(ev.data?.histograms.f_crit_from_rate_pN), [ev.data])

  const lead = (
    <>
      When does each run snap? The paper takes the time of the peak force before the drop, <M tex="t^\ddagger" />, and converts it to a critical force by{' '}
      <M tex="f^\ddagger = \dot f\, t^\ddagger" />. The paper does not say how it finds the peak in a noisy trace, so the picker here is a prototype.
    </>
  )

  if (ev.missing) {
    return (
      <Panel index="1.4" title="Transition events" lead={lead}>
        <Empty>Run the pipeline to detect transition events.</Empty>
      </Panel>
    )
  }
  const d = ev.data

  return (
    <Panel index="1.4" title="Transition events" lead={lead}>
      <div className="controls">
        <Slider label="peak-picker smoothing" value={smoothing} min={1} max={61} step={2} onChange={setSmoothing} format={(v) => `${v} pts · ${(v * 0.64).toFixed(1)} ps`} />
        <span className="tag tag--prototype" style={{ marginLeft: 0, alignSelf: 'center' }}>
          prototype assumption
        </span>
      </div>

      <div className="grid-2">
        <Figure title={<>transition time <M tex="t^\ddagger" /> across traces</>} note="Centred moving average, then the argmax. Wider smoothing reduces noise-triggered early picks but blurs the peak.">
          <div className="chart">
            <ResponsiveContainer>
              <BarChart data={tHist} margin={chart.margin} barCategoryGap={2}>
                <Grid />
                <XNum dataKey="x" label="t‡ [ps]" />
                <YNum label="traces" width={40} />
                <Tooltip content={<Tip unitX="ps" fmtX={(v) => v.toFixed(0)} rows={(p) => `${p.value} traces`} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="n" fill={chart.ink} isAnimationActive={false} radius={[3, 3, 0, 0]} />
                {d && <ReferenceLine x={d.t_peak_ps.mean} stroke={chart.accent} strokeDasharray="3 3" />}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure
          title={<>critical force <M tex="f^\ddagger = \dot f\, t^\ddagger" /></>}
          note={
            fc !== null ? (
              <>
                Dashed: mean. Every event lies below <M tex="f_c" /> = {fixed(fc, 1)} pN, the force at which the paper's landscape barrier vanishes (section 1.5).
              </>
            ) : (
              'Dashed: mean.'
            )
          }
        >
          <div className="chart">
            <ResponsiveContainer>
              <BarChart data={fHist} margin={chart.margin} barCategoryGap={2}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" domain={[0, fc ? Math.ceil(fc * 1.1) : 'dataMax']} />
                <YNum label="traces" width={40} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.value} traces`} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="n" fill={chart.accent} isAnimationActive={false} radius={[3, 3, 0, 0]} />
                {d && <ReferenceLine x={d.f_crit_from_rate_pN.mean} stroke={chart.ink} strokeDasharray="3 3" />}
                {fc !== null && <ReferenceLine x={fc} stroke={chart.model} label={{ value: 'fc', position: 'top', fontSize: 11, fill: chart.model }} />}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      {d && (
        <div className="readouts">
          <Readout label="events" value={d.n} />
          <Readout label={<M tex="\langle t^\ddagger\rangle" />} value={`${fixed(d.t_peak_ps.mean, 1)} ± ${fixed(d.t_peak_ps.std, 1)}`} unit="ps" />
          <Readout label={<><M tex="\langle f^\ddagger\rangle" /> from <M tex="\dot f\,t^\ddagger" /></>} value={`${fixed(d.f_crit_from_rate_pN.mean, 2)} ± ${fixed(d.f_crit_from_rate_pN.std, 2)}`} unit="pN" tone="accent" />
          <Readout label="smoothed peak" value={`${fixed(d.f_peak_smoothed_pN.mean, 2)} ± ${fixed(d.f_peak_smoothed_pN.std, 2)}`} unit="pN" />
          <Readout label="raw peak" value={`${fixed(d.f_peak_raw_pN.mean, 2)} ± ${fixed(d.f_peak_raw_pN.std, 2)}`} unit="pN" />
          <Readout label={<><M tex="\dot f" /> used</>} value={fixed(d.force_rate_used_pN_per_ps, 4)} unit="pN/ps" />
        </div>
      )}
      <p className="prose" style={{ marginTop: 18 }}>
        Three readings of the same peak: <M tex="\dot f\,t^\ddagger" /> is the paper's definition (the ramp value at the transition time); the smoothed peak is what
        the picker actually saw; the raw peak is the largest single noisy sample and overstates the force. The distribution <M tex="p(f^\ddagger\,|\,\dot f)" /> and
        its fit belong to Version 2 and are not attempted here.
      </p>
    </Panel>
  )
}
