import { useMemo, useState } from 'react'
import { ComposedChart, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip } from 'recharts'
import { api, type Events, type Overlay, type Trace } from '../api/client'
import { useResource } from '../api/hooks'
import { fixed } from '../lib/format'
import { Empty, Figure, Grid, Panel, Readout, Seg, Slider, Tip, XNum, YNum, chart } from './ui'

export function TracePanel({ version, selected, onSelect, available, events }: { version: number; selected: string | null; onSelect: (id: string) => void; available: string[]; events: Events | null }) {
  const [smoothing, setSmoothing] = useState(15)
  const [yAxis, setYAxis] = useState<'time' | 'x'>('time')
  const [overlayN, setOverlayN] = useState(8)

  const id = selected ?? available[0] ?? null
  const trace = useResource<Trace>(() => (id ? api.trace(id, smoothing) : Promise.reject(new Error('no trace'))), [id, smoothing], version)
  const overlayIds = useMemo(() => available.slice(0, overlayN), [available, overlayN])
  const overlay = useResource<Overlay>(() => (overlayIds.length ? api.overlay(overlayIds, 2) : Promise.reject(new Error('none'))), [overlayIds.join(',')], version)

  const data = useMemo(() => {
    const t = trace.data
    if (!t) return []
    return t.time_ps.map((tt, i) => ({ t: tt, x: t.x_A[i], f: t.force_pN[i], fs: t.force_smoothed_pN?.[i] }))
  }, [trace.data])

  const overlayData = useMemo(() => {
    const o = overlay.data
    if (!o) return []
    return o.time_ps.map((t, i) => {
      const row: Record<string, number> = { t }
      o.traces.forEach((tr) => (row[tr.trace_id] = tr.force_pN[i]))
      return row
    })
  }, [overlay.data])

  const ev = useMemo(() => {
    if (!events || !id) return null
    const k = events.trace_ids.indexOf(id)
    return k >= 0 ? events.events[k] : null
  }, [events, id])

  const tRange: [number, number] = [0, trace.data ? trace.data.time_ps[trace.data.time_ps.length - 1] : 640]
  const idx = id ? available.indexOf(id) : -1
  const step = (d: number) => {
    if (idx < 0) return
    const n = (idx + d + available.length) % available.length
    onSelect(available[n])
  }

  const lead = (
    <>
      A single run: the force rises linearly as the box is stretched, then drops when the austenite lattice twins into martensite. The raw signal is thermal noise
      on top of that ramp; smoothing is only for reading it.
    </>
  )

  if (!available.length) {
    return (
      <Panel index="1.2" title="One trace" lead={lead}>
        <Empty>Traces appear here once the dataset is processed.</Empty>
      </Panel>
    )
  }

  return (
    <Panel index="1.2" title="One trace" lead={lead}>
      <div className="controls">
        <div className="control">
          <span className="control__label">
            <span>trace</span>
          </span>
          <span className="stepper">
            <button className="btn btn--quiet" onClick={() => step(-1)} aria-label="previous trace">
              ←
            </button>
            <span className="stepper__id">{id}</span>
            <button className="btn btn--quiet" onClick={() => step(1)} aria-label="next trace">
              →
            </button>
          </span>
        </div>
        <Slider label="smoothing window" value={smoothing} min={1} max={61} step={2} onChange={setSmoothing} format={(v) => `${v} pts · ${(v * 0.64).toFixed(1)} ps`} />
        <div className="control">
          <span className="control__label">
            <span>horizontal axis</span>
          </span>
          <Seg
            value={yAxis}
            onChange={setYAxis}
            options={[
              { value: 'time', label: 'time t' },
              { value: 'x', label: 'box length' },
            ]}
          />
        </div>
      </div>

      <Figure
        title={<>force per atom · trace <span className="nocase">{id}</span></>}
        note={
          ev ? (
            <>
              Marker: peak of the {events?.smoothing_points}-point smoothed trace at t‡ = {fixed(ev.t_peak_ps, 1)} ps (prototype picker, section 1.4). The force then falls to{' '}
              {fixed(ev.f_trough_pN, 1)} pN — the twinned lattice is briefly shorter than the box, so it is compressed, and then reloads.
            </>
          ) : undefined
        }
      >
        <div className="chart chart--tall">
          <ResponsiveContainer>
            <ComposedChart data={data} margin={chart.margin}>
              <Grid />
              <XNum dataKey={yAxis === 'time' ? 't' : 'x'} label={yAxis === 'time' ? 't [ps]' : 'box length [Å]'} range={yAxis === 'time' ? tRange : undefined} />
              <YNum label="f [pN]" />
              <Tooltip
                content={
                  <Tip
                    unitX={yAxis === 'time' ? 'ps' : 'Å'}
                    fmtX={(v) => v.toFixed(yAxis === 'time' ? 2 : 3)}
                    rows={(p) => `${p.dataKey === 'f' ? 'raw' : 'smoothed'}: ${typeof p.value === 'number' ? p.value.toFixed(2) : '—'} pN`}
                  />
                }
              />
              <Line type="linear" dataKey="f" stroke={chart.accent} strokeWidth={0.9} strokeOpacity={smoothing > 1 ? 0.42 : 1} dot={false} isAnimationActive={false} />
              {smoothing > 1 && <Line type="linear" dataKey="fs" stroke={chart.ink} strokeWidth={1.4} dot={false} isAnimationActive={false} />}
              {ev && yAxis === 'time' && <ReferenceDot x={ev.t_peak_ps} y={ev.f_peak_smoothed_pN} r={4} fill={chart.ink} stroke="#fff" strokeWidth={1.5} />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Figure>

      {trace.data && (
        <div className="readouts">
          <Readout label="Samples" value={trace.data.time_ps.length} />
          <Readout label="Raw max" value={fixed(Math.max(...trace.data.force_pN), 2)} unit="pN" />
          {ev && <Readout label="Smoothed peak" value={fixed(ev.f_peak_smoothed_pN, 2)} unit="pN" tone="accent" />}
          {ev && <Readout label="t‡" value={fixed(ev.t_peak_ps, 1)} unit="ps" />}
          {ev && <Readout label="box length at t‡" value={fixed(ev.x_peak_A, 3)} unit="Å" />}
          {ev && <Readout label="Trough" value={fixed(ev.f_trough_pN, 1)} unit="pN" />}
        </div>
      )}

      <div style={{ marginTop: 34 }}>
        <div className="controls">
          <Slider label="traces overlaid" value={overlayN} min={2} max={overlay.data?.max_traces ?? 12} step={1} onChange={setOverlayN} />
        </div>
        <Figure title={`the first ${overlayIds.length} traces, raw`} note="Same ramp, different transition times: that spread is the whole subject of the paper. Every second sample is drawn.">
          <div className="chart">
            <ResponsiveContainer>
              <LineChart data={overlayData} margin={chart.margin}>
                <Grid />
                <XNum dataKey="t" label="t [ps]" range={tRange} />
                <YNum label="f [pN]" />
                <Tooltip content={<Tip unitX="ps" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.dataKey}: ${typeof p.value === 'number' ? p.value.toFixed(1) : '—'}`} />} />
                {overlayIds.map((tid, k) => (
                  <Line key={tid} type="linear" dataKey={tid} stroke={tid === id ? chart.accent : chart.series[k % chart.series.length]} strokeWidth={tid === id ? 1.3 : 0.7} strokeOpacity={tid === id ? 1 : 0.75} dot={false} isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>
    </Panel>
  )
}
