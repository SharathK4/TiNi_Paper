import { useMemo, useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip } from 'recharts'
import { api, type Collection, type FileRow, type Inspection } from '../api/client'
import { useResource } from '../api/hooks'
import { fixed, fmt, mb } from '../lib/format'
import { Empty, Figure, Grid, Panel, Readout, Tip, XNum, YNum, chart } from './ui'
import { M } from './Math'

export function DatasetPanel({ version, selected, onSelect }: { version: number; selected: string | null; onSelect: (id: string) => void }) {
  const insp = useResource<Inspection>(() => api.inspection(), [], version)
  const files = useResource<{ files: FileRow[]; collection: Collection }>(() => api.files(), [], version)
  const [sortKey, setSortKey] = useState<keyof FileRow>('trace_id')
  const [asc, setAsc] = useState(true)

  const rows = useMemo(() => {
    const r = files.data?.files ?? []
    return [...r].sort((a, b) => {
      const x = a[sortKey] as number | string | null
      const y = b[sortKey] as number | string | null
      if (x === y) return 0
      if (x === null || x === undefined) return 1
      if (y === null || y === undefined) return -1
      return (x < y ? -1 : 1) * (asc ? 1 : -1)
    })
  }, [files.data, sortKey, asc])

  const peakHist = useMemo(() => {
    const vals = (files.data?.files ?? []).map((f) => f.f_max_pN)
    if (!vals.length) return []
    const lo = Math.floor(Math.min(...vals)),
      hi = Math.ceil(Math.max(...vals))
    const n = 12
    const w = (hi - lo) / n
    const bins = Array.from({ length: n }, (_, i) => ({ x: lo + (i + 0.5) * w, n: 0 }))
    vals.forEach((v) => {
      const i = Math.min(n - 1, Math.floor((v - lo) / w))
      bins[i].n += 1
    })
    return bins
  }, [files.data])

  const sort = (k: keyof FileRow) => {
    if (k === sortKey) setAsc(!asc)
    else {
      setSortKey(k)
      setAsc(true)
    }
  }

  const lead = (
    <>
      One hundred molecular-dynamics runs of the same crystal, pulled at the same rate. Each file is one force–time trace. Everything below is read from the files
      themselves and checked before any averaging.
    </>
  )

  if (insp.missing || files.missing) {
    return (
      <Panel index="1.1" title="The dataset" lead={lead}>
        <Empty>Run the pipeline to inspect and validate the files.</Empty>
      </Panel>
    )
  }
  const i = insp.data
  const c = files.data?.collection

  return (
    <Panel index="1.1" title="The dataset" lead={lead}>
      <div className="grid-2">
        <div>
          <div className="figure__title">What the files contain</div>
          {i && (
            <table className="facts">
              <tbody>
                <tr>
                  <td>Files</td>
                  <td>
                    {i.n_files} · {mb(i.total_size_mb * 1e6)} on disk · {fmt(i.rows_per_file_first, 0)} rows × {i.n_columns} columns each
                  </td>
                </tr>
                <tr>
                  <td>Columns</td>
                  <td>
                    time [ps] · box length <M tex="L_x" /> [Å] · force per atom [pN]
                  </td>
                </tr>
                <tr>
                  <td>Time grid</td>
                  <td>
                    Δt = {i.dt_ps} ps, {i.t_start_ps} → {fixed(i.t_end_ps, 2)} ps
                  </td>
                </tr>
                <tr>
                  <td>Box length</td>
                  <td>
                    {fixed(i.x_start_A, 3)} → {fixed(i.x_end_A, 3)} Å · strain {(100 * i.strain_end).toFixed(2)} %
                  </td>
                </tr>
                <tr>
                  <td>Extension rate</td>
                  <td>
                    {i.extension_rate_A_per_ps.toExponential(4)} Å/ps
                    {i.paper_rate_match && (
                      <>
                        {' '}
                        → paper rate <b>({i.paper_rate_match.index})</b>, {i.paper_rate_match.ext_rate_A_per_ps.toExponential(3)} Å/ps (ratio {fixed(i.paper_rate_match.ratio_measured_over_paper, 4)})
                      </>
                    )}
                  </td>
                </tr>
                <tr>
                  <td>Memory if fully loaded</td>
                  <td>{i.memory_estimate.all_files_float64_mb} MB as float64 — streamed per file anyway, so a run can resume</td>
                </tr>
                <tr>
                  <td>Header</td>
                  <td style={{ fontFamily: 'var(--mono)', fontSize: 11.5, color: 'var(--ink-2)' }}>
                    {i.header_lines.map((h, k) => (
                      <div key={k}>{h}</div>
                    ))}
                  </td>
                </tr>
              </tbody>
            </table>
          )}
          {c && (
            <div className="readouts">
              <Readout label="Validated" value={`${c.n_ok} / ${c.n_files}`} />
              <Readout label="Warnings" value={c.n_with_warnings} />
              <Readout label="Shared time grid" value={c.same_time_grid ? 'yes' : 'no'} />
              <Readout label="Rate spread" value={c.ext_rate_A_per_ps.std ? (c.ext_rate_A_per_ps.std / c.ext_rate_A_per_ps.mean).toExponential(1) : '—'} unit="rel. s.d." />
            </div>
          )}
        </div>
        <Figure title="Raw peak force per file" note="argmax of the raw trace — noise-inflated; the smoothed peak in section 1.4 is the one used for the transition statistics.">
          <div className="chart chart--short">
            <ResponsiveContainer>
              <BarChart data={peakHist} margin={chart.margin} barCategoryGap={2}>
                <Grid />
                <XNum dataKey="x" label="raw max f [pN]" />
                <YNum label="files" width={40} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.value} files`} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="n" fill={chart.accent} isAnimationActive={false} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <div style={{ marginTop: 30 }}>
        <div className="figure__title">Every file · click a row to open it in section 1.2</div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                {(
                  [
                    ['trace_id', 'trace'],
                    ['n_rows', 'rows'],
                    ['x_start_A', 'box length at t = 0 Å'],
                    ['ext_rate_A_per_ps', 'extension rate Å/ps'],
                    ['f_mean_pN', 'mean f pN'],
                    ['f_std_pN', 's.d. pN'],
                    ['f_max_pN', 'raw max pN'],
                    ['t_peak_ps', 't‡ ps'],
                    ['f_peak_smoothed_pN', 'smoothed peak pN'],
                    ['f_crit_from_rate_pN', 'f‡ = rate × t‡ pN'],
                    ['ok', 'valid'],
                  ] as [keyof FileRow, string][]
                ).map(([k, l]) => (
                  <th key={k} onClick={() => sort(k)} style={{ cursor: 'pointer' }}>
                    {l}
                    {sortKey === k ? (asc ? ' ↑' : ' ↓') : ''}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.trace_id} className={r.trace_id === selected ? 'is-selected' : ''} onClick={() => onSelect(r.trace_id)}>
                  <td style={{ fontFamily: 'var(--mono)' }}>{r.trace_id}</td>
                  <td>{r.n_rows}</td>
                  <td>{fixed(r.x_start_A, 3)}</td>
                  <td>{r.ext_rate_A_per_ps.toExponential(3)}</td>
                  <td>{fixed(r.f_mean_pN, 2)}</td>
                  <td>{fixed(r.f_std_pN, 2)}</td>
                  <td>{fixed(r.f_max_pN, 2)}</td>
                  <td>{fixed(r.t_peak_ps, 1)}</td>
                  <td>{fixed(r.f_peak_smoothed_pN, 2)}</td>
                  <td>{fixed(r.f_crit_from_rate_pN, 2)}</td>
                  <td>{r.ok ? (r.n_warnings ? `warn ${r.n_warnings}` : 'ok') : 'error'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Panel>
  )
}
