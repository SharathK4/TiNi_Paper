/** Typed client for the FastAPI backend (proxied at /api in dev). */

export type Progress = {
  status: 'idle' | 'running' | 'done' | 'failed' | 'cancelled'
  stage: string | null
  stages: string[]
  stage_status: Record<string, string>
  files_done: number
  files_total: number
  percent: number
  elapsed_s: number
  eta_s: number | null
  message: string
  started_at: number | null
  finished_at: number | null
  last_checkpoint: { stage: string; path: string; time: number; time_iso: string } | null
  errors: { time: string; stage: string; type: string; message: string }[]
  running: boolean
  checkpoints: Record<string, { completed_at_iso: string; duration_s: number; outputs: string[]; summary: Record<string, unknown> }>
}

export type Inspection = {
  n_files: number
  total_size_mb: number
  rows_per_file_first: number
  n_columns: number
  dt_ps: number
  t_start_ps: number
  t_end_ps: number
  x_start_A: number
  x_end_A: number
  strain_end: number
  extension_rate_A_per_ps: number
  header_lines: string[]
  columns: string[]
  column_units: Record<string, string>
  paper_rate_match: { index: string; ext_rate_A_per_ps: number; force_rate_pN_per_ps: number; n_traces: number; ratio_measured_over_paper: number } | null
  memory_estimate: { bytes_per_file_float64: number; all_files_float64_mb: number; note: string }
  size_bytes_min: number
  size_bytes_max: number
}

export type FileRow = {
  trace_id: string
  name: string
  n_rows: number
  ok: boolean
  n_warnings: number
  n_errors: number
  x_start_A: number
  x_end_A: number
  ext_rate_A_per_ps: number
  f_max_pN: number
  f_mean_pN: number
  f_std_pN: number
  t_peak_ps: number | null
  f_peak_smoothed_pN: number | null
  f_crit_from_rate_pN: number | null
}

export type Collection = {
  n_files: number
  n_ok: number
  n_failed: number
  n_with_warnings: number
  same_row_count: boolean
  same_time_grid: boolean
  row_counts: number[]
  ext_rate_A_per_ps: { mean: number; std: number; min: number; max: number }
  x_start_A: { mean: number; std: number; note: string }
  issues: { level: string; code: string; detail: string }[]
  ok: boolean
}

export type Trace = {
  trace_id: string
  time_ps: number[]
  x_A: number[]
  force_pN: number[]
  force_smoothed_pN?: number[]
}

export type Overlay = {
  time_ps: number[]
  traces: { trace_id: string; force_pN: number[] }[]
  available: string[]
  max_traces: number
}

export type Average = {
  n_traces: number
  time_ps: number[]
  mean_f_pN: number[]
  std_f_pN: number[]
  sem_f_pN: number[]
  median_f_pN: number[]
  p25_f_pN: number[]
  p75_f_pN: number[]
  mean_x_A: number[]
  force_rate: { window_ps: number; slope_pN_per_ps: number; slope_se: number; intercept_pN: number; r2: number; n_points: number; t_fit: number[]; f_fit: number[] }
  sensitivity: { window_ps: number; slope: number; slope_se: number; r2: number; n_points: number }[]
  default_window_ps: number
}

export type Summ = { mean: number; std: number; sem: number; min: number; max: number; median: number }

export type Events = {
  smoothing_points: number
  force_rate_used_pN_per_ps: number
  n: number
  t_peak_ps: Summ
  f_crit_from_rate_pN: Summ
  f_peak_smoothed_pN: Summ
  f_peak_raw_pN: Summ
  events: { t_peak_ps: number; f_peak_smoothed_pN: number; f_peak_raw_pN: number; t_raw_peak_ps: number; t_trough_ps: number; f_trough_pN: number; drop_pN: number; f_crit_from_rate_pN: number; x_peak_A: number }[]
  trace_ids: string[]
  histograms: Record<string, { counts: number[]; edges: number[] }>
  assumption: string
}

export type Tilt = {
  f_pN: number
  f_over_fc: number
  t_ps: number | null
  U: number[]
  barrier_pN_A: number
  barrier_kBT: number
  xi_min: number | null
  xi_ts: number | null
}

export type Potential = {
  parameters: { U_barrier_pN_A: number; xi_barrier_A: number; f_c_pN: number; U_barrier_kBT: number; kBT_pN_A: number; curvature_at_min_pN_per_A: number }
  xi_A: number[]
  U0_pN_A: number[]
  F_pN: number[]
  F_numerical_pN: number[]
  force_check_max_abs_pN: number
  inflection_xi_A: number
  selected: Tilt | null
  tilts: Tilt[]
  force_rate_pN_per_ps: number | null
}

export type ParamSet = {
  label: string
  U_barrier_pN_A: number
  U_barrier_se: number
  xi_barrier_A: number
  xi_barrier_se: number
  gamma0_per_ps: number
  gamma0_se: number
}

export type Config = {
  settings: Record<string, unknown> & { force_rate_window_ps: number; peak_smoothing_points: number; default_parameter_set: string; overlay_max_traces: number }
  constants: { kBT_pN_A: number; temperature_K: number }
  paper_parameter_sets: Record<string, ParamSet>
  paper_rates: { index: string; ext_rate_A_per_ps: number; force_rate_pN_per_ps: number; n_traces: number }[]
  stages: string[]
}

export type Summary = {
  paper: Record<string, string>
  inspection: Partial<Inspection> | null
  validation: Collection | null
  force_rate: { force_rate_pN_per_ps: number; force_rate_se: number; window_ps: number; r2: number; mean_trace_max_pN: number; mean_trace_argmax_ps: number; n_traces: number } | null
  events: { t_peak_ps: Summ; f_crit_from_rate_pN: Summ; f_peak_smoothed_pN: Summ; smoothing_points: number; n: number } | null
  potential: Record<string, { parameters: Potential['parameters']; label: string }> | null
  constants: { kBT_pN_A: number; temperature_K: number }
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = j.detail ?? msg
    } catch {
      /* ignore */
    }
    throw new ApiError(r.status, msg)
  }
  return (await r.json()) as T
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new ApiError(r.status, (await r.json().catch(() => ({}))).detail ?? r.statusText)
  return (await r.json()) as T
}

export const api = {
  progress: () => get<Progress>('/api/progress'),
  config: () => get<Config>('/api/config'),
  run: (force = false, stages?: string[]) => post<{ run_id: string }>('/api/pipeline/run', { force, stages }),
  cancel: () => post<{ cancelling: boolean }>('/api/pipeline/cancel', {}),
  inspection: () => get<Inspection>('/api/dataset/inspection'),
  files: () => get<{ files: FileRow[]; collection: Collection }>('/api/dataset/files'),
  trace: (id: string, smoothing = 0) => get<Trace>(`/api/traces/${id}?smoothing=${smoothing}`),
  overlay: (ids: string[], stride = 2) => get<Overlay>(`/api/traces?ids=${ids.join(',')}&stride=${stride}`),
  average: (window?: number) => get<Average>(`/api/stats/average${window ? `?window=${window}` : ''}`),
  events: (smoothing?: number, window?: number) => {
    const q = new URLSearchParams()
    if (smoothing) q.set('smoothing', String(smoothing))
    if (window) q.set('window', String(window))
    const s = q.toString()
    return get<Events>(`/api/stats/events${s ? `?${s}` : ''}`)
  },
  potential: (U: number, xi: number, f: number) => get<Potential>(`/api/potential?U=${U}&xi=${xi}&f=${f}`),
  summary: () => get<Summary>('/api/summary'),
}
