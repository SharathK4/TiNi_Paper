/** Typed client for Version 4 — Inflect. */

const BASE = '/api/inflect'

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined && v !== null) q.set(k, String(v))
  const url = `${BASE}${path}${q.toString() ? `?${q}` : ''}`
  const r = await fetch(url)
  if (!r.ok) throw new Error(await errorText(r))
  return r.json() as Promise<T>
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: body === undefined ? {} : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!r.ok) throw new Error(await errorText(r))
  return r.json() as Promise<T>
}

async function errorText(r: Response): Promise<string> {
  try {
    const d = await r.json()
    const detail = (d as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (detail) return JSON.stringify(detail)
  } catch {
    /* fall through to the status line */
  }
  return `${r.status} ${r.statusText}`
}

// ------------------------------------------------------------------- shapes
export type Column = {
  index: number
  name: string
  unit: string | null
  label: string
  named_by_file: boolean
}

export type ColumnStat = {
  index: number
  name: string
  unit: string | null
  label: string
  kind: string
  kind_reason: string
  n_finite: number
  n_nan: number
  minimum: number
  maximum: number
  span: number
  monotonic: string
  uniform_spacing: boolean
  noise_sigma: number
  snr: number
  trend_strength: number
  constant: boolean
  x_score: number
  y_score: number
  x_reasons: string[]
  y_reasons: string[]
  duplicate_of: string | null
}

export type ColumnReport = {
  columns: ColumnStat[]
  x_ranking: number[]
  y_ranking: number[]
  recommended_x: number | null
  recommended_y: number | null
  headline: string
}

export type Dialect = {
  delimiter: string
  comment_prefix: string | null
  n_comment_lines: number
  header_line: string | null
  header_from_comment: boolean
  n_fields: number
  reasons: string[]
}

export type TableInfo = {
  name: string
  n_rows: number
  n_cols: number
  columns: Column[]
  dialect: Dialect
  comment_lines: string[]
  warnings: string[]
}

export type Annotation = {
  kind: 'region' | 'vline' | 'hline' | 'circle' | 'bracket' | 'slope' | 'point' | 'curve' | 'band'
  tone: 'accent' | 'model' | 'muted'
  x0?: number
  x1?: number
  y0?: number
  y1?: number
  text?: string
  detail?: string
}

export type Slot = {
  name: string
  value: number
  unit?: string
  decimals: number
  control?: string
  minimum?: number
  maximum?: number
  step?: number
}

export type Stat = { label: string; value: string | number; unit?: string | null }

export type Beat = {
  id: string
  kind: 'overview' | 'shape' | 'regime' | 'transition' | 'caution' | 'knee' | 'confidence'
  title: string
  sentence: string
  slots: Slot[]
  annotations: Annotation[]
  stats: Stat[]
  focus?: [number, number]
  tier?: 'measured' | 'read'
  findings?: string[]
}

export type Reading = {
  x_label: string
  y_label: string
  x_unit: string | null
  y_unit: string | null
  shape: string
  headline: string
  beats: Beat[]
  quantities: { name: string; value: number; unit: string | null; from: [number, number] }[]
  transition: {
    x: number
    x_lo: number
    x_hi: number
    width: number
    onset_value: number
    final_value: number
    step: number
    step_sigma: number
    peak_value: number
    peak_x: number
    onset_vs_peak: number
  } | null
}

export type Boundary = {
  index: number
  x: number
  kind: 'transition' | 'knee' | 'break' | 'step'
  y_before: number
  y_after: number
  gap: number
  gap_sigma: number
  slope_before: number
  slope_after: number
  left_model: string
  right_model: string
  x_lo: number
  x_hi: number
  width: number
  sharpness: number
}

export type Segment = {
  i0: number
  i1: number
  x0: number
  x1: number
  degree: number
  model: string
  coeffs: number[]
  x_mid: number
  sigma: number
  r2: number
  n: number
  y_start: number
  y_end: number
  slope_start: number
  slope_end: number
  curvature: number
  span: number
}

export type Segmentation = {
  segments: Segment[]
  boundaries: Boundary[]
  events: { x0: number; x1: number; width: number; drop: number; drop_sigma: number }[]
  sigma: number
  n_segments: number
  shape: string
  chosen_by: 'penalty' | 'requested'
}

export type CurveResult = {
  file: string
  x_column: number
  y_column: number
  x_label: string
  y_label: string
  x_unit: string | null
  y_unit: string | null
  n_points: number
  curve: { x: number[]; y: number[]; decimated: boolean }
  domain: [number, number]
  range: [number, number]
  fits: { x0: number; x1: number; model: string; degree: number; points: [number, number][] }[]
  segmentation: Segmentation
  reading: Reading
  columns: Column[]
}

export type SessionMeta = {
  id: string
  title: string
  created_at_iso: string
  sources: { name: string; stored: boolean; size_bytes: number }[]
  config: Record<string, unknown>
}

export type SessionDetail = {
  session: SessionMeta
  first_file: TableInfo
  columns: ColumnReport
  files: { name: string; stored: boolean; size_bytes: number }[]
  has_ensemble: boolean
}

export type CurveSummary = {
  name: string
  ok: boolean
  error: string | null
  shape: string
  n_regimes: number
  noise: number | null
  y_range: number | null
  first_slope: number | null
  n_transitions: number
  transition_x: number | null
  transition_onset: number | null
  transition_step: number | null
  transition_width: number | null
}

export type EnsembleResult = {
  n_files: number
  n_ok: number
  n_failed: number
  curves: CurveSummary[]
  stats: Record<string, { n: number; median: number | null; spread: number | null; mean: number | null; sd: number | null; min: number | null; max: number | null }>
  outliers: { name: string; feature: string; feature_label: string; value: number; median: number; z: number; sentence: string }[]
  groups: { signature: string; count: number; members: string[] }[]
  headline: string
  failures: { name: string; error: string }[]
  sparklines?: Record<string, number[]>
}

export type EnsembleProgress = {
  status: 'idle' | 'running' | 'done' | 'failed' | 'cancelled'
  done: number
  total: number
  percent: number
  message: string
  running: boolean
  session_id: string | null
}

export type CurveQuery = {
  file?: string
  x_column?: number
  y_column?: number
  n_segments?: number
  penalty_scale?: number
  detect_transitions?: boolean
  x_scale?: number
}

// ------------------------------------------------------------------ methods
export const inflect = {
  spec: () => get<{ defaults: Record<string, unknown>; max_files: number; sample_available: boolean }>('/spec'),
  sessions: () => get<{ sessions: { id: string; title: string; created_at_iso: string; n_files: number; has_ensemble: boolean }[] }>('/sessions'),
  session: (id: string) => get<SessionDetail>(`/sessions/${id}`),
  sample: (title?: string) => post<{ id: string; accepted: number }>('/sample', title ? { title } : {}),
  setConfig: (id: string, config: Record<string, unknown>) => post<{ config: Record<string, unknown> }>(`/sessions/${id}/config`, config),
  curve: (id: string, q: CurveQuery = {}) => get<CurveResult>(`/sessions/${id}/curve`, q as Record<string, string | number | boolean | undefined>),
  runEnsemble: (id: string) => post<EnsembleProgress>(`/sessions/${id}/ensemble`),
  ensembleProgress: () => get<EnsembleProgress>('/ensemble/progress'),
  ensemble: (id: string) => get<EnsembleResult>(`/sessions/${id}/ensemble`),

  async upload(files: File[], title: string, onProgress?: (fraction: number) => void): Promise<{ id: string; accepted: number; problems: { name: string; error: string }[] }> {
    const form = new FormData()
    form.append('title', title)
    for (const f of files) form.append('files', f)
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/sessions`)
      xhr.upload.onprogress = (e) => {
        if (onProgress && e.lengthComputable) onProgress(e.loaded / e.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          let message = `${xhr.status} ${xhr.statusText}`
          try {
            const d = JSON.parse(xhr.responseText)
            if (typeof d.detail === 'string') message = d.detail
          } catch {
            /* keep the status line */
          }
          reject(new Error(message))
        }
      }
      xhr.onerror = () => reject(new Error('the upload could not reach the server'))
      xhr.send(form)
    })
  },
}

// ---------------------------------------------- phase 2: how sure is it, and
// ---------------------------------------------- what the whole set looks like
export type Interval = { value: number; ci: [number, number] | null; sd: number | null; n: number }

export type Uncertainty = {
  file: string
  x_unit: string | null
  y_unit: string | null
  n_resamples: number
  n_used: number
  block_length: number
  detection_rate: number
  note: string
  transition: { position: Interval; onset: Interval; step: Interval } | null
  segments: { x0: number; x1: number; model: string; slope: Interval }[]
}

export type StabilityPoint = {
  setting: 'threshold' | 'penalty' | 'regimes'
  label: string
  value: number
  is_default: boolean
  found: boolean
  position: number | null
  onset: number | null
  n_segments: number
}

export type Stability = {
  file: string
  points: StabilityPoint[]
  reference: number | null
  spread: number | null
  holds: boolean
  sentence: string
}

export type Arrangement = {
  grid: number[]
  median: (number | null)[]
  low: (number | null)[]
  high: (number | null)[]
  lines: (number | null)[][]
  n_curves: number
  n_drawn: number
  origin: string
}

export type ChorusResult = {
  measured: Arrangement | null
  aligned: Arrangement | null
  names: string[]
  x_label: string
  y_label: string
  x_unit: string | null
  y_unit: string | null
}

export const inflectMore = {
  uncertainty: (id: string, q: CurveQuery & { n_resamples?: number } = {}) =>
    get<Uncertainty>(`/sessions/${id}/uncertainty`, q as Record<string, string | number | boolean | undefined>),
  stability: (id: string, q: CurveQuery = {}) =>
    get<Stability>(`/sessions/${id}/stability`, q as Record<string, string | number | boolean | undefined>),
  chorus: (id: string) => get<ChorusResult>(`/sessions/${id}/chorus`),
  methods: (id: string, file?: string) => get<{ methods: string }>(`/sessions/${id}/methods`, { file }),
  csvUrl: (id: string) => `${BASE}/sessions/${id}/results.csv`,
}

// ------------------------------------------ phase 3: the ledger and the record
export type FindingInterval = Interval & {
  median?: number
  shift?: number
  biased?: boolean
  samples?: number[]
}

export type FindingSensitivity = {
  value: number
  present_in: number
  of: number
  min: number | null
  max: number | null
  spread: number
  holds: boolean
  points: { label: string; value: number | null }[]
}

export type Finding = {
  id: string
  label: string
  tier: 'measured' | 'read'
  value: number | null
  unit: string | null
  method: string
  rule: string | null
  text: string | null
  evidence: Record<string, unknown>
  anchor: { x?: number; y?: number; x0?: number; x1?: number; y0?: number; y1?: number }
  group: string
  interval: FindingInterval | null
  sensitivity: FindingSensitivity | null
}

export type Note = { id: string; file: string; finding_id: string; text: string; created_at_iso: string }

export type Ledger = {
  file: string
  x_label: string
  y_label: string
  x_unit: string | null
  y_unit: string | null
  settings: Record<string, unknown>
  findings: Finding[]
  summary: { n: number; tiers: Record<string, number>; with_interval: number; swept: number; holds: number }
  notes: Note[]
  bootstrap: { n_used: number; block_length: number; detection_rate: number } | null
  stability_sentence: string | null
}

export type RecordDoc = {
  record_version: string
  created_at_iso: string
  software: { name: string; version: string; code_hash: string }
  session: string
  data: { file: string; sha256: string; n_rows: number; columns: Record<string, unknown> }
  settings: Record<string, unknown>
  seed: number
  findings: Finding[]
  interpretations: Note[]
  note: string
}

export type VerifyResult = {
  reproduced: boolean
  n_recorded: number
  n_same: number
  same: string[]
  moved: { id: string; recorded: unknown; now: unknown; difference?: number }[]
  missing: string[]
  new: string[]
  data_matches: boolean
  code_matches: boolean
  file: string
}

export const inflectLedger = {
  ledger: (id: string, q: CurveQuery & { with_uncertainty?: boolean; n_resamples?: number } = {}) =>
    get<Ledger>(`/sessions/${id}/ledger`, q as Record<string, string | number | boolean | undefined>),
  notes: (id: string, file?: string) => get<{ notes: Note[] }>(`/sessions/${id}/notes`, { file }),
  addNote: (id: string, file: string, finding_id: string, text: string) =>
    post<Note>(`/sessions/${id}/notes`, { file, finding_id, text }),
  removeNote: async (id: string, noteId: string) => {
    const r = await fetch(`${BASE}/sessions/${id}/notes/${noteId}`, { method: 'DELETE' })
    if (!r.ok) throw new Error(await errorText(r))
  },
  record: (id: string, file?: string) => get<RecordDoc>(`/sessions/${id}/record`, { file }),
  recordUrl: (id: string, file?: string) =>
    `${BASE}/sessions/${id}/record?download=true${file ? `&file=${encodeURIComponent(file)}` : ''}`,
  verify: (id: string, record: RecordDoc) => post<VerifyResult>(`/sessions/${id}/verify`, record),
}
