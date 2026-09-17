/** Typed client for the Version 2 inverse-problem endpoints. */
import { ApiError, type ParamSet, type Progress } from './client'

export type InvParams = { U_barrier: number; xi_barrier: number; gamma0: number }
export type Hist = { bins: number; rule: string; edges: number[]; centers: number[]; counts: number[]; density: number[]; density_err: number[] }

export type InvSpec = {
  model_version: string
  code_hash: string
  phases: string[]
  defaults: Record<string, unknown> & { bootstrap_n: number; profile_points: number; synthetic_replicates: number; seed: number; source: string; reference_set: string; bounds: Record<string, [number, number]> }
  sources: string[]
  paper_parameter_sets: Record<string, ParamSet>
  kBT_pN_A: number
  temperature_K: number
  paper_rates: { index: string; force_rate_pN_per_ps: number; n_traces: number }[]
  fig3_readings_by_eye: { index: string; force_rate_pN_per_ps: number; mean_f_pN: number; uncertainty_pN: number }[]
}

export type InvSample = {
  n: number
  source: string
  smoothing_points: number
  window_ps: number
  force_rate_pN_per_ps: number
  force_rate_se: number
  mean_pN: number
  sd_pN: number
  sem_pN: number
  median_pN: number
  min_pN: number
  max_pN: number
  skewness: number
  sample_fingerprint: string
  dataset_fingerprint: string
  notes: string[]
  forces_pN: number[]
  histogram: Hist
  kde: { bandwidth_pN: number; rule: string; grid: number[]; density: number[] }
  ecdf: { x: number[]; F: number[] }
}

export type InvModel = {
  parameters: { f_c_pN: number; U_barrier_kBT: number; mu0: number; mu0_times_exp_a: number; eta_pN_ps_per_A: number }
  force_rate_pN_per_ps: number
  f_pN: number[]
  pdf: number[]
  cdf: number[]
  survival: number[]
  log10_rate: number[]
  mean_eq8_pN: number
  mean_exact_pN: number
  sd_exact_pN: number
  mode_pN: number
  residual_mass: number
  spectrum: { force_rate: number; index: string; mean_eq8: number; mean_exact: number; sd_exact: number }[]
}

export type InvProgress = Progress & { run_id?: string }

export type ParamResult = {
  mle: number
  se_asymptotic: number | null
  ci95_asymptotic: [number, number] | null
  ci95_profile: [number | null, number | null]
  profile_unbounded: [boolean, boolean]
  profile_support_limited: [boolean, boolean]
  ci95_bootstrap: [number, number] | null
  bootstrap_median: number | null
  bootstrap_hit_bounds: [number | null, number | null]
  paper: number
}

export type Derived = { f_c_pN: number; U_barrier_kBT: number; mu0: number; eta_pN_ps_per_A: number; mean_eq8_pN: number; mean_exact_pN: number; sd_exact_pN: number; mode_pN: number; residual_mass: number }
export type LR = { statistic: number; df: number; pvalue: number }

export type SynthSummary = Record<string, { truth?: number; median: number; p16: number; p84: number; median_ratio?: number; log10_spread_16_84?: number; within_factor_2?: number; within_20pct?: number }>

export type InvResult = {
  run_id: string
  model_version: string
  code_hash: string
  dataset_fingerprint: string
  sample_fingerprint: string
  created_at_iso: string
  completed_at_iso: string
  config: Record<string, unknown> & { bootstrap_n: number; profile_points: number; synthetic_replicates: number; seed: number; source: string; reference_set: string; maxiter: number; xatol: number; fatol: number; bounds: Record<string, [number, number]> }
  sample: { n: number; source: string; smoothing_points: number; window_ps: number; force_rate_pN_per_ps: number; force_rate_se: number; mean_pN: number; sd_pN: number; sem_pN: number; median_pN: number; min_pN: number; max_pN: number; skewness: number; notes: string[] }
  fit: { params: InvParams; nll: number; log_likelihood: number; converged: boolean; n_eval_total: number; duration_s: number; polish: Record<string, unknown>; derived: Derived; nll_at_reference: number; likelihood_ratio_vs_reference: LR }
  parameters: Record<keyof InvParams, ParamResult>
  hessian: { eigenvalues: number[]; eigenvectors: number[][]; condition_number: number; positive_definite: boolean; correlation: number[][]; free: string[] }
  bootstrap: { n_boot: number; n_ok: number; n_failed: number; seed: number; f_c_pN: { median: number; ci95: [number, number] }; log_correlation: number[][] }
  goodness_of_fit: { ks_statistic: number; ks_pvalue: number; chi2: number; chi2_dof: number; chi2_pvalue: number; aic: number; bic: number; model_mean_exact_pN: number; model_sd_exact_pN: number; model_mean_eq8_pN: number; residual_mass: number; sample_mean_pN: number; sample_sd_pN: number }
  paper_comparison: Record<string, { label: string; params: InvParams; nll: number; ks_statistic: number; ks_pvalue: number; chi2_pvalue: number; model_mean_exact_pN: number; model_sd_exact_pN: number; model_mean_eq8_pN: number; residual_mass: number; likelihood_ratio_vs_fit: LR }>
  constrained: { label: string; fixed: Partial<InvParams>; params: InvParams; nll: number; delta_nll_vs_unconstrained: number; likelihood_ratio: LR; se: Partial<InvParams> | null; converged: boolean; derived: Derived }[]
  sensitivity: { source: string; window_ps: number; smoothing_points: number; axis: string; force_rate: number; sample_mean: number; sample_sd: number; params: InvParams; nll: number; converged: boolean; f_c_pN: number }[]
  synthetic: { designs: { name: string; rates: number[]; S: number; n_events: number; replicates: number; summary: SynthSummary }[]; truth: InvParams; replicates: number } | { skipped: true }
  phase_files: Record<string, string>
}

export type DiagBlock = {
  log_likelihood: number
  nll: number
  ks_statistic: number
  ks_pvalue: number
  chi2: number
  chi2_dof: number
  chi2_pvalue: number
  histogram: Hist
  expected_counts: number[]
  residuals_poisson: number[]
  pdf_grid: number[]
  pdf: number[]
  cdf: number[]
  qq: { model: number[]; data: number[] }
  model_mean_exact_pN: number
  model_sd_exact_pN: number
}
export type InvDiagnostics = { fitted: DiagBlock; paper: Record<string, DiagBlock & { label: string; params: InvParams }> }

export type InvProfiles = {
  chi2_1_95: number
  profiles: Record<string, { grid: { value: number; nll: number | null; delta_nll: number | null; feasible: boolean }[]; mle: number; ci95_lower: number | null; ci95_upper: number | null; lower_unbounded_in_scan: boolean; upper_unbounded_in_scan: boolean; lower_support_limited: boolean; upper_support_limited: boolean; scan_range: [number, number] }>
}

export type InvBootstrap = { n_boot: number; n_ok: number; draws: (InvParams & { nll: number })[] }
export type InvFit = { history: ({ iteration: number; nll: number } & InvParams)[]; starts: { start: InvParams; params: InvParams; nll: number; n_eval: number; n_iter: number; converged: boolean }[] }
export type InvSampleDoc = InvSample
export type InvRunList = { runs: { run_id: string; completed_at_iso: string; source: string; n: number; params: InvParams; nll: number; converged: boolean }[]; latest: { run_id: string } | null }

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) {
    let msg = r.statusText
    try {
      msg = (await r.json()).detail ?? msg
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

export const inverse = {
  spec: () => get<InvSpec>('/api/inverse/spec'),
  sample: (source: string, smoothing?: number, window?: number) => {
    const q = new URLSearchParams({ source })
    if (smoothing) q.set('smoothing', String(smoothing))
    if (window) q.set('window', String(window))
    return get<InvSample>(`/api/inverse/sample?${q.toString()}`)
  },
  model: (U: number, xi: number, gamma0: number, fdot?: number) => get<InvModel>(`/api/inverse/model?U=${U}&xi=${xi}&gamma0=${gamma0}${fdot ? `&fdot=${fdot}` : ''}`),
  progress: () => get<InvProgress>('/api/inverse/progress'),
  run: (config: Record<string, unknown>, run_id?: string) => post<{ run_id: string }>('/api/inverse/run', { config, run_id }),
  cancel: () => post<{ cancelling: boolean }>('/api/inverse/cancel', {}),
  runs: () => get<InvRunList>('/api/inverse/runs'),
  latest: () => get<InvResult>('/api/inverse/latest'),
  result: (id: string) => get<InvResult>(`/api/inverse/runs/${id}`),
  phase: <T,>(id: string, phase: string) => get<T>(`/api/inverse/runs/${id}/${phase}`),
}
