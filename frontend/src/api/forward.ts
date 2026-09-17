/** Typed client for the Version 3 forward-model endpoints. */
import { ApiError, type ParamSet, type Progress } from './client'
import type { Hist } from './inverse'

export type FwdParams = {
  U_barrier_pN_A: number
  xi_barrier_A: number
  gamma0_per_ps: number
  kBT_pN_A: number
  f_c_pN: number
  U_barrier_kBT: number
  curvature_pN_per_A: number
  eta_pN_ps_per_A: number
  D_A2_per_ps: number
  tau_well_ps: number
  tau_diffusion_ps: number
}

export type FwdSpec = {
  model_version: string
  code_hash: string
  phases: string[]
  defaults: Record<string, unknown> & { n_cells: number; n_steps: number; boundary: string; margin_kBT: number; parameter_source: string }
  parameter_sources: string[]
  boundaries: string[]
  paper_parameter_sets: Record<string, ParamSet>
  paper_rates: { index: string; force_rate_pN_per_ps: number }[]
  fig3_readings_by_eye: { index: string; force_rate_pN_per_ps: number; mean_f_pN: number; uncertainty_pN: number }[]
  kBT_pN_A: number
  temperature_K: number
}

export type FwdResolved = { parameters: FwdParams; provenance: Record<string, unknown> & { label: string }; force_rate_pN_per_ps: number; t_end_ps: number }
export type FwdProgress = Progress & { run_id?: string }

export type Level = { level: number; n_cells: number; n_steps: number; h_A: number; dt_ps: number; mean_pN: number; sd_pN: number; duration_s: number; l1_vs_previous?: number; mean_change_pN?: number }

export type FwdResult = {
  run_id: string
  model_version: string
  code_hash: string
  created_at_iso: string
  completed_at_iso: string
  config: Record<string, unknown> & { parameter_source: string; boundary: string; n_cells: number; n_steps: number; theta: number; margin_kBT: number; temperature_K: number; compare_source: string }
  parameters: FwdParams
  parameter_provenance: Record<string, unknown> & { source: string; label: string; inverse_run_id?: string; temperature_note?: string }
  force_rate_pN_per_ps: number
  force_rate_provenance: { source: string; window_ps?: number; se?: number }
  grid: { xi_L_A: number; xi_A_A: number; n_cells: number; h_A: number }
  boundary: { boundary: string; xi_A_A: number; U0_at_xi_A_pN_A: number; xi_L_A: number; U0_at_xi_L_kBT: number; margin_kBT?: number }
  t_end_ps: number
  t_fc_ps: number
  dt_ps: number
  observables: {
    mean_pN: number
    sd_pN: number
    mode_pN: number
    residual_mass: number
    mass_after_fc: number
    norm: number
    eq6: { mean_pN: number; sd_pN: number; residual_mass: number; mean_eq8_pN: number }
    l1_pde_vs_eq6: number
    max_cdf_diff_pde_vs_eq6: number
    data_comparison:
      | { source: string; n: number; sample_mean_pN: number; sample_sd_pN: number; sample_force_rate_pN_per_ps: number; same_force_rate: boolean; log_likelihood_pde: number; log_likelihood_eq6: number | null; ks_D_vs_pde: number; ks_note: string; histogram: Hist }
      | { unavailable: string }
  }
  validation:
    | {
        A_B_conservation_positivity: { conservation_error_max: number; min_density: number; final_survival: number; final_absorbed: number }
        C1_reflecting_equilibrium: { l1_drift: number; mass_error: number; t_end_ps: number }
        C2_mfpt_exact: { force_pN: number; barrier_remaining_kBT: number; tau_exact_ps: number; tau_pde_ps: number; relative_error: number; n_cells: number; n_steps: number }[]
        C3_free_diffusion: { variance_pde_A2: number; variance_exact_A2: number; relative_error: number; mass_error: number }
        F_kramers: { force_pN: number; barrier_remaining_kBT: number; gamma_eq4_per_ps: number; xi_ts_A: number; placements: { placement: string; xi_A_A: number; rate_pde_per_ps: number; ratio_pde_over_eq4: number; mfpt_exact_ps: number; ratio_1_over_mfpt_over_eq4: number }[] }
        G_reproducibility: { identical_survival: boolean; identical_snapshots: boolean; max_abs_diff: number }
      }
    | { skipped: true }
  convergence: { grid_order: number | null; time_order: number | null; richardson_grid_pN: number | null; richardson_time_pN: number | null; grid_levels: Level[]; time_levels: Level[] } | { skipped: true }
  spectrum: { index: string; force_rate: number; mean_pde_pN: number; sd_pde_pN: number; residual_mass_pde: number; mass_after_fc_pde: number; mean_eq8_pN: number; mean_eq6_exact_pN: number; sd_eq6_pN: number }[] | { skipped: true }
  solver_duration_s: number | null
  phase_files: Record<string, string>
}

export type FwdObservables = {
  force_rate_pN_per_ps: number
  t_end_ps: number
  f_pN: number[]
  pdf: number[]
  cdf: number[]
  survival_t: number[]
  t_ps: number[]
  flux: number[]
  pdf_eq6: number[]
  cdf_eq6: number[]
  snapshots: { t_ps: number[]; f_pN: number[]; xi_A: number[]; P: number[][] }
  snapshot_stride: number
  data_comparison: FwdResult['observables']['data_comparison']
}

export type FwdRunList = { runs: { run_id: string; completed_at_iso: string; parameter_source: string; boundary: string; force_rate: number; mean_pN: number; n_cells: number; n_steps: number }[]; latest: { run_id: string } | null }

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

export const forward = {
  spec: () => get<FwdSpec>('/api/forward/spec'),
  parameters: (source: string, inverse_run_id?: string, temperature_K = 300) => {
    const q = new URLSearchParams({ source, temperature_K: String(temperature_K) })
    if (inverse_run_id) q.set('inverse_run_id', inverse_run_id)
    return get<FwdResolved>(`/api/forward/parameters?${q.toString()}`)
  },
  progress: () => get<FwdProgress>('/api/forward/progress'),
  run: (config: Record<string, unknown>, run_id?: string) => post<{ run_id: string }>('/api/forward/run', { config, run_id }),
  cancel: () => post<{ cancelling: boolean }>('/api/forward/cancel', {}),
  runs: () => get<FwdRunList>('/api/forward/runs'),
  latest: () => get<FwdResult>('/api/forward/latest'),
  result: (id: string) => get<FwdResult>(`/api/forward/runs/${id}`),
  observables: (id: string) => get<FwdObservables>(`/api/forward/runs/${id}/observables`),
}
