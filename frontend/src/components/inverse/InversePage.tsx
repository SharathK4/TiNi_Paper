/**
 * Version 2 — the inverse problem, as one scrolling page in seven sections:
 *
 *   A what is inferred · B what was observed · C the model · D running it ·
 *   E model vs observations · F what was inferred · G the record
 *
 * State: the selected run (latest by default) and its phase files; the live
 * progress of a run in flight; the observed sample as currently configured
 * in section B (shared with C so the model is drawn at the same ḟ).
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Config } from '../../api/client'
import { useResource } from '../../api/hooks'
import { inverse, type InvBootstrap, type InvDiagnostics, type InvFit, type InvProfiles, type InvProgress, type InvResult, type InvRunList, type InvSample, type InvSpec } from '../../api/inverse'
import { fixed } from '../../lib/format'
import { ComparisonPanel } from './ComparisonPanel'
import { ModelPanel } from './ModelPanel'
import { ObservedPanel } from './ObservedPanel'
import { OverviewPanel } from './OverviewPanel'
import { RecordPanel } from './RecordPanel'
import { ResultsPanel } from './ResultsPanel'
import { WorkspacePanel } from './WorkspacePanel'
import { PARAM_KEYS, fmtParam } from './shared'

function useInverseProgress() {
  const [progress, setProgress] = useState<InvProgress | null>(null)
  const timer = useRef<number | null>(null)
  const tick = useCallback(async () => {
    try {
      const p = await inverse.progress()
      setProgress(p)
      return p.running
    } catch {
      return false
    }
  }, [])
  useEffect(() => {
    let alive = true
    const loop = async () => {
      const running = await tick()
      if (!alive) return
      timer.current = window.setTimeout(loop, running ? 800 : 5000)
    }
    loop()
    return () => {
      alive = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [tick])
  return { progress, refresh: tick }
}

export function InversePage() {
  const { progress, refresh } = useInverseProgress()
  const [version, setVersion] = useState(0)
  const wasRunning = useRef(false)
  useEffect(() => {
    if (progress?.running) wasRunning.current = true
    else if (wasRunning.current) {
      wasRunning.current = false
      setVersion((v) => v + 1)
    }
  }, [progress?.running])

  const spec = useResource<InvSpec>(() => inverse.spec(), [])
  const config = useResource<Config>(() => api.config(), [])
  const runs = useResource<InvRunList>(() => inverse.runs(), [], version)
  const [selected, setSelected] = useState<string | null>(null)
  const runId = selected ?? runs.data?.latest?.run_id ?? null

  const result = useResource<InvResult>(() => (runId ? inverse.result(runId) : Promise.reject(new Error('none'))), [runId], version)
  const diag = useResource<InvDiagnostics>(() => (runId ? inverse.phase<InvDiagnostics>(runId, 'diagnostics') : Promise.reject(new Error('none'))), [runId], version)
  const profiles = useResource<InvProfiles>(() => (runId ? inverse.phase<InvProfiles>(runId, 'profiles') : Promise.reject(new Error('none'))), [runId], version)
  const boot = useResource<InvBootstrap>(() => (runId ? inverse.phase<InvBootstrap>(runId, 'bootstrap') : Promise.reject(new Error('none'))), [runId], version)
  const fitDoc = useResource<InvFit>(() => (runId ? inverse.phase<InvFit>(runId, 'fit') : Promise.reject(new Error('none'))), [runId], version)

  const [sample, setSample] = useState<InvSample | null>(null)
  const onSample = useCallback((s: InvSample | null) => setSample(s), [])

  const run = useCallback(
    async (cfg: Record<string, unknown>) => {
      try {
        await inverse.run(cfg)
        await refresh()
      } catch (e) {
        window.alert(`Could not start: ${(e as Error).message}`)
      }
    },
    [refresh],
  )
  const resume = useCallback(
    async (id: string) => {
      await inverse.run({}, id)
      await refresh()
    },
    [refresh],
  )
  const cancel = useCallback(async () => {
    await inverse.cancel()
    await refresh()
  }, [refresh])

  const res = result.data
  const s = config.data?.settings

  return (
    <>
      <header className="masthead">
        <h1 className="wordmark">TiNi</h1>
        <p className="standfirst">
          The inverse problem: from the forces at which one hundred crystals snapped, back to the barrier they crossed.
          {res && (
            <>
              {' '}
              Inferred: <b>U‡ = {fmtParam('U_barrier', res.fit.params.U_barrier)} pN·Å</b>, <b>ξ‡ = {fmtParam('xi_barrier', res.fit.params.xi_barrier)} Å</b>,{' '}
              <b>Γ₀ = {fmtParam('gamma0', res.fit.params.gamma0)} ps⁻¹</b> — f_c = {fixed(res.fit.derived.f_c_pN, 1)} pN.
            </>
          )}
        </p>
        <p className="byline">Version 2 · inverse problem &amp; transition statistics · after Maitra &amp; Gururajan, Sci. Rep. 15:3008 (2025)</p>
      </header>

      <OverviewPanel spec={spec.data} result={res} />
      <ObservedPanel version={version} defaults={{ smoothing: s?.peak_smoothing_points ?? 15, window: s?.force_rate_window_ps ?? 100 }} onSample={onSample} />
      <ModelPanel spec={spec.data} sample={sample} fitted={res ? res.fit.params : null} />
      <WorkspacePanel spec={spec.data} progress={progress} runs={runs.data} selectedRun={runId} onSelectRun={setSelected} onRun={run} onResume={resume} onCancel={cancel} />
      <ComparisonPanel result={res} diag={diag.data} sample={sample} />
      <ResultsPanel result={res} profiles={profiles.data} boot={boot.data} />
      <RecordPanel result={res} spec={spec.data} fit={fitDoc.data} />

      <p className="footnote">
        Parameters: {PARAM_KEYS.map((k) => k).join(', ')} in pN·Å, Å, ps⁻¹. Backend: <code>backend/tini/inverse</code>. Runs: <code>artifacts/inverse/runs/</code>.
        Specification and derivations: <code>docs/INVERSE_PROBLEM.md</code>.
      </p>
    </>
  )
}
