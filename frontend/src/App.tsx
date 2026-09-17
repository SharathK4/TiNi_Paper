/**
 * The page.
 *
 * A masthead, one status line, and six numbered panels in the order a reader
 * needs them: the files → one trace → the average → when they snap → the
 * model landscape → the record. Nothing navigates; it is one scroll.
 *
 * `version` bumps when a pipeline run finishes so every panel re-fetches.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Config, type Events, type FileRow, type Summary } from './api/client'
import { useProgress, useResource } from './api/hooks'
import { AveragePanel } from './components/AveragePanel'
import { DatasetPanel } from './components/DatasetPanel'
import { EventsPanel } from './components/EventsPanel'
import { LandscapePanel } from './components/LandscapePanel'
import { StatusStrip } from './components/StatusStrip'
import { SummaryPanel } from './components/SummaryPanel'
import { TracePanel } from './components/TracePanel'
import { InversePage } from './components/inverse/InversePage'
import { ForwardPage } from './components/forward/ForwardPage'
import { fixed } from './lib/format'

type Route = 'traces' | 'inverse' | 'forward'

function useRoute(): Route {
  const read = (): Route => {
    const h = window.location.hash.replace('#', '')
    return h === 'inverse' || h === 'forward' ? h : 'traces'
  }
  const [route, setRoute] = useState<Route>(read)
  useEffect(() => {
    const onHash = () => {
      setRoute(read())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return route
}

export function App() {
  const route = useRoute()
  return (
    <main className="page">
      <nav className="nav" aria-label="versions">
        <a className={`nav__link ${route === 'traces' ? 'is-on' : ''}`} href="#traces">
          <span className="nav__index">v1</span> Force traces &amp; landscape
        </a>
        <a className={`nav__link ${route === 'inverse' ? 'is-on' : ''}`} href="#inverse">
          <span className="nav__index">v2</span> Inverse problem
        </a>
        <a className={`nav__link ${route === 'forward' ? 'is-on' : ''}`} href="#forward">
          <span className="nav__index">v3</span> Forward model
        </a>
      </nav>
      {route === 'inverse' ? <InversePage /> : route === 'forward' ? <ForwardPage /> : <TracesPage />}
    </main>
  )
}

function TracesPage() {
  const { progress, offline, run, cancel } = useProgress()
  const [version, setVersion] = useState(0)
  const wasRunning = useRef(false)
  useEffect(() => {
    if (progress?.running) wasRunning.current = true
    else if (wasRunning.current) {
      wasRunning.current = false
      setVersion((v) => v + 1)
    }
  }, [progress?.running])

  const config = useResource<Config>(() => api.config(), [])
  const summary = useResource<Summary>(() => api.summary(), [], version)
  const files = useResource<{ files: FileRow[] }>(() => api.files(), [], version)
  const available = files.data?.files.filter((f) => f.ok).map((f) => f.trace_id) ?? []

  const [selected, setSelected] = useState<string | null>(null)
  const [events, setEvents] = useState<Events | null>(null)
  const [rate, setRate] = useState<number | null>(null)
  const [fc, setFc] = useState<number | null>(null)

  const select = useCallback((id: string) => {
    setSelected(id)
    document.getElementById('trace')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const cfg = config.data
  const paperRate = cfg && summary.data?.inspection?.paper_rate_match ? summary.data.inspection.paper_rate_match.force_rate_pN_per_ps : null
  const fr = summary.data?.force_rate

  return (
    <>
      <header className="masthead">
        <h1 className="wordmark">TiNi</h1>
        <p className="standfirst">
          One hundred force–time traces of a titanium–nickel crystal pulled at a constant rate, and the cubic free-energy landscape the paper uses to explain
          when each one snaps.
          {fr && (
            <>
              {' '}
              From the files: <b>ḟ = {fixed(fr.force_rate_pN_per_ps, 4)} pN/ps</b>
              {paperRate && <> against the paper's {fixed(paperRate, 4)}</>}.
            </>
          )}
        </p>
        <p className="byline">Version 1 · basic physics model · after Maitra &amp; Gururajan, Sci. Rep. 15:3008 (2025)</p>
        <StatusStrip progress={progress} offline={offline} onRun={run} onCancel={cancel} />
      </header>

      <DatasetPanel version={version} selected={selected} onSelect={select} />

      <div id="trace">
        <TracePanel version={version} selected={selected} onSelect={setSelected} available={available} events={events} />
      </div>

      <AveragePanel version={version} defaultWindow={cfg?.settings.force_rate_window_ps ?? 100} paperRate={paperRate} onRate={setRate} />

      <EventsPanel version={version} defaultSmoothing={cfg?.settings.peak_smoothing_points ?? 15} fc={fc} onEvents={setEvents} />

      {cfg && (
        <LandscapePanel
          paramSets={cfg.paper_parameter_sets}
          defaultSet={cfg.settings.default_parameter_set}
          forceRate={rate}
          meanCritForce={events?.f_crit_from_rate_pN.mean ?? null}
          onFc={setFc}
        />
      )}

      <SummaryPanel summary={summary.data} progress={progress} />

      <p className="footnote">
        Data: <code>data/raw/Force_traces/property.T300.*.dat</code> — time [ps], box length Lₓ [Å], force per atom [pN]. Lₓ is the simulation box, not the
        landscape coordinate ξ; the two are never used interchangeably here. Backend: FastAPI + NumPy, <code>backend/tini</code>. Checkpoints and static charts:{' '}
        <code>artifacts/</code>.
      </p>
    </>
  )
}
