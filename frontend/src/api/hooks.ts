import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, api, type Progress } from './client'

/** Polls /api/progress: every second while running, every 5 s otherwise. */
export function useProgress() {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [offline, setOffline] = useState(false)
  const timer = useRef<number | null>(null)

  const tick = useCallback(async () => {
    try {
      const p = await api.progress()
      setProgress(p)
      setOffline(false)
      return p.running
    } catch {
      setOffline(true)
      return false
    }
  }, [])

  useEffect(() => {
    let alive = true
    const loop = async () => {
      const running = await tick()
      if (!alive) return
      timer.current = window.setTimeout(loop, running ? 1000 : 5000)
    }
    loop()
    return () => {
      alive = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [tick])

  const run = useCallback(
    async (force = false) => {
      await api.run(force)
      await tick()
    },
    [tick],
  )
  const cancel = useCallback(async () => {
    await api.cancel()
    await tick()
  }, [tick])

  return { progress, offline, run, cancel, refresh: tick }
}

type State<T> = { data: T | null; error: string | null; loading: boolean; missing: boolean }

/**
 * Loads a resource, re-fetching when `deps` change or when `version` bumps
 * (the page bumps it after a pipeline run finishes so stale panels refresh).
 * A 404 from the API means "not processed yet" and is reported as `missing`.
 */
export function useResource<T>(loader: () => Promise<T>, deps: unknown[], version = 0): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true, missing: false })
  useEffect(() => {
    let alive = true
    setState((s) => ({ ...s, loading: true }))
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false, missing: false }))
      .catch((e: unknown) => {
        if (!alive) return
        const missing = e instanceof ApiError && e.status === 404
        setState((s) => ({ data: missing ? null : s.data, error: missing ? null : String((e as Error).message), loading: false, missing }))
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, version])
  return state
}

/** Debounce a fast-changing value (slider) before it hits the network. */
export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}
