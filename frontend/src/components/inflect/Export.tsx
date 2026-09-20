/**
 * Getting the work out of the page and into a paper.
 *
 * The figure leaves as SVG rather than a screenshot, so it stays sharp at any
 * size and its text stays text. The chart on screen is already plain SVG, so
 * what downloads is the thing that was being looked at — with the design
 * tokens resolved, because a stylesheet does not travel with the file.
 *
 * The methods paragraph is written from the settings actually used. It is the
 * part of a paper nobody enjoys writing and the part a reader needs most.
 */

import { useCallback, useState } from 'react'
import { inflectMore } from '../../api/inflect'

/** Copy the computed value of every custom property onto the element itself. */
function inlineTokens(source: SVGSVGElement, clone: SVGSVGElement): void {
  const paint = ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-opacity',
    'stroke-linejoin', 'stroke-linecap', 'opacity', 'font-size', 'font-family',
    'font-weight', 'letter-spacing', 'text-anchor', 'text-transform', 'paint-order']
  const from = source.querySelectorAll<SVGElement>('*')
  const to = clone.querySelectorAll<SVGElement>('*')
  for (let i = 0; i < from.length && i < to.length; i++) {
    const computed = getComputedStyle(from[i])
    const parts: string[] = []
    for (const prop of paint) {
      const v = computed.getPropertyValue(prop)
      if (v && v !== 'none' && v !== 'normal' && v !== 'auto') parts.push(`${prop}:${v}`)
    }
    // Animations end with the mark drawn; a still has to start there.
    parts.push('stroke-dashoffset:0')
    to[i].setAttribute('style', parts.join(';'))
    to[i].removeAttribute('class')
  }
}

function download(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export function Export({ sessionId, file }: { sessionId: string; file: string | null }) {
  const [methods, setMethods] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const saveFigure = useCallback(() => {
    const source = document.querySelector<SVGSVGElement>('.stage__svg')
    if (!source) {
      setError('no figure on screen to save')
      return
    }
    const clone = source.cloneNode(true) as SVGSVGElement
    inlineTokens(source, clone)
    clone.removeAttribute('class')
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    bg.setAttribute('width', '100%')
    bg.setAttribute('height', '100%')
    bg.setAttribute('fill', getComputedStyle(document.body).backgroundColor || '#ffffff')
    clone.insertBefore(bg, clone.firstChild)
    const text = `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`
    const stem = (file ?? 'figure').replace(/\.[^.]+$/, '')
    download(`${stem}.svg`, new Blob([text], { type: 'image/svg+xml;charset=utf-8' }))
  }, [file])

  const showMethods = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const r = await inflectMore.methods(sessionId, file ?? undefined)
      setMethods(r.methods)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [sessionId, file])

  return (
    <section className="export">
      <h3>Take it with you</h3>
      <div className="export__row">
        <button className="btn btn--quiet" onClick={saveFigure}>figure as SVG</button>
        <a className="btn btn--quiet" href={inflectMore.csvUrl(sessionId)} download>
          results as CSV
        </a>
        <button className="btn btn--quiet" onClick={showMethods} disabled={busy}>
          {busy ? 'writing…' : 'methods paragraph'}
        </button>
      </div>
      {error && <p className="intake__error">{error}</p>}
      {methods && (
        <div className="export__methods">
          <p>{methods}</p>
          <button
            className="btn btn--primary"
            onClick={() => {
              navigator.clipboard?.writeText(methods).then(
                () => {
                  setCopied(true)
                  setTimeout(() => setCopied(false), 2200)
                },
                () => setError('the browser would not let the page copy to the clipboard'),
              )
            }}
          >
            {copied ? 'copied' : 'copy'}
          </button>
        </div>
      )}
    </section>
  )
}

/** Every file as a thumbnail: which ones differ is visible without reading. */
export function SparkGrid({
  sparklines,
  names,
  current,
  odd,
  bad,
  onOpen,
}: {
  sparklines: Record<string, number[]>
  names: string[]
  current: string | null
  odd: Set<string>
  bad: Set<string>
  onOpen: (name: string) => void
}) {
  return (
    <div className="sparks">
      {names.map((name) => {
        const v = sparklines[name]
        const path = v && v.length
          ? v.map((p, i) => `${i ? 'L' : 'M'}${((i / (v.length - 1)) * 40).toFixed(1)} ${(14 - p * 12).toFixed(1)}`).join('')
          : ''
        const short = name.replace(/\.[^.]+$/, '').slice(-5)
        return (
          <button
            key={name}
            className={`spark ${name === current ? 'is-on' : ''} ${odd.has(name) ? 'is-odd' : ''} ${bad.has(name) ? 'is-bad' : ''}`}
            onClick={() => onOpen(name)}
            title={name}
          >
            <svg viewBox="0 0 40 16" preserveAspectRatio="none" aria-hidden="true">
              {path ? <path d={path} /> : null}
            </svg>
            <span className="spark__name num">{short}</span>
          </button>
        )
      })}
    </div>
  )
}
