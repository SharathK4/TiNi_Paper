import katex from 'katex'
import { useMemo } from 'react'

export function M({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: block, throwOnError: false, strict: false }), [tex, block])
  return <span className={block ? 'eq__math' : undefined} dangerouslySetInnerHTML={{ __html: html }} />
}

/** Frequently used symbols, so plain text never carries combining marks or angle brackets
 * (the system text font substitutes them). */
export const Fdot = () => <M tex="\dot f" />
export const Fc = () => <M tex="f_c" />
export const MeanF = () => <M tex="\langle f^\ddagger\rangle" />
export const MeanT = () => <M tex="\langle t^\ddagger\rangle" />
export const Lx = () => <M tex="L_x" />

export function Eq({ tex, tag, children }: { tex: string; tag?: string; children?: React.ReactNode }) {
  return (
    <div className="eq">
      {tag && <div className="eq__tag">{tag}</div>}
      <M tex={tex} block />
      {children}
    </div>
  )
}
