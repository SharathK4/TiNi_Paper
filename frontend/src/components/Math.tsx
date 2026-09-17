import katex from 'katex'
import { useMemo } from 'react'

export function M({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: block, throwOnError: false, strict: false }), [tex, block])
  return <span className={block ? 'eq__math' : undefined} dangerouslySetInnerHTML={{ __html: html }} />
}

export function Eq({ tex, tag, children }: { tex: string; tag?: string; children?: React.ReactNode }) {
  return (
    <div className="eq">
      {tag && <div className="eq__tag">{tag}</div>}
      <M tex={tex} block />
      {children}
    </div>
  )
}
