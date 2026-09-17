import { useMemo, useState } from 'react'
import { Bar, ComposedChart, ErrorBar, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip } from 'recharts'
import type { InvDiagnostics, InvResult, InvSample } from '../../api/inverse'
import { fixed, pval } from '../../lib/format'
import { Empty, Figure, Grid, Panel, Readout, Seg, Tip, XNum, YNum, chart } from '../ui'

/* E. The fitted model against the observed sample — density, cumulative,
 * residuals and quantiles — with the paper's parameter sets drawn for
 * comparison. Everything terracotta is measured; everything slate is a model. */
export function ComparisonPanel({ result, diag, sample }: { result: InvResult | null; diag: InvDiagnostics | null; sample: InvSample | null }) {
  const [paperKey, setPaperKey] = useState<string>('pdf_mle')
  const fit = diag?.fitted
  const paper = diag?.paper?.[paperKey]

  const dens = useMemo(() => {
    if (!fit) return []
    const h = fit.histogram
    return h.centers.map((x, i) => ({ x, d: h.density[i], err: h.density_err[i] }))
  }, [fit])
  const pdfFit = useMemo(() => (fit ? fit.pdf_grid.map((x, i) => ({ x, p: fit.pdf[i] })) : []), [fit])
  const pdfPaper = useMemo(() => (paper ? paper.pdf_grid.map((x, i) => ({ x, q: paper.pdf[i] })) : []), [paper])
  const cdfFit = useMemo(() => (fit ? fit.pdf_grid.map((x, i) => ({ x, c: fit.cdf[i] })) : []), [fit])
  const cdfPaper = useMemo(() => (paper ? paper.pdf_grid.map((x, i) => ({ x, cq: paper.cdf[i] })) : []), [paper])
  const ecdf = useMemo(() => (sample ? sample.ecdf.x.map((x, i) => ({ x, F: sample.ecdf.F[i] })) : []), [sample])
  const resid = useMemo(() => (fit ? fit.histogram.centers.map((x, i) => ({ x, r: fit.residuals_poisson[i], rq: paper?.residuals_poisson[i] ?? null })) : []), [fit, paper])
  const qq = useMemo(() => (fit ? fit.qq.model.map((m, i) => ({ m, d: fit.qq.data[i] })) : []), [fit])
  const xRange: [number, number] = result ? [Math.floor(result.sample.min_pN / 5) * 5 - 5, Math.ceil(Math.max(result.sample.max_pN, result.fit.derived.f_c_pN) / 5) * 5] : [0, 40]

  const lead = <>Does the fitted density describe the sample? Four views of one question, with the paper's parameters overlaid in lighter slate.</>
  if (!result || !fit) {
    return (
      <Panel index="E" title="Model against observations" lead={lead}>
        <Empty>No run to compare yet.</Empty>
      </Panel>
    )
  }
  const g = result.goodness_of_fit
  const pc = result.paper_comparison[paperKey]

  return (
    <Panel index="E" title="Model against observations" lead={lead}>
      <div className="controls">
        <div className="control">
          <span className="control__label">
            <span>paper parameters overlaid</span>
          </span>
          <Seg
            value={paperKey}
            onChange={setPaperKey}
            options={[
              { value: 'pdf_mle', label: 'Table 1, p(f|ḟ)' },
              { value: 'mean_lsq', label: 'Table 1, ⟨f⟩(ḟ)' },
            ]}
          />
        </div>
        <span className="figure__note" style={{ alignSelf: 'center', marginTop: 0 }}>
          terracotta measured · slate fitted · light slate paper
        </span>
      </div>

      <div className="grid-2">
        <Figure title="density" note="Bars ± √n; the paper's curve uses Table 1 values at this dataset's ḟ.">
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={dens} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xRange} />
                <YNum label="density [1/pN]" />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.dataKey === 'd' ? 'sample' : p.dataKey === 'p' ? 'fit' : 'paper'} ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="d" fill={chart.accent} fillOpacity={0.5} isAnimationActive={false} radius={[2, 2, 0, 0]}>
                  <ErrorBar dataKey="err" width={3} stroke={chart.accent} strokeWidth={1} />
                </Bar>
                <Line data={pdfPaper} dataKey="q" type="monotone" stroke={chart.model2} strokeWidth={1.2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line data={pdfFit} dataKey="p" type="monotone" stroke={chart.model} strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure title="cumulative" note={`Kolmogorov–Smirnov D = ${fixed(g.ks_statistic, 3)}, p = ${pval(g.ks_pvalue)} (fit); D = ${fixed(pc?.ks_statistic, 3)}, p = ${pval(pc?.ks_pvalue)} (paper).`}>
          <div className="chart chart--tall">
            <ResponsiveContainer>
              <ComposedChart data={ecdf} margin={chart.margin}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xRange} />
                <YNum label="P(f ≤ f‡)" range={[0, 1]} />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(2)} rows={(p) => `${p.dataKey === 'F' ? 'sample' : p.dataKey === 'c' ? 'fit' : 'paper'} ${typeof p.value === 'number' ? p.value.toFixed(3) : ''}`} />} />
                <Line dataKey="F" type="stepAfter" stroke={chart.accent} strokeWidth={1.4} dot={false} isAnimationActive={false} />
                <Line data={cdfPaper} dataKey="cq" type="monotone" stroke={chart.model2} strokeWidth={1.2} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line data={cdfFit} dataKey="c" type="monotone" stroke={chart.model} strokeWidth={1.8} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <div className="grid-2" style={{ marginTop: 26 }}>
        <Figure title="residuals per bin, (observed − expected) / √expected" note={`Pearson χ² = ${fixed(g.chi2, 2)}, ${g.chi2_dof} d.o.f., p = ${pval(g.chi2_pvalue)}.`}>
          <div className="chart">
            <ResponsiveContainer>
              <ComposedChart data={resid} margin={chart.margin} barCategoryGap={4}>
                <Grid />
                <XNum dataKey="x" label="f‡ [pN]" range={xRange} />
                <YNum label="σ" />
                <Tooltip content={<Tip unitX="pN" fmtX={(v) => v.toFixed(1)} rows={(p) => `${p.dataKey === 'r' ? 'fit' : 'paper'} ${typeof p.value === 'number' ? p.value.toFixed(2) : ''} σ`} />} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <ReferenceLine y={0} stroke={chart.ink4} />
                <ReferenceLine y={2} stroke={chart.grid} strokeDasharray="3 3" />
                <ReferenceLine y={-2} stroke={chart.grid} strokeDasharray="3 3" />
                <Bar dataKey="rq" fill={chart.model2} fillOpacity={0.35} isAnimationActive={false} />
                <Bar dataKey="r" fill={chart.model} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
        <Figure title="quantile–quantile" note="On the diagonal, the fit reproduces the sample at every quantile.">
          <div className="chart">
            <ResponsiveContainer>
              <ComposedChart data={qq} margin={chart.margin}>
                <Grid />
                <XNum dataKey="m" label="model quantile [pN]" range={xRange} />
                <YNum label="sample [pN]" range={xRange} />
                <Tooltip content={<Tip unitX="pN model" fmtX={(v) => v.toFixed(2)} rows={(p) => `sample ${typeof p.value === 'number' ? p.value.toFixed(2) : ''} pN`} />} />
                <Line data={[{ m: xRange[0], d: xRange[0] }, { m: xRange[1], d: xRange[1] }]} dataKey="d" stroke={chart.ink4} strokeWidth={1} dot={false} isAnimationActive={false} />
                <Scatter dataKey="d" fill={chart.accent} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Figure>
      </div>

      <div className="readouts">
        <Readout label="KS p (fit)" value={pval(g.ks_pvalue)} tone="model" />
        <Readout label="χ² p (fit)" value={pval(g.chi2_pvalue)} tone="model" />
        <Readout label="AIC" value={fixed(g.aic, 1)} />
        <Readout label="fit ⟨f⟩, s.d." value={`${fixed(g.model_mean_exact_pN, 2)}, ${fixed(g.model_sd_exact_pN, 2)}`} unit="pN" tone="model" />
        <Readout label="sample ⟨f⟩, s.d." value={`${fixed(g.sample_mean_pN, 2)}, ${fixed(g.sample_sd_pN, 2)}`} unit="pN" tone="accent" />
        {pc && <Readout label="paper ⟨f⟩, s.d." value={`${fixed(pc.model_mean_exact_pN, 2)}, ${fixed(pc.model_sd_exact_pN, 2)}`} unit="pN" />}
        {pc && <Readout label="LR test, paper vs fit" value={`${fixed(pc.likelihood_ratio_vs_fit.statistic, 1)} · p ${pval(pc.likelihood_ratio_vs_fit.pvalue)}`} unit={`${pc.likelihood_ratio_vs_fit.df} d.o.f.`} />}
      </div>
    </Panel>
  )
}
