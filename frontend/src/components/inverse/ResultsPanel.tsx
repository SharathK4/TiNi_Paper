import { useMemo } from 'react'
import { ComposedChart, Line, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, YAxis } from 'recharts'
import type { InvBootstrap, InvParams, InvProfiles, InvResult } from '../../api/inverse'
import { fixed, pct, pval, sci } from '../../lib/format'
import { M } from '../Math'
import { Empty, Figure, Grid, Panel, Tip, chart } from '../ui'
import { PARAM_KEYS, PARAM_LABEL, PARAM_UNIT, Table, XLog, fmtParam, interval } from './shared'

/* F. The inferred parameters with their uncertainty, the shape of the
 * likelihood around them, what changes when the V1 choices change, and what
 * a synthetic experiment says the design can and cannot determine. */
export function ResultsPanel({ result, profiles, boot }: { result: InvResult | null; profiles: InvProfiles | null; boot: InvBootstrap | null }) {
  const lead = <>Three numbers with three kinds of uncertainty — and which of them this experiment can actually determine.</>
  if (!result) {
    return (
      <Panel index="F" title="What was inferred" lead={lead}>
        <Empty>No run yet.</Empty>
      </Panel>
    )
  }
  const P = result.parameters
  const syn = 'skipped' in result.synthetic ? null : result.synthetic

  return (
    <Panel index="F" title="What was inferred" lead={lead}>
      <div className="figure__title">parameters · maximum likelihood with three uncertainty estimates</div>
      <Table
        head={['parameter', 'MLE', '± s.e. (Fisher)', '95 % profile likelihood', '95 % bootstrap', 'paper, Table 1', 'unit']}
        rows={PARAM_KEYS.map((k) => {
          const p = P[k]
          const hit = p.bootstrap_hit_bounds?.[1] ?? 0
          return [
            PARAM_LABEL[k],
            <b className="num">{fmtParam(k, p.mle)}</b>,
            <span className="num">{p.se_asymptotic !== null ? fmtParam(k, p.se_asymptotic) : '—'}</span>,
            <span className="num">{interval(k, p.ci95_profile[0], p.ci95_profile[1], p.profile_unbounded, p.profile_support_limited)}</span>,
            <span className="num">
              {p.ci95_bootstrap ? interval(k, p.ci95_bootstrap[0], p.ci95_bootstrap[1]) : '—'}
              {hit > 0.02 && <span className="tag tag--prototype">{pct(hit)} at bound</span>}
            </span>,
            <span className="num">{fmtParam(k, p.paper)}</span>,
            PARAM_UNIT[k],
          ]
        })}
      />
      <div className="readouts" style={{ marginTop: 14 }}>
        <div className="readout readout--model">
          <div className="readout__label">
            <M tex="f_c" />
          </div>
          <div className="readout__value num">
            {fixed(result.fit.derived.f_c_pN, 2)}
            <small>pN · bootstrap {fixed(result.bootstrap.f_c_pN?.ci95[0], 1)} – {fixed(result.bootstrap.f_c_pN?.ci95[1], 0)}</small>
          </div>
        </div>
        <div className="readout readout--model">
          <div className="readout__label">
            <M tex="U^\ddagger/k_BT" />
          </div>
          <div className="readout__value num">{fixed(result.fit.derived.U_barrier_kBT, 2)}</div>
        </div>
        <div className="readout">
          <div className="readout__label">Hessian condition number</div>
          <div className="readout__value num">{sci(result.hessian.condition_number, 1)}</div>
        </div>
        <div className="readout">
          <div className="readout__label">corr(ln ξ‡, ln Γ₀)</div>
          <div className="readout__value num">{fixed(result.hessian.correlation?.[1]?.[2], 2)}</div>
        </div>
        <div className="readout">
          <div className="readout__label">converged</div>
          <div className="readout__value">{result.fit.converged ? 'yes' : 'no'}</div>
        </div>
      </div>
      <p className="figure__note">
        Quote the profile interval. "Unbounded": the profile never rises by 1.92 within the scan; "support": the scan reached <M tex="f_c \le \max f^\ddagger" />;
        "at bound": bootstrap refits that ran along the ridge to the box edge.
      </p>

      {profiles && (
        <div style={{ marginTop: 30 }}>
          <div className="figure__title">profile likelihoods · 2 ΔlnL against each parameter, others re-optimised</div>
          <div className="grid-3">
            {PARAM_KEYS.map((k) => (
              <ProfileChart key={k} k={k} prof={profiles} mle={P[k].mle} paper={P[k].paper} />
            ))}
          </div>
          <p className="figure__note">Dashed: the 95 % threshold. A flat profile is a parameter the data do not determine. Grey marker: the paper's value.</p>
        </div>
      )}

      <div className="grid-2" style={{ marginTop: 30, alignItems: 'start' }}>
        {boot && (
          <Figure
            title={
              <>
                bootstrap scatter · <M tex="\Gamma_0" /> against <M tex="\xi^\ddagger" />, {boot.n_ok} refits
              </>
            }
            note="One point per resample. The band is the single-rate ridge: a larger ξ‡ with a smaller Γ₀ fits about as well. Black: this fit; grey: the paper."
          >
            <BootScatter boot={boot} mle={result.fit.params} paper={{ U_barrier: P.U_barrier.paper, xi_barrier: P.xi_barrier.paper, gamma0: P.gamma0.paper }} />
          </Figure>
        )}
        <div>
          <div className="figure__title">constrained fits · one parameter held at the paper's value</div>
          <Table
            head={['held fixed', 'U‡', 'ξ‡', 'Γ₀', 'Δ(−ln L)', 'LR p']}
            rows={result.constrained.map((c) => [
              c.label.replace(' fixed at Table 1', ''),
              <span className="num">{c.fixed.U_barrier !== undefined ? <i>{fmtParam('U_barrier', c.params.U_barrier)}</i> : fmtParam('U_barrier', c.params.U_barrier)}</span>,
              <span className="num">{c.fixed.xi_barrier !== undefined ? <i>{fmtParam('xi_barrier', c.params.xi_barrier)}</i> : fmtParam('xi_barrier', c.params.xi_barrier)}</span>,
              <span className="num">{c.fixed.gamma0 !== undefined ? <i>{fmtParam('gamma0', c.params.gamma0)}</i> : fmtParam('gamma0', c.params.gamma0)}</span>,
              <span className="num">{fixed(c.delta_nll_vs_unconstrained, 2)}</span>,
              <span className="num">{pval(c.likelihood_ratio.pvalue)}</span>,
            ])}
          />
          <p className="figure__note">Italic = held. Holding any one parameter at the paper's value costs under one unit of log-likelihood.</p>

          <div className="figure__title" style={{ marginTop: 24 }}>
            sensitivity to the Version 1 prototype choices
          </div>
          <Table
            head={['varied', 'f‡ from', 'ḟ window', 'smoothing', 'ḟ', 'sample ⟨f‡⟩', 'U‡', 'ξ‡', 'Γ₀', 'f_c']}
            rows={result.sensitivity.map((v) => [
              v.axis,
              v.source === 'rate_x_time' ? 'ḟ·t‡' : 'peak',
              `${v.window_ps} ps`,
              `${v.smoothing_points} pts`,
              <span className="num">{fixed(v.force_rate, 4)}</span>,
              <span className="num">{fixed(v.sample_mean, 2)}</span>,
              <span className="num">{fmtParam('U_barrier', v.params.U_barrier)}</span>,
              <span className="num">{fmtParam('xi_barrier', v.params.xi_barrier)}</span>,
              <span className="num">{fmtParam('gamma0', v.params.gamma0)}</span>,
              <span className="num">{fixed(v.f_c_pN, 1)}</span>,
            ])}
          />
          <p className="figure__note">The ḟ window barely matters; the peak picker does — it moves the fit by about the whole gap to Table 1.</p>
        </div>
      </div>

      {syn && (
        <div style={{ marginTop: 30 }}>
          <div className="figure__title">
            synthetic recovery · truth = Table 1 ({result.config.reference_set === 'pdf_mle' ? 'p(f|ḟ) fit' : '⟨f⟩(ḟ) fit'}), {syn.replicates} replicates per design
          </div>
          <Table
            head={['design', 'events', 'U‡ median / truth', 'U‡ spread (decades, 16–84 %)', 'ξ‡ median / truth', 'ξ‡ spread', 'Γ₀ median / truth', 'Γ₀ spread', 'Γ₀ within ×2']}
            rows={syn.designs.map((d) => [
              d.name,
              d.n_events,
              <span className="num">{fixed(d.summary.U_barrier?.median_ratio, 2)}</span>,
              <span className="num">{fixed(d.summary.U_barrier?.log10_spread_16_84, 2)}</span>,
              <span className="num">{fixed(d.summary.xi_barrier?.median_ratio, 2)}</span>,
              <span className="num">{fixed(d.summary.xi_barrier?.log10_spread_16_84, 2)}</span>,
              <span className="num">{fixed(d.summary.gamma0?.median_ratio, 2)}</span>,
              <span className="num">{fixed(d.summary.gamma0?.log10_spread_16_84, 2)}</span>,
              <span className="num">{pct(d.summary.gamma0?.within_factor_2)}</span>,
            ])}
          />
          <p className="figure__note">
            Data simulated from the model itself. With this dataset's design U‡ and ξ‡ come back within ±20 % but Γ₀ scatters over a decade; the paper's four-rate
            design pins all three.
          </p>
        </div>
      )}

      <p className="prose" style={{ marginTop: 44, color: 'var(--ink-2)' }}>
        Quote <M tex="f_c" />, <M tex="U^\ddagger" /> and <M tex="\xi^\ddagger" />; treat <M tex="\Gamma_0" /> as an order of magnitude. The events come from a
        prototype peak picker whose bias, about one pN, accounts for the gap to Table 1.
      </p>
    </Panel>
  )
}

function ProfileChart({ k, prof, mle, paper }: { k: keyof InvParams; prof: InvProfiles; mle: number; paper: number }) {
  const p = prof.profiles[k]
  const data = useMemo(() => p.grid.filter((r) => r.feasible && r.delta_nll !== null).map((r) => ({ x: r.value, y: 2 * (r.delta_nll as number) })), [p])
  const lo = p.scan_range[0]
  const hi = p.scan_range[1]
  const ymax = Math.max(6, ...data.map((d) => d.y).filter((y) => y < 30))
  return (
    <Figure title={<>{PARAM_LABEL[k]} · 2 ΔlnL</>}>
      <div className="chart chart--short">
        <ResponsiveContainer>
          <ComposedChart data={data} margin={chart.margin}>
            <Grid />
            <XLog dataKey="x" label={PARAM_UNIT[k]} lo={lo} hi={hi} fmt={(v) => fmtParam(k, v, k === 'gamma0' ? 0 : undefined)} />
            <YAxis type="number" domain={[0, Math.ceil(ymax)]} tick={{ fontSize: 11, fill: '#9698a8' }} tickLine={false} axisLine={false} width={40} />
            <Tooltip content={<Tip unitX={PARAM_UNIT[k]} fmtX={(v) => fmtParam(k, v)} rows={(q) => `2ΔlnL = ${typeof q.value === 'number' ? q.value.toFixed(2) : ''}`} />} />
            <ReferenceLine y={prof.chi2_1_95} stroke={chart.ink3} strokeDasharray="4 3" />
            <Line dataKey="y" type="monotone" stroke={chart.model} strokeWidth={1.6} dot={{ r: 2, fill: chart.model, strokeWidth: 0 }} isAnimationActive={false} />
            {paper >= lo && paper <= hi && <ReferenceDot x={paper} y={0} r={4} fill={chart.ink3} stroke="#fff" strokeWidth={1.5} />}
            <ReferenceDot x={mle} y={0} r={4} fill={chart.ink} stroke="#fff" strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Figure>
  )
}

function BootScatter({ boot, mle, paper }: { boot: InvBootstrap; mle: InvParams; paper: InvParams }) {
  const pts = useMemo(() => boot.draws.map((d) => ({ x: d.xi_barrier, y: d.gamma0 })), [boot])
  const xs = pts.map((p) => p.x)
  const ys = pts.map((p) => p.y)
  const xlo = Math.min(...xs, mle.xi_barrier, paper.xi_barrier) * 0.8
  const xhi = Math.max(...xs, mle.xi_barrier, paper.xi_barrier) * 1.25
  const ylo = Math.min(...ys, mle.gamma0, paper.gamma0) / 3
  const yhi = Math.max(...ys, mle.gamma0, paper.gamma0) * 3
  return (
    <div className="chart chart--tall">
      <ResponsiveContainer>
        <ComposedChart data={pts} margin={chart.margin}>
          <Grid />
          <XLog dataKey="x" label="ξ‡ [Å]" lo={xlo} hi={xhi} fmt={(v) => v.toFixed(0)} />
          <YAxis type="number" dataKey="y" scale="log" domain={[ylo, yhi]} tickFormatter={(v: number) => sci(v, 0)} tick={{ fontSize: 11, fill: '#9698a8' }} tickLine={false} axisLine={false} width={64} label={{ value: 'Γ₀ [ps⁻¹]', angle: -90, position: 'insideLeft', offset: 8, fontSize: 11, fill: '#6e7180' }} />
          <Tooltip content={<Tip unitX="Å" fmtX={(v) => v.toFixed(2)} rows={(p) => `Γ₀ = ${typeof p.value === 'number' ? sci(p.value, 2) : ''}`} />} />
          <Scatter dataKey="y" fill={chart.model} fillOpacity={0.35} isAnimationActive={false} />
          <ReferenceDot x={paper.xi_barrier} y={paper.gamma0} r={5} fill={chart.ink3} stroke="#fff" strokeWidth={1.5} />
          <ReferenceDot x={mle.xi_barrier} y={mle.gamma0} r={5} fill={chart.ink} stroke="#fff" strokeWidth={1.5} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
