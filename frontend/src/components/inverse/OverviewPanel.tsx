import type { InvResult, InvSpec } from '../../api/inverse'
import { fixed } from '../../lib/format'
import { M } from '../Math'
import { Empty, Panel } from '../ui'
import { PARAM_KEYS, PARAM_LABEL, PARAM_UNIT, fmtParam, interval } from './shared'

/* A. The question, and the answer if there is one. */
export function OverviewPanel({ spec, result }: { spec: InvSpec | null; result: InvResult | null }) {
  return (
    <Panel
      index="2.1"
      title="The inverse problem"
      lead={
        <>
          The model has three numbers — a barrier <M tex="U^\ddagger" />, its position <M tex="\xi^\ddagger" />, a rate prefactor <M tex="\Gamma_0" />. Given the
          forces at which one hundred crystals snapped, which values make that sample most probable?
        </>
      }
    >
      {result ? (
        <div className="headline">
          <div className="headline__label">
            maximum likelihood from {result.sample.n} transition forces at one loading rate · 95 % profile-likelihood intervals · paper's Table 1 beneath
          </div>
          <div className="headline__row">
            {PARAM_KEYS.map((k) => {
              const p = result.parameters[k]
              return (
                <div className="headline__item" key={k}>
                  <div className="headline__name">{PARAM_LABEL[k]}</div>
                  <div className="headline__value num">
                    {fmtParam(k, p.mle)} <small>{PARAM_UNIT[k]}</small>
                  </div>
                  <div className="headline__ci num">{interval(k, p.ci95_profile[0], p.ci95_profile[1], p.profile_unbounded, p.profile_support_limited)}</div>
                  <div className="headline__paper num">paper {fmtParam(k, p.paper)}</div>
                </div>
              )
            })}
            <div className="headline__item">
              <div className="headline__name">
                <M tex="f_c" />
              </div>
              <div className="headline__value num">
                {fixed(result.fit.derived.f_c_pN, 1)} <small>pN</small>
              </div>
              <div className="headline__ci num">
                {fixed(result.bootstrap.f_c_pN?.ci95[0], 1)} – {fixed(result.bootstrap.f_c_pN?.ci95[1], 0)} bootstrap
              </div>
              <div className="headline__paper num">barrier {fixed(result.fit.derived.U_barrier_kBT, 1)} <M tex="k_BT" /></div>
            </div>
          </div>
          <p className="headline__note">
            One loading rate fixes where the distribution sits and how wide it is — two facts for three parameters. <M tex="\Gamma_0" /> is therefore known to an
            order of magnitude and <M tex="U^\ddagger" /> is open above. Section 2.6 makes that quantitative.
          </p>
        </div>
      ) : (
        <Empty>No inference has been run yet — section 2.4.</Empty>
      )}

      <table className="facts facts--wide" style={{ marginTop: 56 }}>
        <tbody>
          <tr>
            <td>Data</td>
            <td>
              <M tex="f^\ddagger_i = \dot f\,t^\ddagger_i" />, the applied force at each trace's transition (paper, Methods), from Version 1
            </td>
          </tr>
          <tr>
            <td>Known</td>
            <td>
              <M tex="k_BT" /> = {spec ? fixed(spec.kBT_pN_A, 2) : '41.42'} pN·Å · <M tex="\dot f" /> measured in Version 1
            </td>
          </tr>
          <tr>
            <td>Unknown</td>
            <td>
              <M tex="U^\ddagger" /> [pN·Å], <M tex="\xi^\ddagger" /> [Å], <M tex="\Gamma_0" /> [ps⁻¹]
            </td>
          </tr>
          <tr>
            <td>Method</td>
            <td>
              Paper Eq. (1): <M tex="L(w) = \prod_i p(f^\ddagger_i \mid \dot f;\, w)" /> with the density of Eq. (6), maximised by Nelder–Mead
            </td>
          </tr>
          <tr>
            <td>Rests on</td>
            <td>
              the paper's overdamped, absorbing-boundary model; independent events; a linear ramp; and the Version 1 peak picker, which is not the paper's and
              is varied in section 2.6
            </td>
          </tr>
        </tbody>
      </table>
    </Panel>
  )
}
