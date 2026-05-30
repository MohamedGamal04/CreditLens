/* ===================================================================
   PAGE 2 — Explanation panel: SHAP waterfall + plain-language reasons
   =================================================================== */
function ExplanationPage({ applicant, result, thresholds, model, go }) {
  const sorted = [...result.steps].sort((a, b) => Math.abs(b.dp) - Math.abs(a.dp));
  const raisers = sorted.filter(s => s.dp > 0).slice(0, 3);
  const lowerers = sorted.filter(s => s.dp < 0).slice(0, 3);
  const verb = { low: 'is recommended for approval', med: 'requires manual review', high: 'is flagged for decline' }[result.band];
  const m = CL.getModel(model);

  return (
    <div className="page-wrap rise">
      {/* verdict banner */}
      <div className="card" style={{ marginBottom: 18, background: RISK[result.band].soft, borderColor: 'transparent' }}>
        <div className="card-pad" style={{ display: 'flex', alignItems: 'center', gap: 20, padding: '20px 24px' }}>
          <div style={{ width: 52, height: 52, borderRadius: 13, background: RISK[result.band].c, display: 'grid', placeItems: 'center', color: '#fff', flex: 'none' }}>
            <I name="explain" size={26} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, fontWeight: 650, color: RISK[result.band].s, marginBottom: 3, display: 'flex', alignItems: 'center', gap: 8 }}>Why this decision
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: 'var(--ink-3)', fontWeight: 550 }}><ModelGlyph color={m.color} size={14} />{m.name}</span></div>
            <div style={{ fontSize: 17, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.01em', textWrap: 'pretty' }}>
              Applicant <b>SK_ID_CURR 100002</b> {verb} — estimated default probability of <span className="mono" style={{ color: RISK[result.band].s }}>{pct(result.prob, 1)}</span>.
            </div>
          </div>
          <button className="btn btn-ghost" onClick={() => go('applicant')}>Back to profile</button>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.55fr) minmax(300px, 1fr)', alignItems: 'start' }}>
        {/* waterfall */}
        <div className="card">
          <div className="card-head">
            <div className="card-title">Contribution waterfall<div className="sub">How each factor moves probability from the portfolio baseline to this estimate</div></div>
            <div className="spacer" />
            <div className="legend">
              <span className="li"><span className="sw" style={{ background: 'var(--high)' }} />Raises risk</span>
              <span className="li"><span className="sw" style={{ background: 'var(--low)' }} />Lowers risk</span>
            </div>
          </div>
          <div className="card-pad">
            <Waterfall result={result} width={680} />
          </div>
        </div>

        {/* plain-language */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="card">
            <div className="card-head"><div className="card-title">In plain language</div></div>
            <div className="card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 18, fontSize: 13.5, lineHeight: 1.5 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: 'var(--low-strong)', fontWeight: 650, fontSize: 12.5, marginBottom: 9 }}>
                  <I name="check" size={15} /> IN THEIR FAVOR</div>
                <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {lowerers.map(s => (
                    <li key={s.key} style={{ display: 'flex', gap: 9, color: 'var(--ink-2)' }}>
                      <span style={{ color: 'var(--low)', fontWeight: 700 }}>▼</span>
                      <span><b>{s.label}</b> of <span className="mono">{s.display}</span> lowers risk by <span className="mono" style={{ color: 'var(--low-strong)', fontWeight: 600 }}>{pct(Math.abs(s.dp),1)}</span></span>
                    </li>
                  ))}
                </ul>
              </div>
              <hr className="hr" />
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: 'var(--high-strong)', fontWeight: 650, fontSize: 12.5, marginBottom: 9 }}>
                  <I name="info" size={15} /> ADDING RISK</div>
                <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {raisers.length ? raisers.map(s => (
                    <li key={s.key} style={{ display: 'flex', gap: 9, color: 'var(--ink-2)' }}>
                      <span style={{ color: 'var(--high)', fontWeight: 700 }}>▲</span>
                      <span><b>{s.label}</b> of <span className="mono">{s.display}</span> raises risk by <span className="mono" style={{ color: 'var(--high-strong)', fontWeight: 600 }}>{pct(Math.abs(s.dp),1)}</span></span>
                    </li>
                  )) : <li style={{ color: 'var(--ink-4)' }}>No material risk factors.</li>}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* full table */}
      <div className="card" style={{ marginTop: 18 }}>
        <div className="card-head"><div className="card-title">All feature contributions</div><div className="spacer" />
          <span style={{ fontSize: 12, color: 'var(--ink-4)' }}>baseline {pct(result.baseProb, 1)} → estimate {pct(result.prob, 1)}</span></div>
        <table className="tbl">
          <thead><tr><th>Feature</th><th>Source</th><th className="right">Applicant value</th><th className="right">Std. dev</th><th>Direction</th><th className="right">Impact</th></tr></thead>
          <tbody>
            {sorted.map(s => {
              const up = s.dp >= 0;
              return (
                <tr key={s.key}>
                  <td style={{ fontWeight: 550, color: 'var(--ink)' }}>{s.label}</td>
                  <td><TableChip table={s.table} /></td>
                  <td className="right mono">{s.display}</td>
                  <td className="right mono" style={{ color: 'var(--ink-3)' }}>{s.z >= 0 ? '+' : ''}{s.z.toFixed(2)}σ</td>
                  <td><span style={{ color: up ? 'var(--high-strong)' : 'var(--low-strong)', fontWeight: 600, fontSize: 12 }}>{up ? '▲ raises' : '▼ lowers'}</span></td>
                  <td className="right mono" style={{ fontWeight: 600, color: up ? 'var(--high-strong)' : 'var(--low-strong)' }}>{up ? '+' : '−'}{pct(Math.abs(s.dp), 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
window.ExplanationPage = ExplanationPage;
