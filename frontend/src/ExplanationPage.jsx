/* ===================================================================
   PAGE 2 — Explanation panel: real SHAP/coefficient attributions (/explain)
   Contributions are in the model's RAW-score space (log-odds), not the
   calibrated PD — the headline probability stays the calibrated value.
   Stacking ensemble has no tractable single-tree explanation -> unavailable.
   =================================================================== */
function ExplanationPage({ applicant, result, model, go }) {
  const m = CL.getModel(model);
  const [exp, setExp] = React.useState(null);   // { available, contributions } | null

  // Fetch real attributions from the backend (debounced); fall back to the
  // local toy attributions when the API is offline (e.g. opened from file://).
  React.useEffect(() => {
    const ctrl = new AbortController();
    const payload = (typeof sanitize === 'function') ? sanitize(applicant) : applicant;
    const id = setTimeout(() => {
      fetch((window.API_BASE || '') + '/explain', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctrl.signal,
        body: JSON.stringify({ applicant: payload, model, top_n: 15 }),
      })
        .then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
        .then(d => { setExp(d); console.debug('[CreditLens] /explain', model, 'available=' + d.available); })
        .catch(e => { if (e.name !== 'AbortError') { setExp(null); console.warn('[CreditLens] /explain failed:', e.message); } });
    }, 180);
    return () => { clearTimeout(id); ctrl.abort(); };
  }, [applicant, model]);

  const verb = { low: 'is recommended for approval', med: 'requires manual review', high: 'is flagged for decline' }[result.band];

  // ---- verdict banner (shared by all states) -------------------------------
  const banner = (
    <div className="card" style={{ marginBottom: 18, background: RISK[result.band].soft, borderColor: 'transparent' }}>
      <div className="card-pad" style={{ display: 'flex', alignItems: 'center', gap: 20, padding: '20px 24px' }}>
        <div style={{ width: 52, height: 52, borderRadius: 13, background: RISK[result.band].c, display: 'grid', placeItems: 'center', color: '#fff', flex: 'none' }}>
          <I name="explain" size={26} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 650, color: RISK[result.band].s, marginBottom: 3, display: 'flex', alignItems: 'center', gap: 8 }}>Why this decision
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: 'var(--ink-3)', fontWeight: 550 }}><ModelGlyph color={m.color} size={14} />{m.name}</span></div>
          <div style={{ fontSize: 17, fontWeight: 600, color: 'var(--ink)', letterSpacing: '-0.01em', textWrap: 'pretty' }}>
            This applicant {verb} — estimated default probability of <span className="mono" style={{ color: RISK[result.band].s }}>{pct(result.prob, 1)}</span>.
          </div>
        </div>
        <button className="btn btn-ghost" onClick={() => go('applicant')}>Back to profile</button>
      </div>
    </div>
  );

  // ---- stacking: explanation unavailable -----------------------------------
  if (exp && exp.available === false) {
    return (
      <div className="page-wrap rise">
        {banner}
        <div className="card">
          <div className="card-pad" style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '28px 24px' }}>
            <I name="info" size={22} />
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)', marginBottom: 4 }}>Feature-level explanation unavailable for the stacking ensemble</div>
              <div style={{ fontSize: 13.5, color: 'var(--ink-3)', lineHeight: 1.5 }}>
                The stacking model blends several base learners through a meta-model, so it has no single
                tractable attribution. Switch to a single model (e.g. <b>LightGBM</b>) for feature-level reasons.
                The calibrated PD above is still valid.
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ---- build rows: real /explain contributions, else local toy fallback ----
  const useReal = !!(exp && exp.available && exp.contributions && exp.contributions.length);
  const rows = useReal
    ? exp.contributions.map(c => {
        const f = CL.FEATURES[c.feature] || {};
        return { key: c.feature, label: f.label || c.feature, table: f.table || 'engineered',
                 display: f.fmt ? f.fmt(c.value) : String(c.value), contrib: c.contribution };
      })
    : [...result.steps].map(s => ({ key: s.key, label: s.label, table: s.table, display: s.display, contrib: s.dp }));

  rows.sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
  const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.contrib)));
  const raisers = rows.filter(r => r.contrib > 0).slice(0, 3);
  const lowerers = rows.filter(r => r.contrib < 0).slice(0, 3);
  const unit = useReal ? 'log-odds' : 'local estimate';

  return (
    <div className="page-wrap rise">
      {banner}

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.55fr) minmax(300px, 1fr)', alignItems: 'start' }}>
        {/* contribution bars */}
        <div className="card">
          <div className="card-head">
            <div className="card-title">Feature contributions
              <div className="sub">Signed impact on the model's raw score ({unit}). Bars are relative to the largest factor.</div></div>
            <div className="spacer" />
            <div className="legend">
              <span className="li"><span className="sw" style={{ background: 'var(--high)' }} />Raises risk</span>
              <span className="li"><span className="sw" style={{ background: 'var(--low)' }} />Lowers risk</span>
            </div>
          </div>
          <div className="card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {rows.map(r => {
              const up = r.contrib >= 0;
              const w = Math.abs(r.contrib) / maxAbs * 50;  // % of half-width
              return (
                <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13 }}>
                  <span style={{ width: 150, color: 'var(--ink-2)', textAlign: 'right', flex: 'none', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.label}</span>
                  <div style={{ flex: 1, position: 'relative', height: 16, background: 'var(--surface-3)', borderRadius: 4 }}>
                    <div style={{ position: 'absolute', left: '50%', top: 0, height: '100%', width: 1, background: 'var(--border)' }} />
                    <div style={{ position: 'absolute', top: 2, height: 12, borderRadius: 3,
                      background: up ? 'var(--high)' : 'var(--low)',
                      left: up ? '50%' : `calc(50% - ${w}%)`, width: `${w}%`,
                      transition: 'all 0.5s cubic-bezier(.2,.8,.2,1)' }} />
                  </div>
                  <span className="mono" style={{ width: 56, textAlign: 'right', color: 'var(--ink-3)', flex: 'none' }}>{r.display}</span>
                </div>
              );
            })}
            <div style={{ fontSize: 11.5, color: 'var(--ink-4)', marginTop: 6, lineHeight: 1.5 }}>
              Contributions explain the model's raw score; the headline PD is the <b>calibrated</b> probability, so they need not sum to it.
            </div>
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
                  {lowerers.length ? lowerers.map(r => (
                    <li key={r.key} style={{ display: 'flex', gap: 9, color: 'var(--ink-2)' }}>
                      <span style={{ color: 'var(--low)', fontWeight: 700 }}>▼</span>
                      <span><b>{r.label}</b> of <span className="mono">{r.display}</span> lowers risk</span>
                    </li>
                  )) : <li style={{ color: 'var(--ink-4)' }}>No material protective factors.</li>}
                </ul>
              </div>
              <hr className="hr" />
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7, color: 'var(--high-strong)', fontWeight: 650, fontSize: 12.5, marginBottom: 9 }}>
                  <I name="info" size={15} /> ADDING RISK</div>
                <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {raisers.length ? raisers.map(r => (
                    <li key={r.key} style={{ display: 'flex', gap: 9, color: 'var(--ink-2)' }}>
                      <span style={{ color: 'var(--high)', fontWeight: 700 }}>▲</span>
                      <span><b>{r.label}</b> of <span className="mono">{r.display}</span> raises risk</span>
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
          <span style={{ fontSize: 12, color: 'var(--ink-4)' }}>{useReal ? 'real attributions from /explain' : 'local fallback (backend offline)'}</span></div>
        <table className="tbl">
          <thead><tr><th>Feature</th><th>Source</th><th className="right">Applicant value</th><th>Direction</th><th className="right">Contribution</th></tr></thead>
          <tbody>
            {rows.map(r => {
              const up = r.contrib >= 0;
              return (
                <tr key={r.key}>
                  <td style={{ fontWeight: 550, color: 'var(--ink)' }}>{r.label}</td>
                  <td><TableChip table={r.table} /></td>
                  <td className="right mono">{r.display}</td>
                  <td><span style={{ color: up ? 'var(--high-strong)' : 'var(--low-strong)', fontWeight: 600, fontSize: 12 }}>{up ? '▲ raises' : '▼ lowers'}</span></td>
                  <td className="right mono" style={{ fontWeight: 600, color: up ? 'var(--high-strong)' : 'var(--low-strong)' }}>{up ? '+' : '−'}{Math.abs(r.contrib).toFixed(3)}</td>
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
