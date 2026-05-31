/* ===================================================================
   PAGE 1 — Applicant intake (Home Credit schema) + live risk gauge
   Fields grouped by source table: application · bureau · previous_application
   =================================================================== */
function NumField({ label, table, value, onChange, prefix, suffix, step = 1, min, mono = true }) {
  return (
    <div className="field">
      <label>{label}{table && <span style={{ marginLeft: 'auto' }}><TableChip table={table} /></span>}</label>
      <div className={`input-affix ${suffix ? 'has-r' : ''}`}>
        {prefix && <span className="affix">{prefix}</span>}
        <input className={`input ${mono ? 'mono' : ''}`} type="number" value={value} step={step} min={min}
          onChange={e => onChange(e.target.value === '' ? '' : +e.target.value)} />
        {suffix && <span className="affix affix-r">{suffix}</span>}
      </div>
    </div>
  );
}

function ExtSlider({ label, value, onChange }) {
  const band = value < 0.35 ? 'high' : value < 0.6 ? 'med' : 'low';
  return (
    <div className="field">
      <label style={{ display: 'block' }}>{label}</label>
      <span className="mono" style={{ display: 'block', fontSize: 14, fontWeight: 600, color: RISK[band].s, margin: '2px 0 6px' }}>{value.toFixed(3)}</span>
      <input className="range" type="range" min={0} max={1} step={0.001} value={value} onChange={e => onChange(+e.target.value)} />
    </div>
  );
}

function SectionHead({ table, title, note }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
      <TableChip table={table} />
      <span className="eyebrow">{title}</span>
      {note && <span style={{ fontSize: 11.5, color: 'var(--ink-4)', marginLeft: 'auto' }}>{note}</span>}
    </div>
  );
}

function ApplicantPage({ applicant, setApplicant, result, gaugeVariant, thresholds, model, go }) {
  const set = (k, v) => setApplicant({ ...applicant, [k]: v });
  const top = [...result.steps].sort((a, b) => Math.abs(b.dp) - Math.abs(a.dp)).slice(0, 4);
  const m = CL.getModel(model);

  return (
    <div className="page-wrap rise">
      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.4fr) minmax(340px, 1fr)', alignItems: 'start' }}>

        {/* ---- form ---- */}
        <div className="card">
          <div className="card-head">
            <div className="card-title">Applicant profile</div>
            <div className="spacer" />
            <span className="id mono" style={{ fontSize: 12, color: 'var(--ink-4)' }}>application_train</span>
          </div>

          <div className="card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            {/* application */}
            <div>
              <SectionHead table="application" title="Application" note="primary record" />
              <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
                <NumField label="Income (AMT_INCOME_TOTAL)" prefix="$" value={applicant.amt_income} step={5000} onChange={v => set('amt_income', v)} />
                <NumField label="Credit (AMT_CREDIT)" prefix="$" value={applicant.amt_credit} step={5000} onChange={v => set('amt_credit', v)} />
                <NumField label="Annuity (AMT_ANNUITY)" prefix="$" value={applicant.amt_annuity} step={500} onChange={v => set('amt_annuity', v)} />
                <NumField label="Children (CNT_CHILDREN)" value={applicant.cnt_children} min={0} onChange={v => set('cnt_children', v)} />
              </div>
              <div className="field" style={{ marginBottom: 18 }}>
                <label>Age <span className="hint">· from DAYS_BIRTH</span>
                  <span className="mono" style={{ marginLeft: 'auto', fontSize: 15, fontWeight: 600, color: 'var(--blue-700)' }}>{applicant.age} yr</span>
                </label>
                <input className="range" type="range" min={21} max={69} value={applicant.age} onChange={e => set('age', +e.target.value)} />
              </div>
              <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
                <NumField label="Employment tenure" suffix="yr" step={0.5} value={applicant.emp_years} min={0} onChange={v => set('emp_years', v)} />
                <div className="field">
                  <label>Region rating <span className="hint">· 1 best</span></label>
                  <div className="seg">
                    {[1, 2, 3].map(r => <button key={r} className={applicant.region_rating === r ? 'on' : ''} onClick={() => set('region_rating', r)}>{r}</button>)}
                  </div>
                </div>
              </div>
              <div style={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 12, padding: '16px 18px' }}>
                <div style={{ fontSize: 12, fontWeight: 650, color: 'var(--ink-2)', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 7 }}>
                  <I name="bolt" size={14} /> External credit scores <span className="dim" style={{ fontWeight: 450 }}>· strongest predictors</span></div>
                <div className="grid" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 16 }}>
                  <ExtSlider label="EXT_SOURCE_1" value={applicant.ext_source_1} onChange={v => set('ext_source_1', v)} />
                  <ExtSlider label="EXT_SOURCE_2" value={applicant.ext_source_2} onChange={v => set('ext_source_2', v)} />
                  <ExtSlider label="EXT_SOURCE_3" value={applicant.ext_source_3} onChange={v => set('ext_source_3', v)} />
                </div>
              </div>
            </div>

            <hr className="hr" />
            {/* bureau */}
            <div>
              <SectionHead table="bureau" title="Credit bureau" note="aggregated tradelines" />
              <div className="grid" style={{ gridTemplateColumns: '1fr 1fr 1fr', gap: 18 }}>
                <NumField label="Active credits" value={applicant.bureau_active} min={0} onChange={v => set('bureau_active', v)} />
                <NumField label="Past-due count (DPD)" value={applicant.bureau_dpd} min={0} onChange={v => set('bureau_dpd', v)} />
                <NumField label="Bureau debt total" prefix="$" step={5000} value={applicant.bureau_debt} min={0} onChange={v => set('bureau_debt', v)} />
              </div>
            </div>

            <hr className="hr" />
            {/* previous */}
            <div>
              <SectionHead table="previous" title="Previous applications" note="prior Home Credit history" />
              <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
                <NumField label="Applications count" value={applicant.prev_count} min={0} onChange={v => set('prev_count', v)} />
                <NumField label="Refusals" value={applicant.prev_refused} min={0} onChange={v => set('prev_refused', v)} />
              </div>
              <div className="field">
                <label>Approval rate
                  <span className="mono" style={{ marginLeft: 'auto', fontSize: 15, fontWeight: 600, color: 'var(--blue-700)' }}>{Math.round(applicant.prev_approval * 100)}%</span>
                </label>
                <input className="range" type="range" min={0} max={1} step={0.01} value={applicant.prev_approval} onChange={e => set('prev_approval', +e.target.value)} />
              </div>
            </div>

            <div style={{ display: 'flex', gap: 10, paddingTop: 4, alignItems: 'center' }}>
              <button className="btn btn-subtle" onClick={() => setApplicant({ ...CL.DEFAULT_APPLICANT })}>Reset</button>
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 12, color: 'var(--ink-4)', display: 'flex', alignItems: 'center', gap: 6 }}>
                <I name="bolt" size={14} /> Scoring live via {m.name}</span>
            </div>
          </div>
        </div>

        {/* ---- live result ---- */}
        <div style={{ position: 'sticky', top: 92, display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="card" style={{ borderTop: `3px solid ${RISK[result.band].c}` }}>
            <div className="card-pad" style={{ paddingBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span className="eyebrow">Risk assessment</span>
                <RiskBadge band={result.band} />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 6 }}>
                <ModelGlyph color={m.color} size={16} />
                <span style={{ fontSize: 12, color: 'var(--ink-3)', fontWeight: 550 }}>{m.name}</span>
              </div>
              <RiskGauge prob={result.prob} band={result.band} thresholds={thresholds} variant={gaugeVariant} size={250} />
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: gaugeVariant === 'linear' ? 14 : 4 }}>
                <Decision band={result.band} />
              </div>
            </div>
            <hr className="hr" />
            <div className="card-pad" style={{ paddingTop: 16 }}>
              <div className="eyebrow" style={{ marginBottom: 12 }}>Top drivers</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                {top.map(s => {
                  const up = s.dp >= 0;
                  return (
                    <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ width: 16, color: up ? 'var(--high)' : 'var(--low)', fontWeight: 700, fontSize: 14 }}>{up ? '▲' : '▼'}</span>
                      <span style={{ fontSize: 13, color: 'var(--ink-2)', flex: 1 }}>{s.label}</span>
                      <span className="mono" style={{ fontSize: 12, color: 'var(--ink-3)' }}>{s.display}</span>
                      <span className="mono" style={{ fontSize: 12, fontWeight: 600, width: 56, textAlign: 'right', color: up ? 'var(--high-strong)' : 'var(--low-strong)' }}>
                        {up ? '+' : '−'}{pct(Math.abs(s.dp), 1)}</span>
                    </div>
                  );
                })}
              </div>
              <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: 18 }} onClick={() => go('explain')}>
                Full explanation <I name="arrow" size={15} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
window.ApplicantPage = ApplicantPage;
