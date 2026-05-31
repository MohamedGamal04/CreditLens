/* ===================================================================
   PAGE 3 — Portfolio dashboard: batch upload → distribution, KS, table
   =================================================================== */
function StatTile({ label, value, sub, accent }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={accent ? { color: accent } : null}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function UploadState({ onLoad, onUpload }) {
  const fileRef = React.useRef(null);
  const [stage, setStage] = useState('idle'); // idle | parsing | done
  const [pctp, setPctp] = useState(0);
  const run = () => {
    setStage('parsing'); setPctp(0);
    let p = 0;
    const id = setInterval(() => {
      p += Math.random() * 18 + 6; if (p >= 100) { p = 100; clearInterval(id); setTimeout(onLoad, 350); }
      setPctp(p);
    }, 120);
  };
  return (
    <div className="page-wrap rise" style={{ maxWidth: 760, margin: '0 auto' }}>
      <div style={{ textAlign: 'center', margin: '20px 0 26px' }}>
        <h2 style={{ fontSize: 22, fontWeight: 650, letterSpacing: '-0.02em', margin: '0 0 6px' }}>Score a portfolio batch</h2>
        <p className="muted" style={{ fontSize: 14, margin: 0 }}>Upload a CSV of applications to assess default risk across the book at once.</p>
      </div>
      {stage !== 'parsing' ? (
        <div className="dropzone" onClick={() => fileRef.current && fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }}
            onChange={e => { const f = e.target.files[0]; if (f && onUpload) onUpload(f); }} />
          <div style={{ width: 56, height: 56, borderRadius: 14, background: 'var(--blue-soft)', color: 'var(--blue-700)', display: 'grid', placeItems: 'center', margin: '0 auto 16px' }}>
            <I name="upload" size={26} />
          </div>
          <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 5 }}>Click to upload applications.csv → POST /batch_predict</div>
          <div className="muted" style={{ fontSize: 12, lineHeight: 1.5, maxWidth: 560, margin: '0 auto' }}>
            Required columns: amt_income, amt_credit, amt_annuity, age, emp_years, region_rating,
            cnt_children, ext_source_1, ext_source_2, ext_source_3, bureau_active, bureau_dpd,
            bureau_debt, prev_approval, prev_refused, prev_count.
            <br />Optional: SK_ID_CURR, name, TARGET (enables AUC/KS).
          </div>
          <div style={{ marginTop: 18 }}>
            <button className="btn btn-primary" onClick={e => { e.stopPropagation(); run(); }}>
              <I name="bolt" size={15} />Load synthetic sample · 240</button>
          </div>
        </div>
      ) : (
        <div className="card card-pad" style={{ textAlign: 'center', padding: '44px' }}>
          <div className="mono" style={{ fontSize: 13, color: 'var(--ink-3)', marginBottom: 14 }}>POST /batch_predict … {Math.round(pctp)}%</div>
          <div className="bar-track" style={{ maxWidth: 420, margin: '0 auto' }}><div className="bar-fill" style={{ width: pctp + '%', background: 'var(--blue-700)' }} /></div>
          <div className="muted" style={{ fontSize: 12, marginTop: 14 }}>Joining bureau + previous_application · scoring · SHAP attributions</div>
        </div>
      )}
    </div>
  );
}

function PortfolioPage({ portfolio, loadPortfolio, onUpload, thresholds, model, inspect }) {
  // NOTE: all hooks must run unconditionally (before any early return) so the
  // hook order is stable across the no-data → data transition.
  const [sortKey, setSortKey] = useState('prob');
  const [sortDir, setSortDir] = useState(-1);
  const [filter, setFilter] = useState('all');

  // Real backend batch (rows already carry calibrated prob/band) -> use as-is.
  // Synthetic sample -> re-score locally with the active model.
  const scored = useMemo(() => {
    if (!portfolio) return null;
    return portfolio[0] && portfolio[0].real ? portfolio : CL.scoreRows(portfolio, model);
  }, [portfolio, model, thresholds.low, thresholds.high]);
  const counts = useMemo(() => scored && CL.bandCounts(scored), [scored]);
  const hist = useMemo(() => scored && CL.histogram(scored, 20), [scored]);
  const ks = useMemo(() => scored && CL.ksCurve(scored), [scored]);
  const rows = useMemo(() => {
    if (!scored) return [];
    let r = scored.filter(x => filter === 'all' || x.band === filter);
    r = [...r].sort((a, b) => {
      const va = a[sortKey], vb = b[sortKey];
      return (typeof va === 'string' ? va.localeCompare(vb) : va - vb) * sortDir;
    });
    return r.slice(0, 60);
  }, [scored, sortKey, sortDir, filter]);

  if (!portfolio) return <UploadState onLoad={loadPortfolio} onUpload={onUpload} />;

  const m = CL.getModel(model);
  const total = scored.length;
  const exposure = scored.reduce((s, r) => s + r.credit, 0);
  const expDefaultRate = scored.reduce((s, r) => s + r.prob, 0) / total;
  const expLoss = scored.reduce((s, r) => s + r.prob * r.credit * 0.55, 0); // LGD 55%
  const sortBy = k => { if (k === sortKey) setSortDir(-sortDir); else { setSortKey(k); setSortDir(-1); } };

  const fmt$ = v => '$' + (v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : Math.round(v / 1e3) + 'K');

  return (
    <div className="page-wrap rise">
      {/* stat row */}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(5, 1fr)', marginBottom: 18 }}>
        <StatTile label="Applications" value={total} sub="this batch" />
        <StatTile label="Total exposure" value={fmt$(exposure)} sub="principal at issue" />
        <StatTile label="Exp. default rate" value={pct(expDefaultRate, 1)} accent="var(--med-strong)" sub={m.name} />
        <StatTile label="Exp. loss (LGD 55%)" value={fmt$(expLoss)} accent="var(--high-strong)" sub="EL = PD × LGD × EAD" />
        <StatTile label="KS statistic" value={pct(ks.ks, 1)} accent="var(--blue-700)" sub={`@ score ${pct(ks.ksAt,0)}`} />
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1.3fr 1fr', marginBottom: 18 }}>
        <div className="card">
          <div className="card-head"><div className="card-title">Score distribution<div className="sub">Loans by predicted default probability · shaded = realised defaults</div></div></div>
          <div className="card-pad"><Histogram data={hist} thresholds={thresholds} width={580} height={210} /></div>
        </div>
        <div className="card">
          <div className="card-head"><div className="card-title">Risk band mix</div></div>
          <div className="card-pad" style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
            <BandBar counts={counts} total={total} />
            <div className="grid" style={{ gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              {[['low','Low'],['med','Medium'],['high','High']].map(([b, l]) => (
                <div key={b} style={{ padding: '12px 14px', borderRadius: 10, background: RISK[b].soft }}>
                  <div style={{ fontSize: 11, fontWeight: 650, color: RISK[b].s, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{l}</div>
                  <div className="mono" style={{ fontSize: 22, fontWeight: 600, color: RISK[b].s, marginTop: 4 }}>{counts[b]}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: '1fr 1.5fr', marginBottom: 18 }}>
        <div className="card">
          <div className="card-head"><div className="card-title">KS separation<div className="sub">Cumulative goods vs. defaults</div></div></div>
          <div className="card-pad">
            <KSCurve ks={ks} width={420} height={230} />
            <div className="legend" style={{ justifyContent: 'center', marginTop: 4 }}>
              <span className="li"><span className="sw" style={{ background: 'var(--low)' }} />Non-defaults (CDF)</span>
              <span className="li"><span className="sw" style={{ background: 'var(--high)' }} />Defaults (CDF)</span>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-head"><div className="card-title">Loan ledger</div><div className="spacer" />
            <div className="seg">
              {[['all','All'],['low','Low'],['med','Med'],['high','High']].map(([v, l]) => (
                <button key={v} className={filter === v ? 'on' : ''} onClick={() => setFilter(v)}>{l}</button>
              ))}
            </div>
          </div>
          <div className="scroll" style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table className="tbl">
              <thead><tr>
                <th onClick={() => sortBy('id')}>Applicant</th>
                <th onClick={() => sortBy('ext2')} className="right">EXT_SRC_2</th>
                <th onClick={() => sortBy('credit')} className="right">Credit</th>
                <th onClick={() => sortBy('prob')} className="right">PD</th>
                <th onClick={() => sortBy('band')}>Band</th>
                <th></th>
              </tr></thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.id} className={`row-${r.band}`}>
                    <td><div style={{ fontWeight: 550, color: 'var(--ink)' }}>{r.name}</div><div className="id">{r.id}</div></td>
                    <td className="right mono">{r.ext2.toFixed(3)}</td>
                    <td className="right mono">${(r.credit/1000).toFixed(0)}K</td>
                    <td className="right mono" style={{ fontWeight: 600, color: RISK[r.band].s }}>{pct(r.prob,1)}</td>
                    <td><RiskBadge band={r.band} /></td>
                    <td className="right"><button className="btn btn-subtle" style={{ padding: '5px 10px', fontSize: 12 }} onClick={() => inspect(r.applicant)}>Inspect</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card-pad" style={{ padding: '10px 16px', borderTop: '1px solid var(--border)', fontSize: 12, color: 'var(--ink-4)' }}>
            Showing {rows.length} of {scored.filter(x => filter === 'all' || x.band === filter).length} applications
          </div>
        </div>
      </div>
    </div>
  );
}
window.PortfolioPage = PortfolioPage;
