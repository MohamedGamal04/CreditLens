/* ===================================================================
   CreditLens — app shell: nav, routing, shared state, tweaks
   =================================================================== */
const { useEffect: useEff } = React;

// lowCut/highCut bracket the cost-derived bands (metadata.json: low=0.188 high=0.191,
// cost_ratio 5) at integer-percent slider granularity.
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "gauge": "linear",
  "lowCut": 18,
  "highCut": 20,
  "density": "regular"
}/*EDITMODE-END*/;

const NAV = [
  { id: 'applicant', label: 'Applicant', icon: 'applicant' },
  { id: 'explain',   label: 'Explanation', icon: 'explain' },
  { id: 'portfolio', label: 'Portfolio', icon: 'portfolio' },
  { id: 'modelcard', label: 'Model card', icon: 'modelcard' },
];
const TITLES = {
  applicant: ['Applicant assessment', 'Score a single application'],
  explain:   ['Decision explanation', 'Feature-level attribution'],
  portfolio: ['Portfolio dashboard', 'Batch risk across the book'],
  modelcard: ['Model card', 'Performance · fairness · governance'],
};

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [route, setRoute] = useState(() => (location.hash.replace('#', '') || 'applicant'));
  const [applicant, setApplicant] = useState({ ...CL.DEFAULT_APPLICANT });
  const [portfolio, setPortfolio] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [model, setModel] = useState('lgbm');

  // thresholds from tweaks → into the model
  const thresholds = { low: t.lowCut / 100, high: t.highCut / 100 };
  useEff(() => { CL.setThresholds(thresholds.low, thresholds.high); }, [t.lowCut, t.highCut]);

  // validation set + per-model leaderboard (drives selector AUC + model card)
  const valSet = useMemo(() => CL.makePortfolio(2400, 7), []);
  const board = useMemo(() => CL.leaderboard(valSet), [valSet, t.lowCut, t.highCut]);
  const aucMap = useMemo(() => Object.fromEntries(board.map(m => [m.key, m.auc])), [board]);

  // Local toy scorer — instant attributions for the "Top drivers" panel, and the
  // fallback when the API isn't running (so the design still works from file://).
  const localResult = useMemo(() => CL.score(sanitize(applicant), model), [applicant, model, t.lowCut, t.highCut]);

  // Live score from the real backend (creditlens.serve.api /predict). Debounced.
  const [apiScore, setApiScore] = useState(null);
  useEff(() => {
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      fetch('/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ctrl.signal,
        body: JSON.stringify({ applicant: sanitize(applicant), model, low: thresholds.low, high: thresholds.high }),
      })
        .then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
        .then(d => { setApiScore(d); console.debug('[CreditLens] /predict', model, 'PD=' + d.probability.toFixed(4), d.band); })
        .catch(e => { if (e.name !== 'AbortError') console.warn('[CreditLens] /predict failed:', e.message); });
    }, 180);
    return () => { clearTimeout(id); ctrl.abort(); };
  }, [applicant, model, t.lowCut, t.highCut]);

  // Use the calibrated API PD/band when available; keep local attribution steps.
  const result = apiScore
    ? { ...localResult, prob: apiScore.probability, band: apiScore.band }
    : localResult;

  const go = r => { setRoute(r); location.hash = r; document.querySelector('.main').scrollTop = 0; };
  useEff(() => {
    const h = () => setRoute(location.hash.replace('#', '') || 'applicant');
    window.addEventListener('hashchange', h); return () => window.removeEventListener('hashchange', h);
  }, []);

  const inspect = app => { setApplicant({ ...CL.DEFAULT_APPLICANT, ...app }); go('explain'); };
  const loadPortfolio = () => { setBatchFile(null); setPortfolio(CL.makePortfolio(240, 42)); };
  const uploadBatch = file => setBatchFile(file);

  // Real batch scoring: POST the uploaded CSV to the backend; re-score on model/threshold change.
  useEff(() => {
    if (!batchFile) return;
    const ctrl = new AbortController();
    const fd = new FormData(); fd.append('file', batchFile);
    console.info('[CreditLens] /batch_predict uploading', batchFile.name, 'model=' + model);
    fetch(`/batch_predict?model=${model}&low=${thresholds.low}&high=${thresholds.high}`,
      { method: 'POST', body: fd, signal: ctrl.signal })
      .then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
      .then(d => {
        if (d && d.rows) {
          setPortfolio(d.rows.map(x => ({ ...x, real: true })));
          console.info('[CreditLens] /batch_predict scored', d.summary.n, 'rows', d.summary);
        }
      })
      .catch(e => { if (e.name !== 'AbortError') console.error('[CreditLens] /batch_predict failed:', e.message); });
    return () => ctrl.abort();
  }, [batchFile, model, t.lowCut, t.highCut]);

  const [title, sub] = TITLES[route];

  return (
    <div className={`app density-${t.density}`}>
      {/* sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /><path d="M11 8v3l2 2" />
            </svg>
          </div>
          <div>
            <div className="brand-name">CreditLens</div>
            <div className="brand-sub">risk engine</div>
          </div>
        </div>

        <div className="nav-section-label">Underwriting</div>
        {NAV.map(n => (
          <button key={n.id} className={`nav-item ${route === n.id ? 'active' : ''}`} onClick={() => go(n.id)}>
            <I name={n.icon} size={17} />{n.label}
            {n.id === 'portfolio' && portfolio && <span className="nav-badge">{portfolio.length}</span>}
            {n.id === 'applicant' && <span className="nav-badge" style={{ background: route === 'applicant' ? 'rgba(255,255,255,.2)' : RISK[result.band].c, color: '#fff' }}>{pct(result.prob, 0)}</span>}
          </button>
        ))}
      </aside>

      {/* main */}
      <main className="main">
        <header className="topbar">
          <div>
            <div className="crumb">CreditLens <span style={{ color: 'var(--ink-4)' }}>/</span> {sub}</div>
            <h1>{title}</h1>
          </div>
          <div className="spacer" />
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            {route === 'portfolio' && portfolio && (
              <button className="btn btn-ghost" onClick={() => setPortfolio(null)}><I name="upload" size={15} />New batch</button>
            )}
            {route === 'modelcard' && (
              <button className="btn btn-ghost"><I name="download" size={15} />Export PDF</button>
            )}
            <span className="api-pill" title={apiScore ? 'calibrated PD from backend' : 'backend offline — local fallback'}>
              <span className="live" style={{ background: apiScore ? undefined : 'var(--ink-4)' }} />
              POST /predict · {apiScore ? '200' : 'offline'}</span>
            <ModelSelector value={model} onChange={setModel} aucMap={aucMap} />
          </div>
        </header>

        {route === 'applicant' && <ApplicantPage applicant={applicant} setApplicant={setApplicant} result={result} gaugeVariant={t.gauge} thresholds={thresholds} model={model} go={go} />}
        {route === 'explain'   && <ExplanationPage applicant={applicant} result={result} thresholds={thresholds} model={model} go={go} />}
        {route === 'portfolio' && <PortfolioPage portfolio={portfolio} loadPortfolio={loadPortfolio} onUpload={uploadBatch} thresholds={thresholds} model={model} inspect={inspect} />}
        {route === 'modelcard' && <ModelCardPage thresholds={thresholds} model={model} setModel={setModel} valSet={valSet} board={board} />}
      </main>

      {/* tweaks */}
      <TweaksPanel>
        <TweakSection label="Risk gauge" />
        <TweakRadio label="Style" value={t.gauge} options={['radial', 'linear']} onChange={v => setTweak('gauge', v)} />
        <TweakSection label="Band thresholds" />
        <TweakSlider label="Low / Medium cut" value={t.lowCut} min={3} max={25} step={1} unit="%" onChange={v => setTweak('lowCut', Math.min(v, t.highCut - 2))} />
        <TweakSlider label="Medium / High cut" value={t.highCut} min={16} max={45} step={1} unit="%" onChange={v => setTweak('highCut', Math.max(v, t.lowCut + 2))} />
        <TweakSection label="Layout" />
        <TweakRadio label="Density" value={t.density} options={['compact', 'regular']} onChange={v => setTweak('density', v)} />
      </TweaksPanel>
    </div>
  );
}

// guard against empty/NaN inputs while typing
function sanitize(a) {
  const n = (v, d) => (v === '' || v == null || isNaN(v)) ? d : +v;
  const D = CL.DEFAULT_APPLICANT;
  return {
    contract: a.contract, education: a.education, gender: a.gender,
    amt_income: Math.max(1, n(a.amt_income, D.amt_income)),
    amt_credit: n(a.amt_credit, D.amt_credit),
    amt_annuity: n(a.amt_annuity, D.amt_annuity),
    ext_source_1: n(a.ext_source_1, D.ext_source_1),
    ext_source_2: n(a.ext_source_2, D.ext_source_2),
    ext_source_3: n(a.ext_source_3, D.ext_source_3),
    age: n(a.age, D.age), emp_years: n(a.emp_years, D.emp_years),
    region_rating: n(a.region_rating, 2), cnt_children: n(a.cnt_children, 0),
    bureau_dpd: n(a.bureau_dpd, 0), bureau_active: n(a.bureau_active, 1),
    bureau_debt: n(a.bureau_debt, D.bureau_debt),
    prev_approval: n(a.prev_approval, D.prev_approval),
    prev_refused: n(a.prev_refused, 0), prev_count: n(a.prev_count, D.prev_count),
  };
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
