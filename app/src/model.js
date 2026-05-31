/* ===================================================================
   CreditLens — risk model (Home Credit Default Risk schema)
   Mid-scope feature engineering over application + bureau +
   previous_application, scored by a selectable model registry
   (LogReg / RandomForest / XGBoost / LightGBM / CatBoost / Stacking).
   A transparent additive logit core gives every model coherent,
   human-readable SHAP-style attributions. Plain JS — attaches CL.
   =================================================================== */
(function () {
  'use strict';

  // ---- feature definitions (engineered over 3 source tables) ----------------
  // table: which Home Credit source the feature derives from.
  // beta: coefficient on the standardised value (positive => raises PD).
  const FEATURES = {
    ext_source_2:      { label: 'EXT_SOURCE_2',        table: 'application', mean: 0.514,  sd: 0.191,  beta: -1.05, fmt: v => v.toFixed(3) },
    ext_source_3:      { label: 'EXT_SOURCE_3',        table: 'application', mean: 0.510,  sd: 0.194,  beta: -0.95, fmt: v => v.toFixed(3) },
    ext_source_1:      { label: 'EXT_SOURCE_1',        table: 'application', mean: 0.502,  sd: 0.211,  beta: -0.55, fmt: v => v.toFixed(3) },
    credit_to_income:  { label: 'CREDIT / INCOME',     table: 'engineered', mean: 3.9,    sd: 2.4,    beta:  0.55, fmt: v => v.toFixed(2) + 'x' },
    annuity_to_income: { label: 'ANNUITY / INCOME',    table: 'engineered', mean: 0.18,   sd: 0.09,   beta:  0.45, fmt: v => (v * 100).toFixed(1) + '%' },
    age:               { label: 'Age (DAYS_BIRTH)',    table: 'application', mean: 43.9,   sd: 11.9,   beta: -0.45, fmt: v => Math.round(v) + ' yr' },
    emp_years:         { label: 'Employment tenure',   table: 'application', mean: 6.5,    sd: 6.4,    beta: -0.40, fmt: v => v.toFixed(1) + ' yr' },
    region_rating:     { label: 'REGION_RATING',       table: 'application', mean: 2.05,   sd: 0.51,   beta:  0.35, fmt: v => v.toFixed(0) + ' / 3' },
    cnt_children:      { label: 'CNT_CHILDREN',        table: 'application', mean: 0.42,   sd: 0.72,   beta:  0.18, fmt: v => Math.round(v) },
    bureau_dpd:        { label: 'Bureau DPD count',    table: 'bureau',      mean: 0.23,   sd: 0.69,   beta:  0.70, fmt: v => Math.round(v) },
    bureau_active:     { label: 'Active bureau credits', table: 'bureau',    mean: 1.02,   sd: 1.31,   beta:  0.40, fmt: v => Math.round(v) },
    bureau_debt:       { label: 'Bureau debt total',   table: 'bureau',      mean: 137000, sd: 240000, beta:  0.30, fmt: v => '$' + Math.round(v).toLocaleString() },
    prev_approval:     { label: 'Prev. approval rate', table: 'previous',    mean: 0.62,   sd: 0.25,   beta: -0.45, fmt: v => (v * 100).toFixed(0) + '%' },
    prev_refused:      { label: 'Prev. refusals',      table: 'previous',    mean: 1.18,   sd: 1.62,   beta:  0.42, fmt: v => Math.round(v) },
    prev_count:        { label: 'Prev. applications',  table: 'previous',    mean: 4.4,    sd: 3.6,    beta: -0.18, fmt: v => Math.round(v) },
  };
  const ORDER = Object.keys(FEATURES);
  const INTERCEPT = -3.4; // calibrated so mean population PD ≈ 8% (Home Credit base rate)
  // Global signal scale: tempers the additive logit so the Bayes-optimal AUC on
  // the synthetic DGP lands near the real Home Credit ceiling (~0.80) instead of
  // an unrealistic ~0.89. Applied to every feature contribution.
  const SIGNAL = 0.66;
  const TABLE_LABEL = { application: 'application', bureau: 'bureau', previous: 'previous_application', engineered: 'engineered' };

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

  // ---- model registry --------------------------------------------------------
  // slope/offset = post-hoc calibration applied to the additive logit (changes
  //   the point estimate per model). noise = population-level irreducible error
  //   used to spread validation scores so discrimination (AUC/KS) differs
  //   realistically between models. seed keeps everything deterministic.
  const MODELS = [
    { key: 'logreg',   name: 'Logistic Regression', short: 'LogReg',   family: 'Linear baseline', color: '#64748b', slope: 1.00, offset:  0.00, noise: 1.45, latency: 4,  seed: 11, params: 'L2 · C=0.1 · 218 feats', desc: 'Interpretable linear baseline.' },
    { key: 'rf',       name: 'Random Forest',       family: 'Bagging',         short: 'RF',       color: '#0d9488', slope: 1.05, offset: -0.03, noise: 1.05, latency: 31, seed: 23, params: '400 trees · depth 12', desc: 'Bagged trees, low variance.' },
    { key: 'xgb',      name: 'XGBoost',             family: 'Gradient boosting', short: 'XGBoost', color: '#2563eb', slope: 1.12, offset: -0.05, noise: 0.78, latency: 12, seed: 37, params: '600 rounds · lr 0.02', desc: 'Tuned gradient boosting.' },
    { key: 'lgbm',     name: 'LightGBM',            family: 'Gradient boosting', short: 'LightGBM', color: '#7c3aed', slope: 1.14, offset: -0.05, noise: 0.62, latency: 8,  seed: 51, params: '1200 leaves · lr 0.02', desc: 'Fast leaf-wise boosting.' },
    { key: 'catboost', name: 'CatBoost',            family: 'Gradient boosting', short: 'CatBoost', color: '#d97706', slope: 1.12, offset: -0.04, noise: 0.70, latency: 15, seed: 67, params: 'depth 8 · lr 0.03', desc: 'Ordered boosting on categoricals.' },
    { key: 'stack',    name: 'Stacking Ensemble',   family: 'Meta-ensemble',   short: 'Stack',    color: '#1e3a8a', slope: 1.15, offset: -0.05, noise: 0.48, latency: 48, seed: 83, params: 'LGBM+XGB+Cat → LogReg meta', desc: 'Out-of-fold blend of base learners.' },
  ];
  const MODEL_MAP = Object.fromEntries(MODELS.map(m => [m.key, m]));
  const getModel = k => MODEL_MAP[k] || MODEL_MAP.lgbm;

  // ---- derive engineered inputs from a raw applicant form object -------------
  function deriveInputs(a) {
    const income = Math.max(a.amt_income, 1);
    return {
      ext_source_1: a.ext_source_1, ext_source_2: a.ext_source_2, ext_source_3: a.ext_source_3,
      credit_to_income: a.amt_credit / income,
      annuity_to_income: a.amt_annuity / income,
      age: a.age, emp_years: a.emp_years,
      region_rating: a.region_rating, cnt_children: a.cnt_children,
      bureau_dpd: a.bureau_dpd, bureau_active: a.bureau_active, bureau_debt: a.bureau_debt,
      prev_approval: a.prev_approval, prev_refused: a.prev_refused, prev_count: a.prev_count,
    };
  }

  // raw additive logit (base model, slope 1) — used for portfolio truth
  function rawLogit(a) {
    const x = deriveInputs(a);
    let s = INTERCEPT;
    ORDER.forEach(k => { const f = FEATURES[k]; s += SIGNAL * f.beta * ((x[k] - f.mean) / f.sd); });
    return s;
  }

  // Score an applicant with a chosen model. Deterministic point estimate.
  function score(a, modelKey) {
    const model = getModel(modelKey);
    const x = deriveInputs(a);
    const contribs = ORDER.map(k => {
      const f = FEATURES[k];
      const z = (x[k] - f.mean) / f.sd;
      return { key: k, label: f.label, table: f.table, value: x[k], display: f.fmt(x[k]), z, logit: f.beta * z * model.slope * SIGNAL };
    });
    const totalLogit = INTERCEPT + model.offset + contribs.reduce((s, c) => s + c.logit, 0);
    const prob = sigmoid(totalLogit);
    const baseProb = sigmoid(INTERCEPT + model.offset);

    const sorted = [...contribs].sort((p, q) => Math.abs(q.logit) - Math.abs(p.logit));
    let running = INTERCEPT + model.offset;
    const steps = sorted.map(c => {
      const before = sigmoid(running); running += c.logit; const after = sigmoid(running);
      return { ...c, dp: after - before, from: before, to: after };
    });
    return { prob, logit: totalLogit, baseProb, band: band(prob), steps, contribs, model: model.key };
  }

  // ---- bands -----------------------------------------------------------------
  let THRESH = { low: 0.188, high: 0.191 };  // synced to models/metadata.json bands (cost_ratio 5)
  function setThresholds(low, high) { THRESH = { low, high }; }
  function getThresholds() { return THRESH; }
  function band(p) { return p < THRESH.low ? 'low' : (p < THRESH.high ? 'med' : 'high'); }
  const BAND_LABEL = { low: 'Low risk', med: 'Medium risk', high: 'High risk' };

  // ---- PRNG ------------------------------------------------------------------
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    let u = 0, v = 0; while (u === 0) u = rnd(); while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const FIRST = ['Maya','Liam','Sofia','Noah','Aisha','Ethan','Priya','Lucas','Zara','Diego','Hana','Omar','Ava','Kenji','Lena','Marco','Nadia','Theo','Yara','Sam','Elena','Jonas','Imani','Ravi','Clara','Dario','Tess','Kofi','Mina','Pablo'];
  const LAST = ['Okafor','Nguyen','Reyes','Larsen','Khan','Brooks','Patel','Romano','Haddad','Silva','Yamada','Costa','Bauer','Mensah','Petrov','Oduya','Ferreira','Kowalski','Tan','Voss','Adeyemi','Greco','Sharma','Lindqvist','Mwangi','Cruz','Dahl','Bianchi','Abebe','Russo'];
  const CONTRACT = ['Cash loans', 'Revolving loans'];
  const REGIONS = ['Region 1', 'Region 2', 'Region 3', 'Region 4'];

  // ---- synthetic portfolio (Home Credit-shaped) ------------------------------
  function makePortfolio(n, seedBase) {
    const rnd = mulberry32(seedBase || 42);
    const rows = [];
    for (let i = 0; i < n; i++) {
      const income = clamp(168000 + gauss(rnd) * 95000, 27000, 900000);
      const credit = clamp(income * clamp(3.6 + gauss(rnd) * 2.2, 0.5, 12), 45000, 2500000);
      const annuity = clamp(credit / clamp(22 + gauss(rnd) * 9, 6, 60), 3000, 180000);
      const a = {
        amt_income: income, amt_credit: credit, amt_annuity: annuity,
        ext_source_1: clamp(0.50 + gauss(rnd) * 0.21, 0.02, 0.99),
        ext_source_2: clamp(0.514 + gauss(rnd) * 0.191, 0.02, 0.99),
        ext_source_3: clamp(0.510 + gauss(rnd) * 0.194, 0.02, 0.99),
        age: clamp(43.9 + gauss(rnd) * 11.9, 21, 69),
        emp_years: clamp(6.5 + gauss(rnd) * 6.4, 0, 40),
        region_rating: clamp(Math.round(2.05 + gauss(rnd) * 0.6), 1, 3),
        cnt_children: Math.max(0, Math.round(gauss(rnd) * 0.7 + 0.4)),
        bureau_dpd: Math.max(0, Math.round(gauss(rnd) * 0.7 + 0.2)),
        bureau_active: Math.max(0, Math.round(gauss(rnd) * 1.3 + 1)),
        bureau_debt: clamp(137000 + gauss(rnd) * 230000, 0, 1500000),
        prev_approval: clamp(0.62 + gauss(rnd) * 0.25, 0, 1),
        prev_refused: Math.max(0, Math.round(gauss(rnd) * 1.6 + 1.2)),
        prev_count: Math.max(0, Math.round(gauss(rnd) * 3.6 + 4.4)),
        contract: CONTRACT[rnd() < 0.9 ? 0 : 1],
      };
      const lg = rawLogit(a);
      const trueProb = sigmoid(lg);
      const defaulted = rnd() < trueProb ? 1 : 0;
      rows.push({
        id: 'SK-' + (100002 + i),
        name: FIRST[Math.floor(rnd() * FIRST.length)] + ' ' + LAST[Math.floor(rnd() * LAST.length)],
        region: REGIONS[a.region_rating - 1] || REGIONS[0],
        ageGroup: a.age < 30 ? 'Under 30' : a.age < 45 ? '30–44' : a.age < 60 ? '45–59' : '60+',
        gender: rnd() < 0.66 ? 'F' : 'M',
        contract: a.contract, credit: Math.round(credit), income: Math.round(income),
        ext2: a.ext_source_2, trueLogit: lg, trueProb, defaulted, applicant: a,
      });
    }
    return rows;
  }

  // Re-score a portfolio with a model: calibration (slope/offset) + seeded
  // population noise so discrimination genuinely differs between models.
  function scoreRows(rows, modelKey) {
    const m = getModel(modelKey);
    const rnd = mulberry32(m.seed);
    return rows.map(r => {
      const lg = r.trueLogit * m.slope + m.offset + gauss(rnd) * m.noise;
      const prob = sigmoid(lg);
      return { ...r, prob, band: band(prob) };
    });
  }

  // ---- curves ----------------------------------------------------------------
  function rocPr(rows) {
    const data = rows.map(r => ({ s: r.prob, y: r.defaulted })).sort((a, b) => b.s - a.s);
    const P = data.reduce((s, d) => s + d.y, 0), N = data.length - P;
    let tp = 0, fp = 0, auc = 0, prevX = 0, prevY = 0;
    const roc = [{ x: 0, y: 0 }], pr = [];
    for (let i = 0; i < data.length; i++) {
      if (data[i].y === 1) tp++; else fp++;
      const tpr = tp / (P || 1), fpr = fp / (N || 1);
      roc.push({ x: fpr, y: tpr }); pr.push({ x: tpr, y: tp / (tp + fp) });
      auc += (fpr - prevX) * (tpr + prevY) / 2; prevX = fpr; prevY = tpr;
    }
    let ap = 0, lastRec = 0;
    for (const p of pr) { ap += (p.x - lastRec) * p.y; lastRec = p.x; }
    return { roc, pr, auc, ap, baseRate: P / data.length };
  }

  function ksCurve(rows, bins) {
    bins = bins || 50;
    const data = [...rows].sort((a, b) => a.prob - b.prob);
    const totalBad = data.reduce((s, d) => s + d.defaulted, 0), totalGood = data.length - totalBad;
    const pts = []; let cb = 0, cg = 0, ks = 0, ksAt = 0;
    for (let i = 0; i <= bins; i++) {
      const cut = i / bins;
      while ((cb + cg) < data.length && data[cb + cg].prob <= cut) { if (data[cb + cg].defaulted) cb++; else cg++; }
      const fb = cb / (totalBad || 1), fg = cg / (totalGood || 1), sep = Math.abs(fg - fb);
      if (sep > ks) { ks = sep; ksAt = cut; }
      pts.push({ x: cut, good: fg, bad: fb, sep });
    }
    return { pts, ks, ksAt };
  }

  function histogram(rows, bins) {
    bins = bins || 20;
    const h = Array.from({ length: bins }, (_, i) => ({ x0: i / bins, x1: (i + 1) / bins, n: 0, bad: 0 }));
    rows.forEach(r => { const idx = Math.min(bins - 1, Math.floor(r.prob * bins)); h[idx].n++; if (r.defaulted) h[idx].bad++; });
    return h;
  }

  function bandCounts(rows) { const c = { low: 0, med: 0, high: 0 }; rows.forEach(r => c[r.band]++); return c; }

  function fairness(rows, key) {
    const groups = {};
    rows.forEach(r => {
      const g = r[key];
      (groups[g] = groups[g] || { n: 0, approved: 0, defaulted: 0, sumProb: 0 });
      groups[g].n++; if (r.band !== 'high') groups[g].approved++; groups[g].defaulted += r.defaulted; groups[g].sumProb += r.prob;
    });
    const out = Object.entries(groups).map(([k, v]) => ({
      group: k, n: v.n, approvalRate: v.approved / v.n, defaultRate: v.defaulted / v.n, avgProb: v.sumProb / v.n,
    })).sort((a, b) => a.group.localeCompare(b.group));
    const rates = out.map(o => o.approvalRate);
    return { groups: out, dpd: Math.max(...rates) - Math.min(...rates) };
  }

  // leaderboard metrics for every model on a validation set
  function leaderboard(valRows) {
    return MODELS.map(m => {
      const scored = scoreRows(valRows, m.key);
      const rp = rocPr(scored), ks = ksCurve(scored);
      return { ...m, auc: rp.auc, ks: ks.ks, gini: 2 * rp.auc - 1, ap: rp.ap };
    }).sort((a, b) => b.auc - a.auc);
  }

  // ---- default applicant (a clean, low-risk Home Credit profile) -------------
  const DEFAULT_APPLICANT = {
    amt_income: 162000, amt_credit: 640000, amt_annuity: 31500,
    ext_source_1: 0.42, ext_source_2: 0.48, ext_source_3: 0.43,
    age: 36, emp_years: 3.5, region_rating: 2, cnt_children: 1,
    bureau_dpd: 0, bureau_active: 2, bureau_debt: 165000,
    prev_approval: 0.58, prev_refused: 1, prev_count: 4,
    gender: 'F',
  };

  window.CL = {
    FEATURES, ORDER, INTERCEPT, TABLE_LABEL, MODELS, getModel,
    score, band, setThresholds, getThresholds, BAND_LABEL,
    makePortfolio, scoreRows, rocPr, ksCurve, histogram, bandCounts, fairness, leaderboard,
    DEFAULT_APPLICANT, sigmoid,
  };
})();
