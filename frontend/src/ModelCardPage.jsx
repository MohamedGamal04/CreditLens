/* ===================================================================
   PAGE 4 — Model card: leaderboard (selectable) + per-model metrics,
   ROC / PR, fairness, governance. Home Credit Default Risk.
   =================================================================== */
function Metric({ label, value, hint }) {
  return (
    <div className="stat" style={{ padding: '14px 16px' }}>
      <div className="label" style={{ fontSize: 11 }}>{label}</div>
      <div className="value" style={{ fontSize: 26 }}>{value}</div>
      {hint && <div style={{ fontSize: 11, color: 'var(--ink-4)', marginTop: 3 }}>{hint}</div>}
    </div>
  );
}

function Leaderboard({ board, model, setModel }) {
  const maxAuc = Math.max(...board.map(b => b.auc));
  return (
    <table className="tbl">
      <thead><tr>
        <th>Model</th><th>Family</th><th className="right">AUC-ROC</th><th>Discrimination</th>
        <th className="right">KS</th><th className="right">Gini</th><th className="right">Latency</th><th></th>
      </tr></thead>
      <tbody>
        {board.map((b, i) => (
          <tr key={b.key} onClick={() => setModel(b.key)} style={{ cursor: 'pointer', background: b.key === model ? 'var(--blue-soft)' : null }}>
            <td>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <ModelGlyph color={b.color} size={24} />
                <div>
                  <div style={{ fontWeight: 600, color: 'var(--ink)' }}>{b.name}{i === 0 && <span style={{ marginLeft: 8, fontSize: 10.5, fontWeight: 650, color: 'var(--low-strong)', background: 'var(--low-soft)', padding: '1px 7px', borderRadius: 5 }}>BEST</span>}</div>
                  <div className="id" style={{ fontFamily: 'var(--mono)' }}>{b.params}</div>
                </div>
              </div>
            </td>
            <td className="muted" style={{ fontSize: 12.5 }}>{b.family}</td>
            <td className="right mono" style={{ fontWeight: 600, color: 'var(--ink)' }}>{b.auc.toFixed(3)}</td>
            <td style={{ width: 150 }}>
              <div className="bar-track" style={{ height: 7 }}>
                <div className="bar-fill" style={{ width: `${((b.auc - 0.5) / (maxAuc - 0.5)) * 100}%`, background: b.color }} />
              </div>
            </td>
            <td className="right mono">{pct(b.ks, 1)}</td>
            <td className="right mono">{b.gini.toFixed(3)}</td>
            <td className="right mono muted">{b.latency} ms</td>
            <td className="right">
              {b.key === model
                ? <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: 'var(--blue-700)', fontWeight: 650, fontSize: 12 }}><I name="check" size={14} />Active</span>
                : <span className="btn btn-subtle" style={{ padding: '5px 11px', fontSize: 12 }}>Select</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ModelCardPage({ thresholds, model, setModel, valSet, board }) {
  const m = CL.getModel(model);
  const entry = board.find(b => b.key === model) || board[0];
  const scored = useMemo(() => CL.scoreRows(valSet, model), [valSet, model, thresholds.low, thresholds.high]);
  const rp = useMemo(() => CL.rocPr(scored), [scored]);
  const fairRegion = useMemo(() => CL.fairness(scored, 'region'), [scored]);
  const fairAge = useMemo(() => CL.fairness(scored, 'ageGroup'), [scored]);

  const cut = thresholds.high;
  let tp = 0, fp = 0, tn = 0, fn = 0;
  scored.forEach(r => {
    const pred = r.prob >= cut ? 1 : 0;
    if (pred && r.defaulted) tp++; else if (pred && !r.defaulted) fp++;
    else if (!pred && r.defaulted) fn++; else tn++;
  });
  const acc = (tp + tn) / scored.length;
  const prec = tp / (tp + fp || 1);
  const rec = tp / (tp + fn || 1);
  const brier = scored.reduce((s, r) => s + (r.prob - r.defaulted) ** 2, 0) / scored.length;

  return (
    <div className="page-wrap rise">
      {/* header card */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-pad" style={{ display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
          <ModelGlyph color={m.color} size={50} />
          <div style={{ flex: 1, minWidth: 220 }}>
            <div style={{ fontSize: 18, fontWeight: 650, letterSpacing: '-0.01em' }}>{m.name} <span className="mono" style={{ fontSize: 12, color: 'var(--ink-4)', fontWeight: 500 }}>v3.2.1</span></div>
            <div className="muted" style={{ fontSize: 13, marginTop: 2 }}>{m.family} · {m.params} · PD on Home Credit Default Risk</div>
          </div>
          {[['Dataset','application + bureau + prev'],['Train rows','307,511'],['Features','218'],['Refresh','Quarterly']].map(([k, v]) => (
            <div key={k} style={{ paddingLeft: 18, borderLeft: '1px solid var(--border)' }}>
              <div style={{ fontSize: 11, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>{k}</div>
              <div style={{ fontSize: 13.5, fontWeight: 600, marginTop: 3 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* leaderboard */}
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="card-head">
          <div className="card-title">Model leaderboard<div className="sub">5-fold OOF validation · {valSet.length.toLocaleString()} held-out applicants · click a row to activate</div></div>
        </div>
        <Leaderboard board={board} model={model} setModel={setModel} />
      </div>

      {/* metrics for active model */}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(6, 1fr)', marginBottom: 18 }}>
        <Metric label="AUC-ROC" value={entry.auc.toFixed(3)} hint="discrimination" />
        <Metric label="KS" value={pct(entry.ks, 1)} hint="separation" />
        <Metric label="Gini" value={entry.gini.toFixed(3)} hint="2·AUC−1" />
        <Metric label="Avg precision" value={rp.ap.toFixed(3)} hint="PR area" />
        <Metric label="Brier" value={brier.toFixed(3)} hint="calibration" />
        <Metric label="Base rate" value={pct(rp.baseRate, 1)} hint="observed PD" />
      </div>

      {/* curves */}
      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr 1fr', marginBottom: 18 }}>
        <div className="card">
          <div className="card-head"><div className="card-title">ROC curve<div className="sub">AUC {rp.auc.toFixed(3)}</div></div></div>
          <div className="card-pad"><LineCurve points={rp.roc} diagonal color={m.color} xlab="False positive rate" ylab="True positive rate" height={290} /></div>
        </div>
        <div className="card">
          <div className="card-head"><div className="card-title">Precision–Recall<div className="sub">AP {rp.ap.toFixed(3)}</div></div></div>
          <div className="card-pad"><LineCurve points={rp.pr} baseline={rp.baseRate} color="var(--low)" xlab="Recall" ylab="Precision" height={290} /></div>
        </div>
        <div className="card">
          <div className="card-head"><div className="card-title">Confusion @ cut<div className="sub">decision ≥ {pct(cut,0)} PD</div></div></div>
          <div className="card-pad">
            <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[['True neg', tn, 'low'],['False pos', fp, 'med'],['False neg', fn, 'high'],['True pos', tp, 'low']].map(([l, n, b], i) => (
                <div key={i} style={{ padding: '14px', borderRadius: 10, background: RISK[b].soft, textAlign: 'center' }}>
                  <div className="mono" style={{ fontSize: 22, fontWeight: 600, color: RISK[b].s }}>{n.toLocaleString()}</div>
                  <div style={{ fontSize: 11.5, color: 'var(--ink-3)', fontWeight: 600, marginTop: 3 }}>{l}</div>
                </div>
              ))}
            </div>
            <div className="grid" style={{ gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginTop: 12 }}>
              {[['Accuracy', acc],['Precision', prec],['Recall', rec]].map(([l, v]) => (
                <div key={l} style={{ textAlign: 'center' }}>
                  <div className="mono" style={{ fontSize: 15, fontWeight: 600 }}>{pct(v, 1)}</div>
                  <div style={{ fontSize: 10.5, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* fairness */}
      <div className="card">
        <div className="card-head">
          <div style={{ width: 34, height: 34, borderRadius: 9, background: 'var(--blue-soft)', color: 'var(--blue-700)', display: 'grid', placeItems: 'center' }}><I name="shield" size={18} /></div>
          <div className="card-title">Fairness check<div className="sub">Approval & default rates across groups · demographic parity · {m.name}</div></div>
        </div>
        <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', padding: 24, gap: 28 }}>
          {[['By region', fairRegion], ['By age group', fairAge]].map(([title, data]) => {
            const ok = data.dpd < 0.1;
            return (
              <div key={title}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                  <span style={{ fontWeight: 650, fontSize: 13.5 }}>{title}</span>
                  <div style={{ flex: 1 }} />
                  <span style={{ fontSize: 12, fontWeight: 650, padding: '4px 10px', borderRadius: 7,
                    background: ok ? 'var(--low-soft)' : 'var(--med-soft)', color: ok ? 'var(--low-strong)' : 'var(--med-strong)' }}>
                    {ok ? '✓ within tolerance' : '⚠ review'} · DPD {pct(data.dpd, 1)}
                  </span>
                </div>
                <FairnessBars data={data} width={500} height={220} />
              </div>
            );
          })}
        </div>
        <div className="card-pad" style={{ borderTop: '1px solid var(--border)', display: 'flex', gap: 20, fontSize: 12.5, color: 'var(--ink-3)', alignItems: 'center' }}>
          <span className="legend"><span className="li"><span className="sw" style={{ background: 'var(--blue-600)' }} />Approval rate</span>
          <span className="li"><span className="sw" style={{ background: 'var(--high)' }} />Default rate</span></span>
          <div style={{ flex: 1 }} />
          <span><b>DPD</b> = demographic-parity difference (max−min approval). Flagged when &gt; 10%.</span>
        </div>
      </div>

      {/* limitations */}
      <div className="card" style={{ marginTop: 18 }}>
        <div className="card-head"><div className="card-title">Intended use & limitations</div></div>
        <div className="card-pad grid" style={{ gridTemplateColumns: '1fr 1fr', gap: 26, fontSize: 13, lineHeight: 1.55, color: 'var(--ink-2)' }}>
          <div>
            <div className="eyebrow" style={{ marginBottom: 8, color: 'var(--low-strong)' }}>Intended use</div>
            Decision support for loan officers on Home Credit cash & revolving consumer loans. Scores are advisory; a human reviewer retains final authority on all medium- and high-band decisions.
          </div>
          <div>
            <div className="eyebrow" style={{ marginBottom: 8, color: 'var(--high-strong)' }}>Known limitations</div>
            EXT_SOURCE_* fields are missing for ~⅓ of applicants and imputed at serve time. Bureau / previous_application aggregates assume timely upstream syncs. Not validated outside the originating market.
          </div>
        </div>
      </div>
    </div>
  );
}
window.ModelCardPage = ModelCardPage;
