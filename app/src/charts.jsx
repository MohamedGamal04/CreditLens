/* ===================================================================
   CreditLens — SVG chart toolkit (React / Babel)
   Gauge, SHAP waterfall, histogram, KS curve, ROC/PR line curves,
   fairness bars, band distribution. Exports to window.
   =================================================================== */
const { useState, useRef, useMemo } = React;

const RISK = {
  low: { c: 'var(--low)', s: 'var(--low-strong)', soft: 'var(--low-soft)' },
  med: { c: 'var(--med)', s: 'var(--med-strong)', soft: 'var(--med-soft)' },
  high: { c: 'var(--high)', s: 'var(--high-strong)', soft: 'var(--high-soft)' }
};
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';

/* ---------- shared tooltip ---------- */
function useTip() {
  const [tip, setTip] = useState(null);
  const node = tip ?
  <div className="chart-tip" style={{ left: tip.x, top: tip.y, opacity: 1 }}
  dangerouslySetInnerHTML={{ __html: tip.html }} /> :
  null;
  return [node, setTip];
}

/* ===================================================================
   RISK GAUGE — radial arc (default) or linear bar variant
   =================================================================== */
function RiskGauge({ prob, band, thresholds, variant = 'radial', size = 260 }) {
  const t = thresholds || { low: 0.08, high: 0.25 };
  const col = RISK[band];

  if (variant === 'linear') return <LinearGauge prob={prob} band={band} thresholds={t} />;

  // 180° top semicircle. Three EQUAL traffic-light thirds (Low/Med/High):
  // the needle is mapped band-relative so it always sits in the correct
  // colour zone and the arc never reads as "all red" for a low-risk score.
  const R = size * 0.40,cx = size / 2,cy = size * 0.56,sw = size * 0.10;
  const ang = (a) => a * Math.PI / 180;
  const ptAt = (a, r = R) => [cx + r * Math.cos(ang(a)), cy + r * Math.sin(ang(a))];
  const arc = (a0, a1) => {
    const [x0, y0] = ptAt(a0),[x1, y1] = ptAt(a1);
    return `M ${x0} ${y0} A ${R} ${R} 0 0 1 ${x1} ${y1}`;
  };
  // angles: 180=left, 270=top, 360=right
  const zones = [
  { a0: 180, a1: 240, c: 'var(--low)' },
  { a0: 240, a1: 300, c: 'var(--med)' },
  { a0: 300, a1: 360, c: 'var(--high)' }];

  // band-relative needle angle
  const frac = band === 'low' ?
  prob / t.low :
  band === 'med' ?
  (prob - t.low) / (t.high - t.low) :
  Math.min(1, (prob - t.high) / (1 - t.high));
  const base = band === 'low' ? 180 : band === 'med' ? 240 : 300;
  const needleA = base + 60 * Math.min(1, Math.max(0, frac));
  const [nx, ny] = ptAt(needleA, R - sw * 0.35);
  const H = size * 0.80;

  return (
    <div style={{ position: 'relative', width: size, margin: '0 auto' }}>
      <svg width={size} height={H} viewBox={`0 0 ${size} ${H}`} style={{ strokeWidth: "1px" }}>
        <path d={arc(180, 360)} fill="none" stroke="var(--surface-3)" strokeWidth={sw} strokeLinecap="round" />
        {zones.map((z, i) =>
        <path key={i} d={arc(z.a0 + (i ? 0.8 : 0), z.a1 - (i < 2 ? 0.8 : 0))} fill="none" stroke={z.c}
        strokeWidth={sw} strokeLinecap={i === 0 || i === 2 ? 'round' : 'butt'} opacity={0.92} />
        )}
        {/* zone labels */}
        {[['Low', 210], ['Med', 270], ['High', 330]].map(([l, a], i) => {
          const [lx, ly] = ptAt(a, R + sw * 0.95);
          return <text key={i} x={lx} y={ly} fontSize={10} fill="var(--ink-4)" textAnchor="middle"
          dominantBaseline="middle" fontWeight="600" letterSpacing="0.02em">{l}</text>;
        })}
        {/* needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={col.s} strokeWidth={3.6} strokeLinecap="round"
        style={{ transition: 'all 0.7s cubic-bezier(.2,.8,.2,1)' }} />
        <circle cx={cx} cy={cy} r={sw * 0.46} fill="#fff" stroke={col.s} strokeWidth={3} />
        {/* readout below the pivot, clear of the needle */}
        <text x={cx} y={cy + size * 0.135} textAnchor="middle" fontFamily="var(--mono)"
        fontSize={size * 0.17} fontWeight="600" fill={col.s} letterSpacing="-0.03em">{pct(prob, 1)}</text>
        <text x={cx} y={cy + size * 0.205} textAnchor="middle" fontSize={10.5} fill="var(--ink-3)"
        fontWeight="600" letterSpacing="0.05em">DEFAULT PROBABILITY</text>
      </svg>
    </div>);

}

function LinearGauge({ prob, band, thresholds }) {
  const t = thresholds;
  const col = RISK[band];
  return (
    <div style={{ padding: '8px 4px 4px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 14 }}>
        <span className="mono" style={{ fontSize: 44, fontWeight: 600, color: col.s, letterSpacing: '-0.03em', lineHeight: 1 }}>{pct(prob, 1)}</span>
        <span style={{ fontSize: 12.5, color: 'var(--ink-3)', fontWeight: 600 }}>default<br />probability</span>
      </div>
      <div style={{ position: 'relative', height: 16, borderRadius: 10, overflow: 'hidden', display: 'flex' }}>
        <div style={{ width: pct(t.low), background: 'var(--low)' }} />
        <div style={{ width: pct(t.high - t.low), background: 'var(--med)' }} />
        <div style={{ flex: 1, background: 'var(--high)' }} />
      </div>
      <div style={{ position: 'relative', height: 0 }}>
        <div style={{ position: 'absolute', left: `calc(${pct(Math.min(prob, 1))} - 8px)`, top: -19,
          width: 0, height: 0, borderLeft: '8px solid transparent', borderRight: '8px solid transparent',
          borderTop: '11px solid var(--ink)', transition: 'left 0.7s cubic-bezier(.2,.8,.2,1)', filter: 'drop-shadow(0 -1px 1px rgba(0,0,0,.2))' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 9, fontSize: 11, color: 'var(--ink-4)', fontFamily: 'var(--mono)' }}>
        <span>0%</span><span>{pct(t.low, 0)}</span><span>{pct(t.high, 0)}</span><span>100%</span>
      </div>
    </div>);

}

/* ===================================================================
   SHAP WATERFALL
   =================================================================== */
function Waterfall({ result, width = 720, maxRows = 9 }) {
  const [tipNode, setTip] = useTip();
  const rowH = 40,padL = 200,padR = 96,padT = 14,padB = 38;
  const steps = result.steps.slice(0, maxRows);
  const H = padT + steps.length * rowH + padB + 56;
  const probs = [result.baseProb, ...steps.map((s) => s.to)];
  const maxP = Math.max(...probs, result.prob) * 1.08;
  const x = (p) => padL + p / maxP * (width - padL - padR);

  return (
    <div style={{ position: 'relative', overflowX: 'auto' }} className="scroll">
      <svg width={width} height={H} style={{ display: 'block' }}>
        {/* baseline marker */}
        <line x1={x(result.baseProb)} y1={padT + 4} x2={x(result.baseProb)} y2={padT + steps.length * rowH + 8}
        stroke="var(--ink-4)" strokeWidth={1.2} strokeDasharray="3 3" />
        <text x={x(result.baseProb)} y={padT - 2} fontSize={10.5} fill="var(--ink-3)" textAnchor="middle" fontFamily="var(--mono)">
          base {pct(result.baseProb, 1)}
        </text>

        {steps.map((s, i) => {
          const y = padT + 12 + i * rowH;
          const up = s.dp >= 0;
          const x0 = x(s.from),x1 = x(s.to);
          const left = Math.min(x0, x1),w = Math.max(2, Math.abs(x1 - x0));
          const c = up ? 'var(--high)' : 'var(--low)';
          return (
            <g key={s.key}
            onMouseMove={(e) => {const r = e.currentTarget.ownerSVGElement.getBoundingClientRect();
              setTip({ x: e.clientX - r.left, y: y + 6, html: `<b>${s.label}</b> = ${s.display}<br/>${up ? '▲ raises' : '▼ lowers'} risk by ${pct(Math.abs(s.dp), 2)}` });}}
            onMouseLeave={() => setTip(null)} style={{ cursor: 'default' }}>
              <text x={padL - 14} y={y + rowH / 2} textAnchor="end" dominantBaseline="middle"
              fontSize={13} fill="var(--ink-2)" fontWeight="500">{s.label}</text>
              <text x={padL - 14} y={y + rowH / 2 + 14} textAnchor="end" dominantBaseline="middle"
              fontSize={11} fill="var(--ink-4)" fontFamily="var(--mono)">{s.display}</text>
              {/* connector */}
              {i > 0 && <line x1={x(steps[i - 1].to)} y1={y - rowH + rowH / 2 + 7} x2={x(steps[i - 1].to)} y2={y + rowH / 2 - 7}
              stroke="var(--border-strong)" strokeWidth={1} strokeDasharray="2 2" />}
              <rect x={left} y={y + 5} width={w} height={rowH - 18} rx={3} fill={c} opacity={0.9} />
              <text x={up ? x1 + 8 : left - 8} y={y + rowH / 2} dominantBaseline="middle"
              textAnchor={up ? 'start' : 'end'} fontSize={11.5} fontFamily="var(--mono)" fontWeight="600" fill={up ? 'var(--high-strong)' : 'var(--low-strong)'}>
                {up ? '+' : '−'}{pct(Math.abs(s.dp), 1)}
              </text>
            </g>);

        })}

        {/* final */}
        {(() => {
          const y = padT + 12 + steps.length * rowH + 10;
          const xf = x(result.prob);
          return <g>
            <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="var(--border)" strokeWidth={1} />
            <line x1={xf} y1={padT + 4} x2={xf} y2={y + 30} stroke={RISK[result.band].s} strokeWidth={1.6} />
            <rect x={padL} y={y + 12} width={xf - padL} height={20} rx={4} fill={RISK[result.band].c} opacity={0.18} />
            <rect x={padL} y={y + 12} width={Math.max(3, xf - padL)} height={20} rx={4} fill="none" stroke={RISK[result.band].s} strokeWidth={1.4} />
            <text x={padL - 14} y={y + 23} textAnchor="end" dominantBaseline="middle" fontSize={13} fontWeight="700" fill="var(--ink)">Final estimate</text>
            <text x={xf + 9} y={y + 23} dominantBaseline="middle" fontSize={13} fontFamily="var(--mono)" fontWeight="700" fill={RISK[result.band].s}>{pct(result.prob, 1)}</text>
          </g>;
        })()}
      </svg>
      {tipNode}
    </div>);

}

/* ===================================================================
   RISK-DISTRIBUTION HISTOGRAM (stacked: count + default overlay)
   =================================================================== */
function Histogram({ data, thresholds, width = 560, height = 220 }) {
  const [tipNode, setTip] = useTip();
  const padL = 40,padB = 30,padT = 12,padR = 12;
  const maxN = Math.max(...data.map((d) => d.n), 1);
  const bw = (width - padL - padR) / data.length;
  const y = (n) => padT + (1 - n / maxN) * (height - padT - padB);
  const bandOf = (p) => p < thresholds.low ? 'low' : p < thresholds.high ? 'med' : 'high';
  return (
    <div style={{ position: 'relative' }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.5, 1].map((g, i) =>
        <g key={i}>
            <line x1={padL} y1={y(maxN * g)} x2={width - padR} y2={y(maxN * g)} stroke="var(--border)" strokeWidth={1} />
            <text x={padL - 7} y={y(maxN * g) + 3} textAnchor="end" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{Math.round(maxN * g)}</text>
          </g>
        )}
        {data.map((d, i) => {
          const bx = padL + i * bw,mid = (d.x0 + d.x1) / 2;
          const c = RISK[bandOf(mid)].c;
          return (
            <g key={i}
            onMouseMove={(e) => {const r = e.currentTarget.ownerSVGElement.getBoundingClientRect();
              setTip({ x: (bx + bw / 2) / width * r.width, y: y(d.n) - 6, html: `${pct(d.x0, 0)}–${pct(d.x1, 0)}<br/><b>${d.n}</b> loans · ${d.bad} defaults` });}}
            onMouseLeave={() => setTip(null)}>
              <rect x={bx + 1} y={y(d.n)} width={bw - 2} height={Math.max(0, height - padB - y(d.n))} fill={c} opacity={0.85} rx={2} />
              <rect x={bx + 1} y={y(d.bad)} width={bw - 2} height={Math.max(0, height - padB - y(d.bad))} fill={RISK[bandOf(mid)].s} opacity={0.55} rx={2} />
            </g>);

        })}
        {[0, 0.25, 0.5, 0.75, 1].map((g, i) =>
        <text key={i} x={padL + g * (width - padL - padR)} y={height - 10} textAnchor="middle" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{pct(g, 0)}</text>
        )}
      </svg>
      {tipNode}
    </div>);

}

/* ===================================================================
   KS CURVE
   =================================================================== */
function KSCurve({ ks, width = 480, height = 240 }) {
  const [tipNode, setTip] = useTip();
  const padL = 42,padB = 34,padT = 14,padR = 14;
  const X = (x) => padL + x * (width - padL - padR);
  const Y = (v) => padT + (1 - v) * (height - padT - padB);
  const line = (key) => ks.pts.map((p, i) => `${i ? 'L' : 'M'} ${X(p.x)} ${Y(p[key])}`).join(' ');
  const ksPt = ks.pts.reduce((a, b) => b.sep > a.sep ? b : a, ks.pts[0]);
  return (
    <div style={{ position: 'relative' }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.5, 1].map((g, i) =>
        <line key={i} x1={padL} y1={Y(g)} x2={width - padR} y2={Y(g)} stroke="var(--border)" strokeWidth={1} />
        )}
        {/* KS separation marker */}
        <line x1={X(ksPt.x)} y1={Y(ksPt.good)} x2={X(ksPt.x)} y2={Y(ksPt.bad)} stroke="var(--blue-700)" strokeWidth={2} strokeDasharray="4 3" />
        <path d={line('good')} fill="none" stroke="var(--low)" strokeWidth={2.4} />
        <path d={line('bad')} fill="none" stroke="var(--high)" strokeWidth={2.4} />
        <circle cx={X(ksPt.x)} cy={Y(ksPt.good)} r={3.5} fill="var(--low-strong)" />
        <circle cx={X(ksPt.x)} cy={Y(ksPt.bad)} r={3.5} fill="var(--high-strong)" />
        <rect x={X(ksPt.x) + 6} y={(Y(ksPt.good) + Y(ksPt.bad)) / 2 - 9} width={64} height={18} rx={4} fill="var(--blue-700)" />
        <text x={X(ksPt.x) + 38} y={(Y(ksPt.good) + Y(ksPt.bad)) / 2 + 3.5} textAnchor="middle" fontSize={10.5} fill="#fff" fontFamily="var(--mono)" fontWeight="600">KS {pct(ks.ks, 1)}</text>
        {[0, 0.5, 1].map((g, i) => <text key={i} x={padL - 7} y={Y(g) + 3} textAnchor="end" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{pct(g, 0)}</text>)}
        {[0, 0.5, 1].map((g, i) => <text key={i} x={X(g)} y={height - 10} textAnchor="middle" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{pct(g, 0)}</text>)}
      </svg>
      {tipNode}
    </div>);

}

/* ===================================================================
   GENERIC LINE CURVE (ROC / PR) with diagonal & area
   =================================================================== */
function LineCurve({ points, width = 360, height = 300, color = 'var(--blue-700)', diagonal = false, baseline = null, xlab = 'x', ylab = 'y', fillArea = true }) {
  const padL = 44,padB = 38,padT = 14,padR = 16;
  const X = (x) => padL + x * (width - padL - padR);
  const Y = (v) => padT + (1 - v) * (height - padT - padB);
  const path = points.map((p, i) => `${i ? 'L' : 'M'} ${X(p.x)} ${Y(p.y)}`).join(' ');
  const area = `${path} L ${X(points[points.length - 1].x)} ${Y(0)} L ${X(points[0].x)} ${Y(0)} Z`;
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((g, i) =>
      <line key={i} x1={padL} y1={Y(g)} x2={width - padR} y2={Y(g)} stroke="var(--border)" strokeWidth={1} opacity={0.7} />
      )}
      {diagonal && <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke="var(--ink-4)" strokeWidth={1.3} strokeDasharray="4 4" />}
      {baseline != null && <line x1={X(0)} y1={Y(baseline)} x2={X(1)} y2={Y(baseline)} stroke="var(--ink-4)" strokeWidth={1.3} strokeDasharray="4 4" />}
      {fillArea && <path d={area} fill={color} opacity={0.1} />}
      <path d={path} fill="none" stroke={color} strokeWidth={2.6} strokeLinejoin="round" />
      {[0, 0.5, 1].map((g, i) => <text key={i} x={padL - 7} y={Y(g) + 3} textAnchor="end" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{g}</text>)}
      {[0, 0.5, 1].map((g, i) => <text key={i} x={X(g)} y={height - 16} textAnchor="middle" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{g}</text>)}
      <text x={(padL + width - padR) / 2} y={height - 2} textAnchor="middle" fontSize={10.5} fill="var(--ink-3)" fontWeight="600">{xlab}</text>
      <text x={12} y={(padT + height - padB) / 2} textAnchor="middle" fontSize={10.5} fill="var(--ink-3)" fontWeight="600" transform={`rotate(-90 12 ${(padT + height - padB) / 2})`}>{ylab}</text>
    </svg>);

}

/* ===================================================================
   FAIRNESS BARS — grouped approval / default by demographic group
   =================================================================== */
function FairnessBars({ data, width = 520, height = 230 }) {
  const [tipNode, setTip] = useTip();
  const padL = 44,padB = 44,padT = 14,padR = 14;
  const groups = data.groups;
  const gw = (width - padL - padR) / groups.length;
  const Y = (v) => padT + (1 - v) * (height - padT - padB);
  const barW = Math.min(26, gw * 0.3);
  return (
    <div style={{ position: 'relative' }}>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}>
        {[0, 0.5, 1].map((g, i) =>
        <g key={i}><line x1={padL} y1={Y(g)} x2={width - padR} y2={Y(g)} stroke="var(--border)" strokeWidth={1} />
          <text x={padL - 7} y={Y(g) + 3} textAnchor="end" fontSize={10} fill="var(--ink-4)" fontFamily="var(--mono)">{pct(g, 0)}</text></g>
        )}
        {groups.map((gr, i) => {
          const cx = padL + i * gw + gw / 2;
          return (
            <g key={gr.group}>
              <rect x={cx - barW - 3} y={Y(gr.approvalRate)} width={barW} height={Y(0) - Y(gr.approvalRate)} rx={3} fill="var(--blue-600)"
              onMouseMove={(e) => {const r = e.currentTarget.ownerSVGElement.getBoundingClientRect();setTip({ x: cx / width * r.width, y: Y(gr.approvalRate) - 6, html: `<b>${gr.group}</b><br/>approval ${pct(gr.approvalRate, 1)}` });}} onMouseLeave={() => setTip(null)} />
              <rect x={cx + 3} y={Y(gr.defaultRate)} width={barW} height={Y(0) - Y(gr.defaultRate)} rx={3} fill="var(--high)"
              onMouseMove={(e) => {const r = e.currentTarget.ownerSVGElement.getBoundingClientRect();setTip({ x: cx / width * r.width, y: Y(gr.defaultRate) - 6, html: `<b>${gr.group}</b><br/>default ${pct(gr.defaultRate, 1)}` });}} onMouseLeave={() => setTip(null)} />
              <text x={cx} y={height - 26} textAnchor="middle" fontSize={11} fill="var(--ink-2)" fontWeight="600">{gr.group}</text>
              <text x={cx} y={height - 12} textAnchor="middle" fontSize={9.5} fill="var(--ink-4)" fontFamily="var(--mono)">n={gr.n}</text>
            </g>);

        })}
      </svg>
      {tipNode}
    </div>);

}

/* ===================================================================
   BAND DISTRIBUTION — horizontal stacked bar
   =================================================================== */
function BandBar({ counts, total }) {
  const segs = [['low', counts.low], ['med', counts.med], ['high', counts.high]];
  return (
    <div>
      <div style={{ display: 'flex', height: 30, borderRadius: 8, overflow: 'hidden', gap: 2 }}>
        {segs.map(([b, n]) => n > 0 &&
        <div key={b} style={{ width: `${n / total * 100}%`, background: RISK[b].c,
          display: 'grid', placeItems: 'center', minWidth: 28 }}>
            <span className="mono" style={{ color: '#fff', fontSize: 12, fontWeight: 600 }}>{n}</span>
          </div>
        )}
      </div>
      <div className="legend" style={{ marginTop: 12 }}>
        {segs.map(([b, n]) =>
        <span className="li" key={b}><span className="sw" style={{ background: RISK[b].c }} />
            {b === 'low' ? 'Low' : b === 'med' ? 'Medium' : 'High'} · <b style={{ color: 'var(--ink-2)' }}>{pct(n / total, 1)}</b></span>
        )}
      </div>
    </div>);

}

Object.assign(window, { RiskGauge, Waterfall, Histogram, KSCurve, LineCurve, FairnessBars, BandBar, RISK, pct });