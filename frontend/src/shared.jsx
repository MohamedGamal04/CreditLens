/* ===================================================================
   CreditLens — shared UI atoms + icons
   =================================================================== */

/* minimal stroke icons */
const Icon = {
  applicant: <path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4 20c0-3.3 3.6-6 8-6s8 2.7 8 6" />,
  explain: <g><path d="M4 19V5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v14" /><path d="M8 15l3-3 2 2 4-5" /></g>,
  portfolio: <g><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" /></g>,
  modelcard: <g><path d="M4 4h16v16H4z" /><path d="M8 9h8M8 13h8M8 17h5" /></g>,
  upload: <g><path d="M12 16V4M7 9l5-5 5 5" /><path d="M5 20h14" /></g>,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  bolt: <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />,
  check: <path d="M5 12l5 5L20 6" />,
  info: <g><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></g>,
  sliders: <g><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5" /><circle cx="16" cy="6" r="2" /><circle cx="8" cy="12" r="2" /><circle cx="13" cy="18" r="2" /></g>,
  download: <g><path d="M12 4v12M7 11l5 5 5-5" /><path d="M5 20h14" /></g>,
  shield: <g><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3Z" /><path d="M9 12l2 2 4-4" /></g>,
};
function I({ name, size = 18, stroke = 2 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">{Icon[name]}</svg>;
}

function RiskBadge({ band, size }) {
  const label = { low: 'Low', med: 'Medium', high: 'High' }[band];
  return <span className={`badge ${band}`} style={size === 'lg' ? { fontSize: 13, padding: '5px 13px 5px 11px' } : null}>
    <span className="dot" />{label} risk</span>;
}

/* recommendation chip derived from band */
function Decision({ band }) {
  const map = {
    low:  { txt: 'Auto-approve eligible', c: 'var(--low-strong)', bg: 'var(--low-soft)' },
    med:  { txt: 'Manual review', c: 'var(--med-strong)', bg: 'var(--med-soft)' },
    high: { txt: 'Decline / escalate', c: 'var(--high-strong)', bg: 'var(--high-soft)' },
  }[band];
  return <span style={{ background: map.bg, color: map.c, fontWeight: 650, fontSize: 12.5, padding: '6px 12px', borderRadius: 8 }}>{map.txt}</span>;
}

/* ---- model family glyph (tiny inline mark per model) ---- */
function ModelGlyph({ color, size = 18 }) {
  return (
    <span style={{ width: size, height: size, borderRadius: 5, background: color, display: 'grid', placeItems: 'center', flex: 'none' }}>
      <svg width={size * 0.62} height={size * 0.62} viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 18V9M10 18V5M16 18v-6M22 18h-22" />
      </svg>
    </span>
  );
}

/* ---- global model selector (topbar) ---- */
function ModelSelector({ value, onChange, aucMap }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const m = CL.getModel(value);
  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', h); return () => document.removeEventListener('mousedown', h);
  }, []);
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button className="model-trigger" onClick={() => setOpen(o => !o)}>
        <ModelGlyph color={m.color} size={20} />
        <span style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', lineHeight: 1.25 }}>
          <span style={{ fontSize: 9.5, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Active model</span>
          <span style={{ fontSize: 13.5, fontWeight: 650, color: 'var(--ink)' }}>{m.name}</span>
        </span>
        {aucMap && aucMap[m.key] != null && (
          <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--blue-700)', background: 'var(--blue-soft)', padding: '2px 8px', borderRadius: 20 }}>
            AUC {aucMap[m.key].toFixed(3)}</span>
        )}
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink-4)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}><path d="m6 9 6 6 6-6" /></svg>
      </button>
      {open && (
        <div className="model-menu">
          <div style={{ fontSize: 10.5, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 650, padding: '4px 12px 8px' }}>Select model · /predict</div>
          {CL.MODELS.map(mm => (
            <button key={mm.key} className={`model-opt ${mm.key === value ? 'on' : ''}`} onClick={() => { onChange(mm.key); setOpen(false); }}>
              <ModelGlyph color={mm.color} size={26} />
              <span style={{ flex: 1, textAlign: 'left' }}>
                <span style={{ display: 'block', fontSize: 13.5, fontWeight: 600, color: 'var(--ink)' }}>{mm.name}</span>
                <span style={{ display: 'block', fontSize: 11.5, color: 'var(--ink-4)' }}>{mm.family} · {mm.latency} ms</span>
              </span>
              {aucMap && aucMap[mm.key] != null && <span className="mono" style={{ fontSize: 12, color: 'var(--ink-3)', fontWeight: 600 }}>{aucMap[mm.key].toFixed(3)}</span>}
              {mm.key === value && <I name="check" size={15} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- source-table chip ---- */
function TableChip({ table }) {
  const map = {
    application: { c: 'var(--blue-700)', bg: 'var(--blue-soft)' },
    bureau: { c: 'var(--low-strong)', bg: 'var(--low-soft)' },
    previous: { c: 'var(--med-strong)', bg: 'var(--med-soft)' },
    engineered: { c: '#7c3aed', bg: '#f3edff' },
  }[table] || { c: 'var(--ink-3)', bg: 'var(--surface-3)' };
  return <span style={{ fontFamily: 'var(--mono)', fontSize: 10, fontWeight: 600, color: map.c, background: map.bg, padding: '1px 6px', borderRadius: 5 }}>{CL.TABLE_LABEL[table]}</span>;
}

const { useEffect } = React;
Object.assign(window, { I, Icon, RiskBadge, Decision, ModelSelector, ModelGlyph, TableChip });
