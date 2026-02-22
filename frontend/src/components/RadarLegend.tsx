export default function RadarLegend() {
  const stops = [
    { color: '#00ecef', label: '5' },
    { color: '#01a0f6', label: '15' },
    { color: '#00ff00', label: '25' },
    { color: '#ffff00', label: '35' },
    { color: '#ff9900', label: '45' },
    { color: '#ff0000', label: '55' },
    { color: '#cc00cc', label: '65+' },
  ];

  return (
    <div className="radar-legend">
      <div className="radar-legend__title">dBZ</div>
      <div className="radar-legend__bar">
        {stops.map((s) => (
          <div key={s.label} className="radar-legend__stop">
            <div className="radar-legend__color" style={{ backgroundColor: s.color }} />
            <span className="radar-legend__label">{s.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
