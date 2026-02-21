import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchForecastAdvisory } from '../../api/nhcClient';
import { ActiveCyclone, ForecastAdvisory, WindRadii } from '../../api/nhcTypes';
import './NHCForecastAdvisoryView.css';

export default function NHCForecastAdvisoryView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [advisory, setAdvisory] = useState<ForecastAdvisory | null>(null);
  const [loading, setLoading] = useState(true);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    fetchActiveCyclones().then(data => {
      setCyclones(data);
      if (data.length > 0) setSelectedId(data[0].id);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    fetchForecastAdvisory(selectedId).then(data => { setAdvisory(data); setLoading(false); });
  }, [selectedId]);

  return (
    <div className="view-container nhc-forecast-advisory">
      <div className="view-header">
        <h1>Tropical Cyclone Forecast/Advisory</h1>
        <p className="view-subtitle">
          TCM - Technical forecast with positions, wind speeds, and wind radii through 120 hours
        </p>
      </div>

      <div className="storm-selector">
        <span className="selector-label">Storm:</span>
        {cyclones.map(c => (
          <button key={c.id} className={`selector-btn ${selectedId === c.id ? 'active' : ''}`} onClick={() => setSelectedId(c.id)}>
            {c.name}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading">Loading forecast advisory...</div>
      ) : advisory ? (
        <div className="advisory-content">
          <h3 className="section-title">Current Conditions - Advisory #{advisory.advisoryNumber}</h3>
          <div className="current-conditions">
            <ConditionCard label="Position" value={`${advisory.currentLat.toFixed(1)}N`} unit={`${Math.abs(advisory.currentLon).toFixed(1)}W`} />
            <ConditionCard label="Max Wind" value={`${advisory.maxWindKt}`} unit="kt" />
            <ConditionCard label="Gusts" value={`${advisory.gustKt}`} unit="kt" />
            <ConditionCard label="Pressure" value={`${advisory.pressureMb}`} unit="mb" />
            <ConditionCard label="Movement" value={`${advisory.movementDir}\u00B0`} unit={`at ${advisory.movementSpeedKt} kt`} />
          </div>

          <div className="radii-section">
            <h3 className="section-title">Wind Radii (nautical miles)</h3>
            <div className="radii-grid">
              {advisory.windRadii34 && <RadiiCard threshold="34 kt (TS)" radii={advisory.windRadii34} className="r34" />}
              {advisory.windRadii50 && <RadiiCard threshold="50 kt" radii={advisory.windRadii50} className="r50" />}
              {advisory.windRadii64 && <RadiiCard threshold="64 kt (Hurricane)" radii={advisory.windRadii64} className="r64" />}
            </div>
          </div>

          <h3 className="section-title">Forecast Positions</h3>
          <div className="forecast-table-wrap">
            <table className="forecast-table">
              <thead>
                <tr>
                  <th>Hour</th>
                  <th>Date/Time</th>
                  <th>Lat</th>
                  <th>Lon</th>
                  <th>Max Wind (kt)</th>
                  <th>Category</th>
                </tr>
              </thead>
              <tbody>
                {advisory.forecastPositions.map(pos => (
                  <tr key={pos.hour}>
                    <td>{pos.hour}H</td>
                    <td>{new Date(pos.dateTime).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</td>
                    <td>{pos.lat.toFixed(1)}N</td>
                    <td>{Math.abs(pos.lon).toFixed(1)}W</td>
                    <td>
                      <span className={`wind-indicator ${windClass(pos.maxWindKt)}`} />
                      {pos.maxWindKt}
                    </td>
                    <td>{formatCat(pos.maxWindKt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 24 }}>
            <h3 className="section-title">Full Advisory Text</h3>
            <button className="toggle-btn" onClick={() => setShowRaw(!showRaw)}>{showRaw ? 'Hide' : 'Show'}</button>
          </div>
          {showRaw && <pre className="raw-text">{advisory.text}</pre>}
        </div>
      ) : (
        <div className="loading">No active storms with forecast advisories.</div>
      )}
    </div>
  );
}

function ConditionCard({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="condition-card">
      <div className="condition-label">{label}</div>
      <div className="condition-value">{value}</div>
      <div className="condition-unit">{unit}</div>
    </div>
  );
}

function RadiiCard({ threshold, radii, className }: { threshold: string; radii: WindRadii; className: string }) {
  return (
    <div className={`radii-card ${className}`}>
      <h4>{threshold}</h4>
      <div className="radii-quadrants">
        <div className="quad-item"><span className="quad-label">NE:</span><span className="quad-value">{radii.ne} nm</span></div>
        <div className="quad-item"><span className="quad-label">SE:</span><span className="quad-value">{radii.se} nm</span></div>
        <div className="quad-item"><span className="quad-label">SW:</span><span className="quad-value">{radii.sw} nm</span></div>
        <div className="quad-item"><span className="quad-label">NW:</span><span className="quad-value">{radii.nw} nm</span></div>
      </div>
    </div>
  );
}

function windClass(kt: number): string {
  if (kt < 34) return 'td';
  if (kt < 64) return 'ts';
  if (kt < 83) return 'cat1';
  if (kt < 96) return 'cat2';
  if (kt < 113) return 'cat3';
  if (kt < 137) return 'cat4';
  return 'cat5';
}

function formatCat(kt: number): string {
  if (kt < 34) return 'Depression';
  if (kt < 64) return 'Tropical Storm';
  if (kt < 83) return 'Category 1';
  if (kt < 96) return 'Category 2';
  if (kt < 113) return 'Category 3';
  if (kt < 137) return 'Category 4';
  return 'Category 5';
}
