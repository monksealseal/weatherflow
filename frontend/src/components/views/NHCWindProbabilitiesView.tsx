import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchWindProbabilities } from '../../api/nhcClient';
import { ActiveCyclone, WindSpeedProbabilities } from '../../api/nhcTypes';
import './NHCWindProbabilitiesView.css';

type SortKey = 'location' | 'prob34kt' | 'prob50kt' | 'prob64kt';

export default function NHCWindProbabilitiesView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [probs, setProbs] = useState<WindSpeedProbabilities | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>('prob64kt');
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
    fetchWindProbabilities(selectedId).then(data => { setProbs(data); setLoading(false); });
  }, [selectedId]);

  const sortedLocations = probs
    ? [...probs.locations].sort((a, b) => {
        if (sortKey === 'location') return a.location.localeCompare(b.location);
        return b[sortKey] - a[sortKey];
      })
    : [];

  return (
    <div className="view-container nhc-wind-probs">
      <div className="view-header">
        <h1>Wind Speed Probabilities</h1>
        <p className="view-subtitle">
          Probability of sustained winds reaching 34, 50, and 64 knot thresholds through 120 hours
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
        <div className="loading">Loading wind probabilities...</div>
      ) : probs ? (
        <div className="probs-content">
          <div className="probs-meta">
            <div className="meta-item">
              <span className="meta-label">Storm</span>
              <span className="meta-value">{probs.stormName}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Advisory</span>
              <span className="meta-value">#{probs.advisoryNumber}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Valid Through</span>
              <span className="meta-value">{new Date(probs.validThrough).toLocaleDateString()}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Locations</span>
              <span className="meta-value">{probs.locations.length}</span>
            </div>
          </div>

          <div className="legend">
            <div className="legend-item"><div className="legend-swatch ts" /> 34 kt (Tropical Storm)</div>
            <div className="legend-item"><div className="legend-swatch sw" /> 50 kt (Strong TS)</div>
            <div className="legend-item"><div className="legend-swatch hu" /> 64 kt (Hurricane)</div>
          </div>

          <div className="controls">
            <span className="control-label">Sort by:</span>
            <select className="control-select" value={sortKey} onChange={e => setSortKey(e.target.value as SortKey)}>
              <option value="prob64kt">Hurricane Probability (64 kt)</option>
              <option value="prob50kt">50 kt Probability</option>
              <option value="prob34kt">TS Probability (34 kt)</option>
              <option value="location">Location Name</option>
            </select>
          </div>

          <div className="probs-table-wrap">
            <table className="probs-table">
              <thead>
                <tr>
                  <th>Location</th>
                  <th>34 kt (TS)</th>
                  <th>50 kt</th>
                  <th>64 kt (Hurricane)</th>
                </tr>
              </thead>
              <tbody>
                {sortedLocations.map(loc => (
                  <tr key={loc.location}>
                    <td><strong>{loc.location}</strong></td>
                    <td>
                      <div className="prob-bar">
                        <div className="prob-bar-fill ts" style={{ width: `${loc.prob34kt * 1.2}px` }} />
                        <span className="prob-pct">{loc.prob34kt}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="prob-bar">
                        <div className="prob-bar-fill sw" style={{ width: `${loc.prob50kt * 1.2}px` }} />
                        <span className="prob-pct">{loc.prob50kt}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="prob-bar">
                        <div className="prob-bar-fill hu" style={{ width: `${loc.prob64kt * 1.2}px` }} />
                        <span className="prob-pct">{loc.prob64kt}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 24 }}>
            <h3 className="section-title">Raw Text Product</h3>
            <button className="toggle-btn" onClick={() => setShowRaw(!showRaw)}>{showRaw ? 'Hide' : 'Show'}</button>
          </div>
          {showRaw && <pre className="raw-text">{probs.text}</pre>}

          <div className="info-box">
            <strong>About Wind Speed Probabilities</strong>
            These probabilities represent the chance of sustained surface wind speeds reaching
            or exceeding specified thresholds (34 kt, 50 kt, 64 kt) at each location through
            the forecast period. They account for uncertainty in the track, intensity, and
            wind structure forecasts. A 34 kt probability of 80% means there is an 80% chance
            of tropical-storm-force winds at that location.
          </div>
        </div>
      ) : (
        <div className="loading">No active storms with wind probability data.</div>
      )}
    </div>
  );
}
