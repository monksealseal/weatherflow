import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchStormSurge } from '../../api/nhcClient';
import { ActiveCyclone, StormSurgeData } from '../../api/nhcTypes';
import './NHCStormSurgeView.css';

export default function NHCStormSurgeView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [surge, setSurge] = useState<StormSurgeData | null>(null);
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
    fetchStormSurge(selectedId).then(data => { setSurge(data); setLoading(false); });
  }, [selectedId]);

  return (
    <div className="view-container nhc-storm-surge">
      <div className="view-header">
        <h1>Storm Surge Watch/Warning</h1>
        <p className="view-subtitle">
          Potential storm surge inundation levels and watch/warning areas
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
        <div className="loading">Loading storm surge data...</div>
      ) : surge ? (
        <div className="surge-content">
          <div className="surge-alert">
            <h3>Life-Threatening Storm Surge Possible</h3>
            <p>Peak storm surge of {surge.peakSurgeFt} feet above normally dry ground is possible in the warning area.</p>
          </div>

          <div className="ww-sections">
            {surge.warnings.length > 0 && (
              <div className="ww-box warnings">
                <h4>Storm Surge Warnings</h4>
                <ul>
                  {surge.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}
            {surge.watches.length > 0 && (
              <div className="ww-box watches">
                <h4>Storm Surge Watches</h4>
                <ul>
                  {surge.watches.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}
          </div>

          <h3 className="section-title">Potential Storm Surge Inundation</h3>
          <div className="surge-zones">
            {surge.surgeZones.map((zone, i) => (
              <div key={i} className="surge-zone-card">
                <div className="zone-area">{zone.area}</div>
                <div className="zone-surge">
                  <span className={`zone-surge-value ${surgeClass(zone.maxFt)}`}>
                    {zone.minFt}-{zone.maxFt} ft
                  </span>
                  <span className="zone-surge-label">above ground</span>
                  {zone.tidalAdjusted && <span className="zone-badge">Tide adjusted</span>}
                </div>
              </div>
            ))}
          </div>

          <div className="inundation-section">
            <h3 className="section-title">Potential Storm Surge Flooding Map</h3>
            <div className="inundation-graphic">
              <img
                src={surge.inundationMapUrl}
                alt="Potential storm surge inundation map"
                onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="inundation-caption">
                Potential storm surge inundation map showing areas that could be flooded by
                water moving inland. Values represent height above normally dry ground.
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 24 }}>
            <h3 className="section-title">Full Surge Text</h3>
            <button className="toggle-btn" onClick={() => setShowRaw(!showRaw)}>{showRaw ? 'Hide' : 'Show'}</button>
          </div>
          {showRaw && <pre className="raw-text">{surge.text}</pre>}
        </div>
      ) : (
        <div className="loading">No active storms with storm surge data.</div>
      )}
    </div>
  );
}

function surgeClass(ft: number): string {
  if (ft >= 12) return 'extreme';
  if (ft >= 8) return 'high';
  if (ft >= 4) return 'moderate';
  return 'low';
}
