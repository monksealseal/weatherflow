import { useEffect, useState } from 'react';
import { fetchMarineForecasts } from '../../api/nhcClient';
import { MarineForecast, MarineZone } from '../../api/nhcTypes';
import './NHCMarineView.css';

const ZONES: { key: MarineZone; label: string }[] = [
  { key: 'offshore', label: 'Offshore Waters' },
  { key: 'high_seas', label: 'High Seas' },
  { key: 'coastal', label: 'Coastal Waters' },
];

export default function NHCMarineView(): JSX.Element {
  const [zone, setZone] = useState<MarineZone>('offshore');
  const [forecasts, setForecasts] = useState<MarineForecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedRaw, setExpandedRaw] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchMarineForecasts(zone).then(data => { setForecasts(data); setLoading(false); });
  }, [zone]);

  return (
    <div className="view-container nhc-marine">
      <div className="view-header">
        <h1>Marine Forecasts</h1>
        <p className="view-subtitle">
          NHC offshore waters, high seas, and coastal marine forecasts during tropical cyclone activity
        </p>
      </div>

      <div className="zone-tabs">
        {ZONES.map(z => (
          <button
            key={z.key}
            className={`zone-tab ${zone === z.key ? 'active' : ''}`}
            onClick={() => setZone(z.key)}
          >
            {z.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="no-data">Loading marine forecasts...</div>
      ) : forecasts.length === 0 ? (
        <div className="no-data">No marine forecasts available for this zone.</div>
      ) : (
        <div className="marine-content">
          {forecasts.map((fc, i) => {
            const rawKey = `${fc.zone}-${fc.region}`;
            return (
              <div key={i} className="marine-card">
                <div className="marine-card-header">
                  <span className="marine-region">{fc.region}</span>
                  <span className="marine-issued">{new Date(fc.issuedAt).toLocaleString()}</span>
                </div>
                <div className="marine-card-body">
                  <div className="marine-section">
                    <div className="marine-section-label">Synopsis</div>
                    <div className="marine-section-text">{fc.synopsis}</div>
                  </div>
                  <div className="marine-section">
                    <div className="marine-section-label">Forecast</div>
                    <div className="marine-section-text">{fc.forecast}</div>
                  </div>
                  <div className="marine-section" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      className="toggle-raw"
                      onClick={() => setExpandedRaw(expandedRaw === rawKey ? null : rawKey)}
                    >
                      {expandedRaw === rawKey ? 'Hide raw text' : 'Show raw text'}
                    </button>
                  </div>
                  {expandedRaw === rawKey && <pre className="marine-raw-text">{fc.text}</pre>}
                </div>
              </div>
            );
          })}

          <div className="marine-info">
            <strong>About NHC Marine Forecasts</strong>
            The National Hurricane Center issues marine forecasts for offshore waters,
            high seas, and coastal areas when tropical cyclones are active. These products
            provide critical information for mariners including wind speeds, sea heights,
            and hazardous conditions. Offshore waters forecasts cover areas from 60 nautical
            miles to the boundary of the EEZ. High seas forecasts cover open ocean areas.
          </div>
        </div>
      )}
    </div>
  );
}
