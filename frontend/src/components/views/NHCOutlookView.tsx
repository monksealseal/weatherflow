import { useEffect, useState } from 'react';
import { fetchTropicalOutlook } from '../../api/nhcClient';
import { Basin, BASIN_LABELS, TropicalWeatherOutlook } from '../../api/nhcTypes';
import './NHCOutlookView.css';

const BASINS: Basin[] = ['atlantic', 'eastern_pacific', 'central_pacific'];

export default function NHCOutlookView(): JSX.Element {
  const [basin, setBasin] = useState<Basin>('atlantic');
  const [outlook, setOutlook] = useState<TropicalWeatherOutlook | null>(null);
  const [loading, setLoading] = useState(true);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchTropicalOutlook(basin).then(data => {
      if (!cancelled) { setOutlook(data); setLoading(false); }
    });
    return () => { cancelled = true; };
  }, [basin]);

  return (
    <div className="view-container nhc-outlook-view">
      <div className="view-header">
        <h1>Tropical Weather Outlook</h1>
        <p className="view-subtitle">
          NHC 2-day and 5-day tropical cyclone formation probabilities
        </p>
      </div>

      <div className="basin-tabs">
        {BASINS.map(b => (
          <button
            key={b}
            className={`basin-tab ${b === basin ? 'active' : ''}`}
            onClick={() => setBasin(b)}
          >
            {BASIN_LABELS[b]}
          </button>
        ))}
      </div>

      {outlook && (
        <div className="issued-at">
          Issued: {new Date(outlook.issuedAt).toLocaleString()} UTC
        </div>
      )}

      {loading ? (
        <div className="no-areas">Loading outlook data...</div>
      ) : outlook && outlook.areas.length === 0 ? (
        <div className="no-areas">
          No tropical cyclone formation is expected in the {BASIN_LABELS[basin]} basin during the next 7 days.
        </div>
      ) : outlook ? (
        <>
          <div className="outlook-content">
            <div className="outlook-graphic">
              <img
                src={outlook.imageUrl}
                alt={`${BASIN_LABELS[basin]} Tropical Weather Outlook`}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="graphic-caption">
                {BASIN_LABELS[basin]} 5-Day Tropical Weather Outlook.
                Areas with development potential are highlighted.
              </div>
            </div>

            <div className="outlook-areas">
              {outlook.areas.map(area => (
                <div key={area.id} className={`outlook-area-card ${area.type}`}>
                  <div className="area-header">
                    <span className="area-title">{area.title}</span>
                    <span className={`area-badge ${area.type}`}>{area.type}</span>
                  </div>
                  <div className="area-probs">
                    <div className="prob-item">
                      <span className="prob-label">48-Hour</span>
                      <span className={`prob-value ${probClass(area.probability48h)}`}>
                        {area.probability48h}%
                      </span>
                    </div>
                    <div className="prob-item">
                      <span className="prob-label">7-Day</span>
                      <span className={`prob-value ${probClass(area.probability7d)}`}>
                        {area.probability7d}%
                      </span>
                    </div>
                    <div className="prob-item">
                      <span className="prob-label">Location</span>
                      <span className="prob-value" style={{ fontSize: '0.85rem' }}>
                        {area.lat.toFixed(1)}N {Math.abs(area.lon).toFixed(1)}W
                      </span>
                    </div>
                  </div>
                  <p className="area-description">{area.description}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="outlook-text-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 className="section-title">Raw Outlook Text</h3>
              <button
                onClick={() => setShowRaw(!showRaw)}
                style={{ background: 'none', border: '1px solid #cbd5e0', borderRadius: 4, padding: '4px 12px', cursor: 'pointer', fontSize: '0.85rem' }}
              >
                {showRaw ? 'Hide' : 'Show'}
              </button>
            </div>
            {showRaw && <pre className="outlook-raw-text">{outlook.text}</pre>}
          </div>
        </>
      ) : null}
    </div>
  );
}

function probClass(pct: number): string {
  if (pct >= 60) return 'high';
  if (pct >= 30) return 'medium';
  return 'low';
}
