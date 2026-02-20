import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchPublicAdvisory } from '../../api/nhcClient';
import { ActiveCyclone, PublicAdvisory } from '../../api/nhcTypes';
import './NHCPublicAdvisoryView.css';

export default function NHCPublicAdvisoryView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [advisory, setAdvisory] = useState<PublicAdvisory | null>(null);
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
    fetchPublicAdvisory(selectedId).then(data => { setAdvisory(data); setLoading(false); });
  }, [selectedId]);

  return (
    <div className="view-container nhc-public-advisory">
      <div className="view-header">
        <h1>Tropical Cyclone Public Advisory</h1>
        <p className="view-subtitle">
          TCP - Primary advisory for general public with watches, warnings, and storm information
        </p>
      </div>

      <div className="storm-selector">
        <span className="selector-label">Storm:</span>
        {cyclones.map(c => (
          <button
            key={c.id}
            className={`selector-btn ${selectedId === c.id ? 'active' : ''}`}
            onClick={() => setSelectedId(c.id)}
          >
            {c.name}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading">Loading advisory...</div>
      ) : advisory ? (
        <div className="advisory-content">
          <div className="advisory-headline">
            <h2>{advisory.stormName} Advisory #{advisory.advisoryNumber}</h2>
            <p>{advisory.headline}</p>
          </div>

          <div className="advisory-meta">
            <div className="meta-item">
              <span className="meta-label">Issued</span>
              <span className="meta-value">{new Date(advisory.issuedAt).toLocaleString()}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Advisory Number</span>
              <span className="meta-value">{advisory.advisoryNumber}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Storm ID</span>
              <span className="meta-value">{advisory.stormId}</span>
            </div>
          </div>

          {advisory.warnings.length > 0 && (
            <div className="ww-section">
              <h3>Warnings in Effect</h3>
              <div className="ww-grid">
                {advisory.warnings.map((w, i) => (
                  <div key={i} className="ww-card warning">
                    <span className="ww-type">{formatWWType(w.type)}</span>
                    <span>{w.areas}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {advisory.watches.length > 0 && (
            <div className="ww-section">
              <h3>Watches in Effect</h3>
              <div className="ww-grid">
                {advisory.watches.map((w, i) => (
                  <div key={i} className="ww-card watch">
                    <span className="ww-type">{formatWWType(w.type)}</span>
                    <span>{w.areas}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="advisory-summary">{advisory.summary}</div>

          <div className="next-advisory">
            Next advisory at: {new Date(advisory.nextAdvisoryAt).toLocaleString()}
          </div>

          <div className="raw-text-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 className="section-title">Full Advisory Text</h3>
              <button className="toggle-btn" onClick={() => setShowRaw(!showRaw)}>
                {showRaw ? 'Hide' : 'Show'}
              </button>
            </div>
            {showRaw && <pre className="raw-text">{advisory.text}</pre>}
          </div>
        </div>
      ) : (
        <div className="loading">No active storms with public advisories.</div>
      )}
    </div>
  );
}

function formatWWType(type: string): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
