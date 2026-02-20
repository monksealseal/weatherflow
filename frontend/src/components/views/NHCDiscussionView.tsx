import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchDiscussion } from '../../api/nhcClient';
import { ActiveCyclone, TropicalCycloneDiscussion } from '../../api/nhcTypes';
import './NHCDiscussionView.css';

export default function NHCDiscussionView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [discussion, setDiscussion] = useState<TropicalCycloneDiscussion | null>(null);
  const [loading, setLoading] = useState(true);

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
    fetchDiscussion(selectedId).then(data => { setDiscussion(data); setLoading(false); });
  }, [selectedId]);

  return (
    <div className="view-container nhc-discussion">
      <div className="view-header">
        <h1>Tropical Cyclone Discussion</h1>
        <p className="view-subtitle">
          TCD - Detailed meteorological analysis and forecast reasoning by NHC forecasters
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
        <div className="loading">Loading discussion...</div>
      ) : discussion ? (
        <div className="discussion-content">
          <div className="discussion-meta">
            <div className="meta-item">
              <span className="meta-label">Storm</span>
              <span className="meta-value">{discussion.stormName}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Advisory</span>
              <span className="meta-value">#{discussion.advisoryNumber}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Issued</span>
              <span className="meta-value">{new Date(discussion.issuedAt).toLocaleString()}</span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Forecaster</span>
              <span className="meta-value">{discussion.forecaster}</span>
            </div>
          </div>

          <pre className="discussion-text-container">{discussion.text}</pre>

          <div className="discussion-info">
            <strong>About the Tropical Cyclone Discussion</strong>
            The Tropical Cyclone Discussion (TCD) is written by NHC forecasters to explain the
            meteorological reasoning behind the official forecast. It includes analysis of satellite
            imagery, reconnaissance data, model guidance, SSTs, wind shear, and other factors.
            This is the most technically detailed product in the advisory package and is primarily
            intended for meteorologists and weather enthusiasts.
          </div>
        </div>
      ) : (
        <div className="loading">No active storms with discussions.</div>
      )}
    </div>
  );
}
