import { useEffect, useState } from 'react';
import { fetchActiveCyclones } from '../../api/nhcClient';
import { ActiveCyclone, Basin, BASIN_LABELS, CycloneClassification } from '../../api/nhcTypes';
import './NHCActiveStormsView.css';

interface Props {
  onNavigateToProduct?: (path: string) => void;
}

export default function NHCActiveStormsView({ onNavigateToProduct }: Props): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [basinFilter, setBasinFilter] = useState<Basin | 'all'>('all');

  useEffect(() => {
    fetchActiveCyclones().then(data => { setCyclones(data); setLoading(false); });
  }, []);

  const filtered = basinFilter === 'all' ? cyclones : cyclones.filter(c => c.basin === basinFilter);
  const selectedStorm = cyclones.find(c => c.id === selected);

  return (
    <div className="view-container nhc-active-storms">
      <div className="view-header">
        <h1>Active Tropical Cyclones</h1>
        <p className="view-subtitle">
          Currently active storms across all NHC basins
        </p>
      </div>

      <div className="filter-bar">
        <span className="filter-label">Basin:</span>
        <button className={`filter-btn ${basinFilter === 'all' ? 'active' : ''}`} onClick={() => setBasinFilter('all')}>All</button>
        {(['atlantic', 'eastern_pacific', 'central_pacific'] as Basin[]).map(b => (
          <button key={b} className={`filter-btn ${basinFilter === b ? 'active' : ''}`} onClick={() => setBasinFilter(b)}>
            {BASIN_LABELS[b]}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="no-storms">Loading active storms...</div>
      ) : filtered.length === 0 ? (
        <div className="no-storms">
          <h3>No Active Tropical Cyclones</h3>
          <p>There are currently no active tropical cyclones in the {basinFilter === 'all' ? 'monitored basins' : BASIN_LABELS[basinFilter as Basin]}.</p>
        </div>
      ) : (
        <>
          <div className="storms-grid">
            {filtered.map(storm => (
              <div
                key={storm.id}
                className={`storm-card ${selected === storm.id ? 'selected' : ''}`}
                onClick={() => setSelected(selected === storm.id ? null : storm.id)}
              >
                <div className="storm-card-header">
                  <span className="storm-name">{storm.name}</span>
                  <span className={`storm-category ${categoryClass(storm.classification, storm.maxWindKt)}`}>
                    {formatClassification(storm.classification, storm.maxWindKt)}
                  </span>
                </div>
                <div className="storm-card-body">
                  <div className="storm-stats">
                    <div className="stat-item">
                      <span className="stat-label">Max Wind</span>
                      <span className="stat-value">{storm.maxWindMph} mph ({storm.maxWindKt} kt)</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Gusts</span>
                      <span className="stat-value">{storm.gustMph} mph</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Pressure</span>
                      <span className="stat-value">{storm.pressureMb} mb</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Movement</span>
                      <span className="stat-value">{storm.movementDir} at {storm.movementSpeedMph} mph</span>
                    </div>
                  </div>
                  <div className="storm-position">
                    <span>{storm.lat.toFixed(1)}N, {Math.abs(storm.lon).toFixed(1)}W</span>
                    <span>|</span>
                    <span>{BASIN_LABELS[storm.basin]}</span>
                  </div>
                </div>
                <div className="storm-card-footer">
                  <span>ID: {storm.id}</span>
                  <span>Updated: {new Date(storm.lastUpdated).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>

          {selectedStorm && (
            <div className="storm-detail-panel">
              <div className="detail-header">
                <h2>{selectedStorm.name} - Available Products</h2>
                <button className="detail-close" onClick={() => setSelected(null)}>Close</button>
              </div>
              <div className="detail-products">
                {[
                  { label: 'Public Advisory (TCP)', path: '/nhc/public-advisory', icon: '\u{1F4E2}' },
                  { label: 'Forecast Advisory (TCM)', path: '/nhc/forecast-advisory', icon: '\u{1F4CB}' },
                  { label: 'TC Discussion (TCD)', path: '/nhc/discussion', icon: '\u{1F4AC}' },
                  { label: 'Track Forecast Cone', path: '/nhc/track-cone', icon: '\u{1F5FA}\uFE0F' },
                  { label: 'Wind Probabilities', path: '/nhc/wind-probabilities', icon: '\u{1F4CA}' },
                  { label: 'Storm Surge', path: '/nhc/storm-surge', icon: '\u{1F30A}' },
                ].map(product => (
                  <div
                    key={product.path}
                    className="product-link"
                    onClick={() => onNavigateToProduct?.(product.path)}
                  >
                    <span>{product.icon}</span>
                    <span>{product.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function categoryClass(classification: CycloneClassification, windKt: number): string {
  if (classification === 'tropical_depression' || classification === 'subtropical_depression') return 'td';
  if (classification === 'tropical_storm' || classification === 'subtropical_storm') return 'ts';
  if (windKt >= 137) return 'cat5';
  if (windKt >= 113) return 'cat4';
  if (windKt >= 96) return 'cat3';
  if (windKt >= 83) return 'cat2';
  return 'cat1';
}

function formatClassification(classification: CycloneClassification, windKt: number): string {
  switch (classification) {
    case 'tropical_depression': return 'TD';
    case 'tropical_storm': return 'TS';
    case 'subtropical_depression': return 'STD';
    case 'subtropical_storm': return 'STS';
    case 'post_tropical': return 'Post-Tropical';
    case 'potential_tropical_cyclone': return 'PTC';
    case 'remnant_low': return 'Remnant';
    case 'hurricane':
    case 'major_hurricane':
      if (windKt >= 137) return 'Cat 5';
      if (windKt >= 113) return 'Cat 4';
      if (windKt >= 96) return 'Cat 3';
      if (windKt >= 83) return 'Cat 2';
      return 'Cat 1';
    default: return classification;
  }
}
