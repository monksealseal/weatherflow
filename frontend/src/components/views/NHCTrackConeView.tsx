import { useEffect, useState } from 'react';
import { fetchActiveCyclones, fetchTrackCone } from '../../api/nhcClient';
import { ActiveCyclone, TrackForecastCone } from '../../api/nhcTypes';
import './NHCTrackConeView.css';

export default function NHCTrackConeView(): JSX.Element {
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [cone, setCone] = useState<TrackForecastCone | null>(null);
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
    fetchTrackCone(selectedId).then(data => { setCone(data); setLoading(false); });
  }, [selectedId]);

  return (
    <div className="view-container nhc-track-cone">
      <div className="view-header">
        <h1>Track Forecast Cone &amp; Warnings</h1>
        <p className="view-subtitle">
          5-day forecast track cone graphic with coastal watches and warnings
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
        <div className="loading">Loading track forecast cone...</div>
      ) : cone ? (
        <div className="cone-content">
          <div className="cone-graphic-section">
            <div className="cone-graphic">
              <img
                src={cone.coneImageUrl}
                alt={`${cone.stormName} Track Forecast Cone`}
                onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
              <div className="cone-caption">
                {cone.stormName} 5-Day Track Forecast Cone with Watch/Warning areas.
                The cone represents the probable track of the center of the tropical cyclone.
                It is formed by enclosing the area swept out by circles along the forecast track;
                the size of each circle is set so that two-thirds of historical official forecast
                errors fall within the circle.
              </div>
            </div>
          </div>

          <div className="ww-legend">
            <div className="ww-legend-item"><div className="ww-swatch hurricane-warning" /> Hurricane Warning</div>
            <div className="ww-legend-item"><div className="ww-swatch hurricane-watch" /> Hurricane Watch</div>
            <div className="ww-legend-item"><div className="ww-swatch ts-warning" /> Tropical Storm Warning</div>
            <div className="ww-legend-item"><div className="ww-swatch ts-watch" /> Tropical Storm Watch</div>
            <div className="ww-legend-item"><div className="ww-swatch surge-warning" /> Storm Surge Warning</div>
            <div className="ww-legend-item"><div className="ww-swatch surge-watch" /> Storm Surge Watch</div>
          </div>

          <div className="ww-details">
            {cone.warnings.length > 0 && (
              <div className="ww-group warnings">
                <h4>Warnings</h4>
                <ul>
                  {cone.warnings.map((w, i) => (
                    <li key={i}><strong>{formatType(w.type)}:</strong> {w.areas}</li>
                  ))}
                </ul>
              </div>
            )}
            {cone.watches.length > 0 && (
              <div className="ww-group watches">
                <h4>Watches</h4>
                <ul>
                  {cone.watches.map((w, i) => (
                    <li key={i}><strong>{formatType(w.type)}:</strong> {w.areas}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <h3 className="section-title">Forecast Track Positions</h3>
          <div className="positions-timeline">
            {cone.forecastPositions.map(pos => (
              <div key={pos.hour} className="position-item">
                <div className={`position-dot ${windClass(pos.maxWindKt)}`} />
                <span className="position-hour">{pos.hour}H</span>
                <span className="position-info">
                  {formatCat(pos.maxWindKt)} ({pos.maxWindKt} kt)
                  <span className="position-coords"> — {pos.lat.toFixed(1)}N, {Math.abs(pos.lon).toFixed(1)}W</span>
                </span>
              </div>
            ))}
          </div>

          <div className="cone-info">
            <strong>About the Forecast Cone</strong>
            The cone represents the probable track of the center of a tropical cyclone,
            and is formed by enclosing the area swept out by a set of circles along the
            forecast track. The size of each circle is set so that two-thirds of historical
            official forecast errors over a 5-year sample fall within the circle. The cone
            does not represent the size of the storm; hazardous conditions can occur well
            outside the cone. The "X" marks the current position, and dots indicate forecast
            positions. Letters inside dots indicate forecast intensity: (D)epression, (S)torm,
            (H)urricane, or (M)ajor hurricane.
          </div>
        </div>
      ) : (
        <div className="loading">No active storms with track forecast data.</div>
      )}
    </div>
  );
}

function formatType(type: string): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
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
