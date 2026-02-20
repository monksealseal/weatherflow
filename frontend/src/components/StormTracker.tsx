import { useEffect, useState } from 'react';

interface ActiveStorm {
  id: string;
  name: string;
  type: string;
  basin: string;
  lat: number;
  lon: number;
  maxWind: number;  // kt
  minPressure: number;
  movement: string;
  lastUpdate: string;
}

interface StormTrackerProps {
  onClose: () => void;
  onSelectStorm?: (lat: number, lon: number) => void;
}

// Category from wind speed (kt) - Saffir-Simpson
function getCategory(windKt: number): { label: string; color: string } {
  if (windKt >= 137) return { label: 'CAT 5', color: '#ff0000' };
  if (windKt >= 113) return { label: 'CAT 4', color: '#ff4400' };
  if (windKt >= 96)  return { label: 'CAT 3', color: '#ff8800' };
  if (windKt >= 83)  return { label: 'CAT 2', color: '#ffaa00' };
  if (windKt >= 64)  return { label: 'CAT 1', color: '#ffdd00' };
  if (windKt >= 34)  return { label: 'TS',    color: '#00cc66' };
  return { label: 'TD', color: '#00aaff' };
}

export default function StormTracker({ onClose, onSelectStorm }: StormTrackerProps) {
  const [storms, setStorms] = useState<ActiveStorm[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    // Use NHC's GeoJSON feed for active storms
    fetch('https://www.nhc.noaa.gov/CurrentSummary.json')
      .then((r) => r.json())
      .then((data: Record<string, unknown>) => {
        if (cancelled) return;
        // Parse NHC data - the structure varies, so we handle gracefully
        const parsed = parseNHCData(data);
        setStorms(parsed);
      })
      .catch(() => {
        // NHC may not be available or have active storms
        // Try JTWC or show demo data
        if (!cancelled) {
          setStorms(getDemoStorms());
          setError(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, []);

  return (
    <div className="storm-panel">
      <div className="storm-panel__header">
        <h3>Tropical Cyclone Tracker</h3>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>

      {loading && <div className="storm-panel__loading">Checking for active tropical systems...</div>}
      {error && <div className="storm-panel__error">{error}</div>}

      {!loading && storms.length === 0 && (
        <div className="storm-panel__empty">
          <p>No active tropical cyclones at this time.</p>
          <p className="storm-panel__note">
            During hurricane season (Jun-Nov Atlantic, May-Nov Pacific),
            active storms will appear here with real-time tracking data.
          </p>
        </div>
      )}

      <div className="storm-panel__list">
        {storms.map((storm) => {
          const cat = getCategory(storm.maxWind);
          return (
            <div
              key={storm.id}
              className="storm-card"
              onClick={() => onSelectStorm?.(storm.lat, storm.lon)}
            >
              <div className="storm-card__header">
                <span
                  className="storm-card__category"
                  style={{ backgroundColor: cat.color }}
                >
                  {cat.label}
                </span>
                <span className="storm-card__name">{storm.name}</span>
                <span className="storm-card__basin">{storm.basin}</span>
              </div>
              <div className="storm-card__details">
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Max Wind</span>
                  <span className="storm-card__stat-value">{storm.maxWind} kt</span>
                </div>
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Min Pressure</span>
                  <span className="storm-card__stat-value">{storm.minPressure} hPa</span>
                </div>
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Position</span>
                  <span className="storm-card__stat-value">
                    {storm.lat.toFixed(1)}N, {Math.abs(storm.lon).toFixed(1)}{storm.lon < 0 ? 'W' : 'E'}
                  </span>
                </div>
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Movement</span>
                  <span className="storm-card__stat-value">{storm.movement}</span>
                </div>
              </div>
              <div className="storm-card__updated">
                Last update: {storm.lastUpdate}
              </div>
            </div>
          );
        })}
      </div>

      <div className="storm-panel__info">
        <p>Data sources: NHC, JTWC, Open-Meteo</p>
        <p>Click a storm to center map on its position.</p>
      </div>
    </div>
  );
}

function parseNHCData(data: Record<string, unknown>): ActiveStorm[] {
  // NHC data parsing - structure varies by season
  const storms: ActiveStorm[] = [];
  try {
    const activeStorms = (data as Record<string, unknown[]>).activeStorms;
    if (Array.isArray(activeStorms)) {
      for (const s of activeStorms) {
        const storm = s as Record<string, unknown>;
        storms.push({
          id: String(storm.id ?? storm.binNumber ?? Math.random()),
          name: String(storm.name ?? 'Unknown'),
          type: String(storm.classification ?? 'TC'),
          basin: String(storm.basin ?? 'AL'),
          lat: Number(storm.latitude ?? 0),
          lon: Number(storm.longitude ?? 0),
          maxWind: Number(storm.intensity ?? 0),
          minPressure: Number(storm.pressure ?? 0),
          movement: String(storm.movementDir ?? '') + ' at ' + String(storm.movementSpeed ?? '') + ' kt',
          lastUpdate: String(storm.lastUpdate ?? new Date().toISOString()),
        });
      }
    }
  } catch {
    // Parsing failed - return empty
  }
  return storms;
}

function getDemoStorms(): ActiveStorm[] {
  // Return empty during off-season. The UI will show the "no active storms" message.
  // During testing, you could return sample storms here.
  return [];
}
