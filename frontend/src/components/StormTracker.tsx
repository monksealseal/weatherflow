import { useEffect, useState } from 'react';

interface Storm {
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
  season: number;
  dates: string;
  landfall?: string;
}

interface StormTrackerProps {
  onClose: () => void;
  onSelectStorm?: (lat: number, lon: number) => void;
}

type SeasonFilter = 'active' | 2025 | 2024 | 2023;

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

// ── Historical Data (NHC Best Track) ─────────────────────────────────

const HISTORICAL_STORMS: Storm[] = [
  // ══════════════════════════════════════════════════════════════════
  // 2025 Atlantic Hurricane Season
  // 13 named storms, 5 hurricanes, 4 major hurricanes
  // 3 Category 5 storms: Erin, Humberto, Melissa
  // ══════════════════════════════════════════════════════════════════
  { id: '2025-01', name: 'Andrea',   season: 2025, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 1001, lat: 28.5, lon: -88.0, dates: 'Jun 23-26',  movement: 'NW at 10 kt', lastUpdate: 'Jun 26, 2025', landfall: undefined },
  { id: '2025-02', name: 'Barry',    season: 2025, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 19.5, lon: -60.0, dates: 'Jun 30-Jul 4', movement: 'WNW at 12 kt', lastUpdate: 'Jul 4, 2025', landfall: undefined },
  { id: '2025-03', name: 'Chantal',  season: 2025, type: 'TS', basin: 'ATL', maxWind: 55,  minPressure: 995,  lat: 34.5, lon: -76.5, dates: 'Jul 5-9',    movement: 'N at 8 kt', lastUpdate: 'Jul 9, 2025', landfall: 'North Carolina' },
  { id: '2025-04', name: 'Dexter',   season: 2025, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 1000, lat: 22.0, lon: -55.0, dates: 'Aug 2-5',    movement: 'NW at 14 kt', lastUpdate: 'Aug 5, 2025', landfall: undefined },
  { id: '2025-05', name: 'Erin',     season: 2025, type: 'HU', basin: 'ATL', maxWind: 145, minPressure: 915,  lat: 25.0, lon: -55.0, dates: 'Aug 10-20',  movement: 'NW at 12 kt', lastUpdate: 'Aug 20, 2025', landfall: undefined },
  { id: '2025-06', name: 'Fernand',  season: 2025, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 20.0, lon: -96.0, dates: 'Aug 18-21',  movement: 'W at 10 kt', lastUpdate: 'Aug 21, 2025', landfall: undefined },
  { id: '2025-07', name: 'Gabrielle',season: 2025, type: 'HU', basin: 'ATL', maxWind: 120, minPressure: 935,  lat: 32.0, lon: -62.0, dates: 'Sep 17-25',  movement: 'NNE at 12 kt', lastUpdate: 'Sep 25, 2025', landfall: undefined },
  { id: '2025-08', name: 'Humberto', season: 2025, type: 'HU', basin: 'ATL', maxWind: 140, minPressure: 918,  lat: 22.5, lon: -60.0, dates: 'Sep 23-Oct 1', movement: 'NW at 10 kt', lastUpdate: 'Oct 1, 2025', landfall: undefined },
  { id: '2025-09', name: 'Imelda',   season: 2025, type: 'TS', basin: 'ATL', maxWind: 40,  minPressure: 1004, lat: 16.0, lon: -40.0, dates: 'Oct 5-8',    movement: 'WNW at 15 kt', lastUpdate: 'Oct 8, 2025', landfall: undefined },
  { id: '2025-10', name: 'Jerry',    season: 2025, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 997,  lat: 18.0, lon: -50.0, dates: 'Oct 10-14',  movement: 'NW at 12 kt', lastUpdate: 'Oct 14, 2025', landfall: undefined },
  { id: '2025-11', name: 'Karen',    season: 2025, type: 'TS', basin: 'ATL', maxWind: 55,  minPressure: 996,  lat: 14.0, lon: -58.0, dates: 'Oct 18-22',  movement: 'WNW at 14 kt', lastUpdate: 'Oct 22, 2025', landfall: undefined },
  { id: '2025-12', name: 'Lorenzo',  season: 2025, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 30.0, lon: -45.0, dates: 'Oct 22-26',  movement: 'NE at 18 kt', lastUpdate: 'Oct 26, 2025', landfall: undefined },
  { id: '2025-13', name: 'Melissa',  season: 2025, type: 'HU', basin: 'ATL', maxWind: 160, minPressure: 892,  lat: 18.0, lon: -77.5, dates: 'Oct 24-Nov 1', movement: 'NW at 10 kt', lastUpdate: 'Nov 1, 2025', landfall: 'Jamaica (Cat 5)' },

  // ══════════════════════════════════════════════════════════════════
  // 2024 Atlantic Hurricane Season
  // 18 named storms, 11 hurricanes, 5 major hurricanes
  // Beryl: earliest Cat 5; Milton: 895 mb; Helene: deadly Big Bend landfall
  // ══════════════════════════════════════════════════════════════════
  { id: '2024-01', name: 'Alberto',  season: 2024, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 992,  lat: 22.0, lon: -96.0, dates: 'Jun 19-20',  movement: 'WSW at 12 kt', lastUpdate: 'Jun 20, 2024', landfall: 'NE Mexico' },
  { id: '2024-02', name: 'Beryl',    season: 2024, type: 'HU', basin: 'ATL', maxWind: 145, minPressure: 932,  lat: 14.5, lon: -64.5, dates: 'Jun 28-Jul 9', movement: 'WNW at 20 kt', lastUpdate: 'Jul 9, 2024', landfall: 'Windward Islands, Mexico, Texas' },
  { id: '2024-03', name: 'Chris',    season: 2024, type: 'TS', basin: 'ATL', maxWind: 40,  minPressure: 1005, lat: 19.5, lon: -97.0, dates: 'Jun 30-Jul 1', movement: 'W at 8 kt', lastUpdate: 'Jul 1, 2024', landfall: 'Mexico' },
  { id: '2024-04', name: 'Debby',    season: 2024, type: 'HU', basin: 'ATL', maxWind: 70,  minPressure: 979,  lat: 29.5, lon: -83.5, dates: 'Aug 3-9',    movement: 'NNE at 5 kt', lastUpdate: 'Aug 9, 2024', landfall: 'Florida' },
  { id: '2024-05', name: 'Ernesto',  season: 2024, type: 'HU', basin: 'ATL', maxWind: 85,  minPressure: 967,  lat: 35.0, lon: -62.0, dates: 'Aug 12-20',  movement: 'NE at 16 kt', lastUpdate: 'Aug 20, 2024', landfall: 'Bermuda' },
  { id: '2024-06', name: 'Francine', season: 2024, type: 'HU', basin: 'ATL', maxWind: 90,  minPressure: 972,  lat: 29.0, lon: -90.5, dates: 'Sep 9-12',   movement: 'NE at 14 kt', lastUpdate: 'Sep 12, 2024', landfall: 'Louisiana' },
  { id: '2024-07', name: 'Gordon',   season: 2024, type: 'TS', basin: 'ATL', maxWind: 40,  minPressure: 1004, lat: 19.0, lon: -40.0, dates: 'Sep 11-16',  movement: 'W at 10 kt', lastUpdate: 'Sep 16, 2024', landfall: undefined },
  { id: '2024-08', name: 'Helene',   season: 2024, type: 'HU', basin: 'ATL', maxWind: 120, minPressure: 939,  lat: 29.8, lon: -84.3, dates: 'Sep 24-28',  movement: 'NNE at 25 kt', lastUpdate: 'Sep 28, 2024', landfall: 'Florida Big Bend (Cat 4)' },
  { id: '2024-09', name: 'Isaac',    season: 2024, type: 'HU', basin: 'ATL', maxWind: 90,  minPressure: 963,  lat: 41.0, lon: -38.0, dates: 'Sep 26-Oct 1', movement: 'NE at 20 kt', lastUpdate: 'Oct 1, 2024', landfall: undefined },
  { id: '2024-10', name: 'Joyce',    season: 2024, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 1001, lat: 22.0, lon: -48.0, dates: 'Sep 27-30',  movement: 'NW at 8 kt', lastUpdate: 'Sep 30, 2024', landfall: undefined },
  { id: '2024-11', name: 'Kirk',     season: 2024, type: 'HU', basin: 'ATL', maxWind: 130, minPressure: 928,  lat: 22.0, lon: -46.0, dates: 'Sep 29-Oct 7', movement: 'NE at 22 kt', lastUpdate: 'Oct 7, 2024', landfall: undefined },
  { id: '2024-12', name: 'Leslie',   season: 2024, type: 'HU', basin: 'ATL', maxWind: 90,  minPressure: 970,  lat: 16.0, lon: -42.0, dates: 'Oct 1-10',   movement: 'NW at 10 kt', lastUpdate: 'Oct 10, 2024', landfall: undefined },
  { id: '2024-13', name: 'Milton',   season: 2024, type: 'HU', basin: 'ATL', maxWind: 155, minPressure: 895,  lat: 21.5, lon: -90.0, dates: 'Oct 5-11',   movement: 'ENE at 16 kt', lastUpdate: 'Oct 11, 2024', landfall: 'Florida (Cat 3)' },
  { id: '2024-14', name: 'Nadine',   season: 2024, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 1002, lat: 17.5, lon: -85.0, dates: 'Oct 22-24',  movement: 'W at 12 kt', lastUpdate: 'Oct 24, 2024', landfall: 'Belize' },
  { id: '2024-15', name: 'Oscar',    season: 2024, type: 'HU', basin: 'ATL', maxWind: 75,  minPressure: 984,  lat: 20.5, lon: -74.0, dates: 'Oct 19-23',  movement: 'NW at 8 kt', lastUpdate: 'Oct 23, 2024', landfall: 'Cuba' },
  { id: '2024-16', name: 'Patty',    season: 2024, type: 'TS', basin: 'ATL', maxWind: 55,  minPressure: 982,  lat: 38.0, lon: -28.0, dates: 'Oct 23-25',  movement: 'E at 15 kt', lastUpdate: 'Oct 25, 2024', landfall: 'Azores' },
  { id: '2024-17', name: 'Rafael',   season: 2024, type: 'HU', basin: 'ATL', maxWind: 105, minPressure: 954,  lat: 22.5, lon: -84.0, dates: 'Nov 4-8',    movement: 'NW at 12 kt', lastUpdate: 'Nov 8, 2024', landfall: 'Cuba' },
  { id: '2024-18', name: 'Sara',     season: 2024, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 997,  lat: 16.0, lon: -86.0, dates: 'Nov 13-17',  movement: 'W at 5 kt', lastUpdate: 'Nov 17, 2024', landfall: 'Honduras, Belize' },

  // ══════════════════════════════════════════════════════════════════
  // 2023 Atlantic Hurricane Season
  // 20 named storms, 7 hurricanes, 3 major hurricanes
  // Lee: Cat 5 (145 kt); Idalia: Cat 4 Florida landfall; Franklin: Cat 4
  // ══════════════════════════════════════════════════════════════════
  { id: '2023-01', name: 'Arlene',   season: 2023, type: 'TS', basin: 'ATL', maxWind: 35,  minPressure: 998,  lat: 26.0, lon: -87.0, dates: 'Jun 2-3',    movement: 'S at 5 kt', lastUpdate: 'Jun 3, 2023', landfall: undefined },
  { id: '2023-02', name: 'Bret',     season: 2023, type: 'TS', basin: 'ATL', maxWind: 60,  minPressure: 996,  lat: 13.0, lon: -62.0, dates: 'Jun 19-24',  movement: 'W at 18 kt', lastUpdate: 'Jun 24, 2023', landfall: undefined },
  { id: '2023-03', name: 'Cindy',    season: 2023, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 1004, lat: 14.0, lon: -50.0, dates: 'Jun 22-26',  movement: 'NW at 12 kt', lastUpdate: 'Jun 26, 2023', landfall: undefined },
  { id: '2023-04', name: 'Don',      season: 2023, type: 'HU', basin: 'ATL', maxWind: 65,  minPressure: 986,  lat: 38.0, lon: -43.0, dates: 'Jul 14-24',  movement: 'NE at 15 kt', lastUpdate: 'Jul 24, 2023', landfall: undefined },
  { id: '2023-05', name: 'Emily',    season: 2023, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 998,  lat: 24.0, lon: -50.0, dates: 'Aug 20-22',  movement: 'NE at 20 kt', lastUpdate: 'Aug 22, 2023', landfall: undefined },
  { id: '2023-06', name: 'Franklin', season: 2023, type: 'HU', basin: 'ATL', maxWind: 130, minPressure: 926,  lat: 29.0, lon: -68.0, dates: 'Aug 20-Sep 1', movement: 'NNE at 15 kt', lastUpdate: 'Sep 1, 2023', landfall: 'Dominican Republic (TS)' },
  { id: '2023-07', name: 'Gert',     season: 2023, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 21.0, lon: -55.0, dates: 'Aug 20-Sep 4', movement: 'NW at 10 kt', lastUpdate: 'Sep 4, 2023', landfall: undefined },
  { id: '2023-08', name: 'Harold',   season: 2023, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 995,  lat: 21.0, lon: -97.5, dates: 'Aug 21-23',  movement: 'WNW at 12 kt', lastUpdate: 'Aug 23, 2023', landfall: 'S Texas' },
  { id: '2023-09', name: 'Idalia',   season: 2023, type: 'HU', basin: 'ATL', maxWind: 115, minPressure: 942,  lat: 29.8, lon: -83.5, dates: 'Aug 26-31',  movement: 'NNE at 18 kt', lastUpdate: 'Aug 31, 2023', landfall: 'Florida Big Bend (Cat 3)' },
  { id: '2023-10', name: 'Jose',     season: 2023, type: 'TS', basin: 'ATL', maxWind: 55,  minPressure: 996,  lat: 16.0, lon: -34.0, dates: 'Aug 30-Sep 1', movement: 'WNW at 15 kt', lastUpdate: 'Sep 1, 2023', landfall: undefined },
  { id: '2023-11', name: 'Katia',    season: 2023, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 24.0, lon: -32.0, dates: 'Sep 1-4',    movement: 'NW at 12 kt', lastUpdate: 'Sep 4, 2023', landfall: undefined },
  { id: '2023-12', name: 'Lee',      season: 2023, type: 'HU', basin: 'ATL', maxWind: 145, minPressure: 926,  lat: 20.0, lon: -56.0, dates: 'Sep 5-16',   movement: 'NNW at 14 kt', lastUpdate: 'Sep 16, 2023', landfall: 'Nova Scotia (Post-tropical)' },
  { id: '2023-13', name: 'Margot',   season: 2023, type: 'HU', basin: 'ATL', maxWind: 80,  minPressure: 969,  lat: 35.0, lon: -35.0, dates: 'Sep 7-16',   movement: 'NE at 10 kt', lastUpdate: 'Sep 16, 2023', landfall: undefined },
  { id: '2023-14', name: 'Nigel',    season: 2023, type: 'HU', basin: 'ATL', maxWind: 85,  minPressure: 971,  lat: 35.0, lon: -46.0, dates: 'Sep 15-22',  movement: 'NE at 15 kt', lastUpdate: 'Sep 22, 2023', landfall: undefined },
  { id: '2023-15', name: 'Ophelia',  season: 2023, type: 'TS', basin: 'ATL', maxWind: 60,  minPressure: 981,  lat: 34.0, lon: -77.0, dates: 'Sep 22-24',  movement: 'N at 10 kt', lastUpdate: 'Sep 24, 2023', landfall: 'North Carolina' },
  { id: '2023-16', name: 'Philippe', season: 2023, type: 'TS', basin: 'ATL', maxWind: 50,  minPressure: 998,  lat: 20.0, lon: -60.0, dates: 'Sep 23-Oct 5', movement: 'NW at 12 kt', lastUpdate: 'Oct 5, 2023', landfall: 'Bermuda' },
  { id: '2023-17', name: 'Rina',     season: 2023, type: 'TS', basin: 'ATL', maxWind: 45,  minPressure: 999,  lat: 24.0, lon: -50.0, dates: 'Oct 4-7',    movement: 'NE at 10 kt', lastUpdate: 'Oct 7, 2023', landfall: undefined },
  { id: '2023-18', name: 'Sean',     season: 2023, type: 'TS', basin: 'ATL', maxWind: 40,  minPressure: 1005, lat: 15.0, lon: -40.0, dates: 'Oct 10-15',  movement: 'NW at 10 kt', lastUpdate: 'Oct 15, 2023', landfall: undefined },
  { id: '2023-19', name: 'Tammy',    season: 2023, type: 'HU', basin: 'ATL', maxWind: 95,  minPressure: 965,  lat: 18.0, lon: -62.0, dates: 'Oct 18-28',  movement: 'N at 10 kt', lastUpdate: 'Oct 28, 2023', landfall: 'Barbuda (Cat 1)' },
];

// Season summaries
const SEASON_SUMMARIES: Record<number, string> = {
  2025: '13 named storms, 5 hurricanes, 4 major. Three Cat 5s (Erin, Humberto, Melissa). No US hurricane landfalls. Melissa devastated Jamaica (185 mph, 892 mb).',
  2024: '18 named storms, 11 hurricanes, 5 major. Beryl: earliest Cat 5 on record. Milton: 895 mb (2nd lowest Gulf). Helene: deadliest US hurricane since Katrina.',
  2023: '20 named storms, 7 hurricanes, 3 major (4th most active). Lee: Cat 5 (145 kt). Idalia: Cat 3 Florida landfall. Above-normal despite El Nino.',
};

export default function StormTracker({ onClose, onSelectStorm }: StormTrackerProps) {
  const [activeStorms, setActiveStorms] = useState<Storm[]>([]);
  const [loading, setLoading] = useState(true);
  const [seasonFilter, setSeasonFilter] = useState<SeasonFilter>('active');
  const [sortBy, setSortBy] = useState<'wind' | 'name'>('wind');

  // Fetch active storms on mount
  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    const nhcUrl = 'https://www.nhc.noaa.gov/CurrentSummary.json';
    const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(nhcUrl)}`;

    const tryFetch = async () => {
      try {
        const res = await fetch(proxyUrl);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            const parsed = parseNHCData(data);
            setActiveStorms(parsed);
            return;
          }
        }
      } catch { /* fall through */ }
      try {
        const res = await fetch(nhcUrl);
        if (res.ok) {
          const data = await res.json();
          if (!cancelled) {
            const parsed = parseNHCData(data);
            setActiveStorms(parsed);
            return;
          }
        }
      } catch { /* fall through */ }
      if (!cancelled) setActiveStorms([]);
    };

    tryFetch().finally(() => {
      if (!cancelled) setLoading(false);
    });

    return () => { cancelled = true; };
  }, []);

  // Get storms for current filter
  const displayStorms = seasonFilter === 'active'
    ? activeStorms
    : HISTORICAL_STORMS.filter((s) => s.season === seasonFilter);

  const sortedStorms = [...displayStorms].sort((a, b) =>
    sortBy === 'wind' ? b.maxWind - a.maxWind : a.name.localeCompare(b.name),
  );

  const seasonSummary = typeof seasonFilter === 'number' ? SEASON_SUMMARIES[seasonFilter] : null;

  return (
    <div className="storm-panel">
      <div className="storm-panel__header">
        <h3>Tropical Cyclone Tracker</h3>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>

      {/* Season filter tabs */}
      <div className="storm-panel__tabs">
        {(['active', 2025, 2024, 2023] as SeasonFilter[]).map((f) => (
          <button
            key={String(f)}
            className={`storm-tab ${seasonFilter === f ? 'storm-tab--active' : ''}`}
            onClick={() => setSeasonFilter(f)}
          >
            {f === 'active' ? 'Live' : f}
          </button>
        ))}
        <div className="storm-panel__sort">
          <button
            className={`storm-sort ${sortBy === 'wind' ? 'storm-sort--active' : ''}`}
            onClick={() => setSortBy('wind')}
            title="Sort by intensity"
          >
            Wind
          </button>
          <button
            className={`storm-sort ${sortBy === 'name' ? 'storm-sort--active' : ''}`}
            onClick={() => setSortBy('name')}
            title="Sort by name"
          >
            A-Z
          </button>
        </div>
      </div>

      {/* Season summary */}
      {seasonSummary && (
        <div className="storm-panel__summary">{seasonSummary}</div>
      )}

      {/* Active storms loading */}
      {seasonFilter === 'active' && loading && (
        <div className="storm-panel__loading">Checking for active tropical systems...</div>
      )}

      {/* No storms message */}
      {seasonFilter === 'active' && !loading && activeStorms.length === 0 && (
        <div className="storm-panel__empty">
          <p>No active tropical cyclones at this time.</p>
          <p className="storm-panel__note">
            During hurricane season (Jun-Nov Atlantic, May-Nov Pacific),
            active storms will appear here with real-time tracking data.
          </p>
          <p className="storm-panel__note">
            Browse historical seasons using the tabs above.
          </p>
        </div>
      )}

      {/* Storm list */}
      <div className="storm-panel__list">
        {sortedStorms.map((storm) => {
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
                {storm.dates && <span className="storm-card__dates">{storm.dates}</span>}
              </div>
              <div className="storm-card__details">
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Max Wind</span>
                  <span className="storm-card__stat-value">{storm.maxWind} kt</span>
                </div>
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Min Pressure</span>
                  <span className="storm-card__stat-value">{storm.minPressure} mb</span>
                </div>
                <div className="storm-card__stat">
                  <span className="storm-card__stat-label">Peak Position</span>
                  <span className="storm-card__stat-value">
                    {storm.lat.toFixed(1)}N, {Math.abs(storm.lon).toFixed(1)}{storm.lon < 0 ? 'W' : 'E'}
                  </span>
                </div>
                {storm.landfall && (
                  <div className="storm-card__stat">
                    <span className="storm-card__stat-label">Landfall</span>
                    <span className="storm-card__stat-value">{storm.landfall}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="storm-panel__info">
        <p>Data: NHC Best Track Archive</p>
        <p>Click a storm to center map on its peak position.</p>
      </div>
    </div>
  );
}

function parseNHCData(data: Record<string, unknown>): Storm[] {
  const storms: Storm[] = [];
  try {
    const activeStorms = (data as Record<string, unknown[]>).activeStorms;
    if (Array.isArray(activeStorms)) {
      for (const s of activeStorms) {
        const storm = s as Record<string, unknown>;
        storms.push({
          id: String(storm.id ?? storm.binNumber ?? Math.random()),
          name: String(storm.name ?? 'Unknown'),
          type: String(storm.classification ?? 'TC'),
          basin: String(storm.basin ?? 'ATL'),
          lat: Number(storm.latitude ?? 0),
          lon: Number(storm.longitude ?? 0),
          maxWind: Number(storm.intensity ?? 0),
          minPressure: Number(storm.pressure ?? 0),
          movement: String(storm.movementDir ?? '') + ' at ' + String(storm.movementSpeed ?? '') + ' kt',
          lastUpdate: String(storm.lastUpdate ?? new Date().toISOString()),
          season: new Date().getFullYear(),
          dates: '',
        });
      }
    }
  } catch {
    // Parsing failed
  }
  return storms;
}
