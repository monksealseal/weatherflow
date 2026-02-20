import { useState, useRef, useEffect } from 'react';
import type { LatLng, GeoSearchResult } from '../types/weather';
import { searchLocations } from '../api/openMeteo';

interface LocationSearchProps {
  onSelect: (latlng: LatLng, name: string) => void;
}

export default function LocationSearch({ onSelect }: LocationSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<GeoSearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleInput = (value: string) => {
    setQuery(value);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (value.length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    timerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await searchLocations(value);
        setResults(res);
        setOpen(res.length > 0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
  };

  const handleSelect = (result: GeoSearchResult) => {
    const displayName = [result.name, result.admin1, result.country].filter(Boolean).join(', ');
    setQuery(displayName);
    setOpen(false);
    onSelect({ lat: result.latitude, lng: result.longitude }, displayName);
  };

  return (
    <div className="location-search" ref={containerRef}>
      <div className="location-search__input-wrapper">
        <svg className="location-search__icon" width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="1.5"/>
          <line x1="11" y1="11" x2="14" y2="14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <input
          className="location-search__input"
          type="text"
          placeholder="Search location..."
          value={query}
          onChange={(e) => handleInput(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
        />
        {loading && <span className="location-search__spinner" />}
      </div>

      {open && (
        <ul className="location-search__dropdown">
          {results.map((r) => (
            <li
              key={r.id}
              className="location-search__item"
              onClick={() => handleSelect(r)}
            >
              <span className="location-search__name">{r.name}</span>
              <span className="location-search__meta">
                {[r.admin1, r.country].filter(Boolean).join(', ')}
                {r.elevation != null && ` (${r.elevation}m)`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
