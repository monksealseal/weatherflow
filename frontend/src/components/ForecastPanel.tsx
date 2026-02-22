import { useEffect, useState } from 'react';
import type { LatLng, PointForecast, WeatherModel } from '../types/weather';
import { MODELS, WMO_CODES } from '../types/weather';
import { fetchPointForecast } from '../api/openMeteo';
import type { Units } from '../App';

// Weather condition emoji map based on WMO codes
const WMO_EMOJI: Record<number, string> = {
  0: '\u2600\uFE0F',     // Clear sky
  1: '\u{1F324}\uFE0F',  // Mainly clear
  2: '\u26C5',           // Partly cloudy
  3: '\u2601\uFE0F',     // Overcast
  45: '\u{1F32B}\uFE0F', // Fog
  48: '\u{1F32B}\uFE0F', // Rime fog
  51: '\u{1F326}\uFE0F', // Light drizzle
  53: '\u{1F326}\uFE0F', // Moderate drizzle
  55: '\u{1F327}\uFE0F', // Dense drizzle
  61: '\u{1F327}\uFE0F', // Slight rain
  63: '\u{1F327}\uFE0F', // Moderate rain
  65: '\u{1F327}\uFE0F', // Heavy rain
  66: '\u{1F9CA}',       // Freezing rain
  67: '\u{1F9CA}',       // Heavy freezing rain
  71: '\u{1F328}\uFE0F', // Slight snow
  73: '\u{1F328}\uFE0F', // Moderate snow
  75: '\u{1F328}\uFE0F', // Heavy snow
  77: '\u{1F328}\uFE0F', // Snow grains
  80: '\u{1F326}\uFE0F', // Rain showers
  81: '\u{1F327}\uFE0F', // Moderate showers
  82: '\u{1F327}\uFE0F', // Violent showers
  85: '\u{1F328}\uFE0F', // Snow showers
  86: '\u{1F328}\uFE0F', // Heavy snow showers
  95: '\u26C8\uFE0F',    // Thunderstorm
  96: '\u26C8\uFE0F',    // T-storm w/ hail
  99: '\u26C8\uFE0F',    // T-storm w/ heavy hail
};

// Color based on temperature (Celsius)
function tempColor(tempC: number): string {
  if (tempC <= -20) return '#7c3aed';
  if (tempC <= -10) return '#6366f1';
  if (tempC <= 0)   return '#3b82f6';
  if (tempC <= 10)  return '#06b6d4';
  if (tempC <= 20)  return '#22c55e';
  if (tempC <= 30)  return '#f59e0b';
  if (tempC <= 35)  return '#f97316';
  return '#ef4444';
}

function toF(c: number): number { return c * 9 / 5 + 32; }
function toMph(kmh: number): number { return kmh * 0.621371; }
function toInHg(hPa: number): number { return hPa * 0.02953; }
function toIn(mm: number): number { return mm * 0.03937; }
function toMi(m: number): number { return m / 1609.344; }

interface ForecastPanelProps {
  location: LatLng;
  model: WeatherModel;
  forecastHour: number;
  onModelChange: (model: WeatherModel) => void;
  onClose: () => void;
  onOpenMeteogram: () => void;
  onOpenSounding: () => void;
  units: Units;
}

export default function ForecastPanel({
  location,
  model,
  forecastHour,
  onModelChange,
  onClose,
  onOpenMeteogram,
  onOpenSounding,
  units,
}: ForecastPanelProps) {
  const [forecast, setForecast] = useState<PointForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const imp = units === 'imperial';

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchPointForecast(location.lat, location.lng, model)
      .then((data) => {
        if (!cancelled) setForecast(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [location.lat, location.lng, model]);

  const hourIndex = Math.min(forecastHour, (forecast?.hourly.time?.length ?? 1) - 1);

  const getValue = (key: string): number | null => {
    const arr = forecast?.hourly[key];
    if (!Array.isArray(arr)) return null;
    const val = arr[hourIndex];
    return typeof val === 'number' ? val : null;
  };

  const temp = getValue('temperature_2m');
  const feelsLike = getValue('apparent_temperature');
  const dewpoint = getValue('dewpoint_2m');
  const rh = getValue('relative_humidity_2m');
  const windSpeed = getValue('wind_speed_10m');
  const windDir = getValue('wind_direction_10m');
  const windGust = getValue('wind_gusts_10m');
  const precip = getValue('precipitation');
  const pressure = getValue('pressure_msl');
  const cloud = getValue('cloud_cover');
  const cape = getValue('cape');
  const vis = getValue('visibility');
  const weatherCode = getValue('weather_code');

  const wmoInfo = weatherCode != null ? WMO_CODES[weatherCode] : null;
  const emoji = weatherCode != null ? WMO_EMOJI[weatherCode] ?? '' : '';

  const windArrowStyle = windDir != null
    ? { transform: `rotate(${windDir}deg)` }
    : undefined;

  const forecastTime = forecast?.hourly.time?.[hourIndex];

  // Formatting helpers based on units
  const fmtTemp = (c: number | null) => c != null ? `${(imp ? toF(c) : c).toFixed(1)}${imp ? 'F' : 'C'}` : '--';
  const fmtWind = (kmh: number | null) => kmh != null ? `${(imp ? toMph(kmh) : kmh).toFixed(0)} ${imp ? 'mph' : 'km/h'}` : '--';
  const fmtPrecip = (mm: number | null) => mm != null ? `${(imp ? toIn(mm) : mm).toFixed(imp ? 2 : 1)} ${imp ? 'in' : 'mm'}` : '--';
  const fmtPressure = (hPa: number | null) => hPa != null ? `${(imp ? toInHg(hPa) : hPa).toFixed(imp ? 2 : 1)} ${imp ? 'inHg' : 'hPa'}` : '--';
  const fmtVis = (m: number | null) => m != null ? `${(imp ? toMi(m) : m / 1000).toFixed(1)} ${imp ? 'mi' : 'km'}` : '--';

  return (
    <div className="forecast-panel">
      <div className="forecast-panel__header">
        <div>
          <h3 className="forecast-panel__title">Point Forecast</h3>
          <p className="forecast-panel__coords">
            {location.lat.toFixed(3)}N, {location.lng.toFixed(3)}E
            {forecast && ` | ${forecast.elevation}m`}
          </p>
        </div>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>

      {/* Model selector */}
      <div className="forecast-panel__model-selector">
        {MODELS.slice(0, 5).map((m) => (
          <button
            key={m.id}
            className={`model-chip ${model === m.id ? 'model-chip--active' : ''}`}
            style={model === m.id ? { borderColor: m.color, color: m.color } : undefined}
            onClick={() => onModelChange(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {loading && (
        <div className="forecast-panel__loading">
          <div className="loading-skeleton">
            <div className="skeleton-line skeleton-line--lg" />
            <div className="skeleton-line skeleton-line--sm" />
            <div className="skeleton-grid">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="skeleton-cell" />
              ))}
            </div>
          </div>
        </div>
      )}
      {error && <div className="forecast-panel__error">{error}</div>}

      {forecast && !loading && (
        <>
          {/* Time display */}
          {forecastTime && (
            <div className="forecast-panel__time">
              {new Date(forecastTime as string).toLocaleString(undefined, {
                weekday: 'short',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </div>
          )}

          {/* Weather condition with emoji */}
          {wmoInfo && (
            <div className="forecast-panel__condition">
              <span className="forecast-panel__emoji">{emoji}</span>
              {wmoInfo.description}
            </div>
          )}

          {/* Primary temperature with color coding */}
          <div className="forecast-panel__primary">
            <div className="forecast-value forecast-value--large">
              <span
                className="forecast-value__number"
                style={temp != null ? { color: tempColor(temp) } : undefined}
              >
                {temp != null ? (imp ? toF(temp) : temp).toFixed(1) : '--'}
              </span>
              <span className="forecast-value__unit" style={temp != null ? { color: tempColor(temp) } : undefined}>
                {imp ? 'F' : 'C'}
              </span>
              <span className="forecast-value__label">Temperature</span>
            </div>
          </div>

          {/* Grid of values */}
          <div className="forecast-panel__grid">
            <div className="forecast-cell">
              <span className="forecast-cell__label">Feels Like</span>
              <span className="forecast-cell__value">{fmtTemp(feelsLike)}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Dewpoint</span>
              <span className="forecast-cell__value">{fmtTemp(dewpoint)}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Humidity</span>
              <span className="forecast-cell__value">{rh != null ? `${rh.toFixed(0)}%` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Wind</span>
              <span className="forecast-cell__value">
                {fmtWind(windSpeed)}
                {windDir != null && (
                  <span className="wind-arrow" style={windArrowStyle}>
                    &darr;
                  </span>
                )}
              </span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Gusts</span>
              <span className="forecast-cell__value">{fmtWind(windGust)}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Precip</span>
              <span className="forecast-cell__value">{fmtPrecip(precip)}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Pressure</span>
              <span className="forecast-cell__value">{fmtPressure(pressure)}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Cloud Cover</span>
              <span className="forecast-cell__value">{cloud != null ? `${cloud.toFixed(0)}%` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">CAPE</span>
              <span className="forecast-cell__value">{cape != null ? `${cape.toFixed(0)} J/kg` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Visibility</span>
              <span className="forecast-cell__value">{fmtVis(vis)}</span>
            </div>
          </div>

          {/* Action buttons */}
          <div className="forecast-panel__actions">
            <button className="forecast-action" onClick={onOpenMeteogram}>
              Meteogram
            </button>
            <button className="forecast-action" onClick={onOpenSounding}>
              Sounding
            </button>
          </div>
        </>
      )}
    </div>
  );
}
