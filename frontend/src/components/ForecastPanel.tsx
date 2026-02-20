import { useEffect, useState } from 'react';
import type { LatLng, PointForecast, WeatherModel, WMO_CODES as WMOType } from '../types/weather';
import { MODELS, WMO_CODES } from '../types/weather';
import { fetchPointForecast } from '../api/openMeteo';

interface ForecastPanelProps {
  location: LatLng;
  model: WeatherModel;
  forecastHour: number;
  onModelChange: (model: WeatherModel) => void;
  onClose: () => void;
  onOpenMeteogram: () => void;
  onOpenSounding: () => void;
}

export default function ForecastPanel({
  location,
  model,
  forecastHour,
  onModelChange,
  onClose,
  onOpenMeteogram,
  onOpenSounding,
}: ForecastPanelProps) {
  const [forecast, setForecast] = useState<PointForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  const windArrowStyle = windDir != null
    ? { transform: `rotate(${windDir}deg)` }
    : undefined;

  const forecastTime = forecast?.hourly.time?.[hourIndex];

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

      {loading && <div className="forecast-panel__loading">Loading forecast data...</div>}
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

          {/* Weather condition */}
          {wmoInfo && (
            <div className="forecast-panel__condition">
              {wmoInfo.description}
            </div>
          )}

          {/* Primary values */}
          <div className="forecast-panel__primary">
            <div className="forecast-value forecast-value--large">
              <span className="forecast-value__number">
                {temp != null ? temp.toFixed(1) : '--'}
              </span>
              <span className="forecast-value__unit">C</span>
              <span className="forecast-value__label">Temperature</span>
            </div>
          </div>

          {/* Grid of values */}
          <div className="forecast-panel__grid">
            <div className="forecast-cell">
              <span className="forecast-cell__label">Feels Like</span>
              <span className="forecast-cell__value">{feelsLike != null ? `${feelsLike.toFixed(1)}C` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Dewpoint</span>
              <span className="forecast-cell__value">{dewpoint != null ? `${dewpoint.toFixed(1)}C` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Humidity</span>
              <span className="forecast-cell__value">{rh != null ? `${rh.toFixed(0)}%` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Wind</span>
              <span className="forecast-cell__value">
                {windSpeed != null ? `${windSpeed.toFixed(0)} km/h` : '--'}
                {windDir != null && (
                  <span className="wind-arrow" style={windArrowStyle}>
                    &darr;
                  </span>
                )}
              </span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Gusts</span>
              <span className="forecast-cell__value">{windGust != null ? `${windGust.toFixed(0)} km/h` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Precip</span>
              <span className="forecast-cell__value">{precip != null ? `${precip.toFixed(1)} mm` : '--'}</span>
            </div>
            <div className="forecast-cell">
              <span className="forecast-cell__label">Pressure</span>
              <span className="forecast-cell__value">{pressure != null ? `${pressure.toFixed(1)} hPa` : '--'}</span>
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
              <span className="forecast-cell__value">{vis != null ? `${(vis / 1000).toFixed(1)} km` : '--'}</span>
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
