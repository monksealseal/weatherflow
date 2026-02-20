/* ------------------------------------------------------------------ */
/*  Open-Meteo API client                                             */
/*  Free weather data – no API key required                           */
/*  https://open-meteo.com/en/docs                                    */
/* ------------------------------------------------------------------ */

import type {
  WeatherModel,
  SurfaceVariable,
  PointForecast,
  SoundingPoint,
  GeoSearchResult,
  PRESSURE_LEVELS,
} from '../types/weather';

// ── Helpers ─────────────────────────────────────────────────────────

const BASE = 'https://api.open-meteo.com/v1';

function modelEndpoint(model: WeatherModel): string {
  switch (model) {
    case 'gfs':         return `${BASE}/gfs`;
    case 'ecmwf':       return `${BASE}/ecmwf`;
    case 'hrrr':        return `${BASE}/gfs`; // HRRR is part of GFS endpoint
    case 'icon':        return `${BASE}/dwd-icon`;
    case 'gem':         return `${BASE}/gem`;
    case 'jma':         return `${BASE}/jma`;
    case 'meteofrance': return `${BASE}/meteofrance`;
    default:            return `${BASE}/forecast`;
  }
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Open-Meteo error ${res.status}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

// ── Surface forecast ────────────────────────────────────────────────

const DEFAULT_HOURLY: SurfaceVariable[] = [
  'temperature_2m',
  'dewpoint_2m',
  'relative_humidity_2m',
  'apparent_temperature',
  'precipitation',
  'rain',
  'snowfall',
  'weather_code',
  'pressure_msl',
  'surface_pressure',
  'cloud_cover',
  'cloud_cover_low',
  'cloud_cover_mid',
  'cloud_cover_high',
  'wind_speed_10m',
  'wind_direction_10m',
  'wind_gusts_10m',
  'cape',
  'visibility',
  'shortwave_radiation',
];

export async function fetchPointForecast(
  lat: number,
  lng: number,
  model: WeatherModel = 'best_match',
  variables: SurfaceVariable[] = DEFAULT_HOURLY,
  forecastDays = 7,
): Promise<PointForecast> {
  const endpoint = modelEndpoint(model);
  const hourly = variables.join(',');
  const params = new URLSearchParams({
    latitude: lat.toFixed(4),
    longitude: lng.toFixed(4),
    hourly,
    wind_speed_unit: 'kmh',
    precipitation_unit: 'mm',
    timezone: 'auto',
    forecast_days: String(forecastDays),
  });

  // HRRR uses minutely_15 data and is US-only within GFS endpoint
  if (model === 'hrrr') {
    params.set('models', 'gfs_hrrr');
  }

  const data = await fetchJSON<Record<string, unknown>>(
    `${endpoint}?${params}`,
  );

  return {
    latitude: data.latitude as number,
    longitude: data.longitude as number,
    elevation: (data.elevation as number) ?? 0,
    timezone: (data.timezone as string) ?? 'UTC',
    model,
    hourly: data.hourly as PointForecast['hourly'],
    hourly_units: (data.hourly_units as Record<string, string>) ?? {},
  };
}

// ── Multi-model comparison ──────────────────────────────────────────

export async function fetchMultiModelForecast(
  lat: number,
  lng: number,
  models: WeatherModel[],
  variables: SurfaceVariable[] = ['temperature_2m', 'wind_speed_10m', 'precipitation'],
): Promise<PointForecast[]> {
  const promises = models.map((m) => fetchPointForecast(lat, lng, m, variables));
  return Promise.allSettled(promises).then((results) =>
    results
      .filter((r): r is PromiseFulfilledResult<PointForecast> => r.status === 'fulfilled')
      .map((r) => r.value),
  );
}

// ── Pressure-level (sounding) data ──────────────────────────────────

const SOUNDING_LEVELS: (typeof PRESSURE_LEVELS)[number][] = [
  1000, 975, 950, 925, 900, 875, 850, 800, 750, 700,
  650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
  150, 100, 70, 50,
];

export async function fetchSounding(
  lat: number,
  lng: number,
  model: WeatherModel = 'best_match',
  forecastHour = 0,
): Promise<SoundingPoint[]> {
  const endpoint = modelEndpoint(model);

  // Build variable list for each pressure level
  const vars: string[] = [];
  for (const lvl of SOUNDING_LEVELS) {
    vars.push(
      `temperature_${lvl}hPa`,
      `relative_humidity_${lvl}hPa`,
      `wind_speed_${lvl}hPa`,
      `wind_direction_${lvl}hPa`,
      `geopotential_height_${lvl}hPa`,
    );
  }

  const params = new URLSearchParams({
    latitude: lat.toFixed(4),
    longitude: lng.toFixed(4),
    hourly: vars.join(','),
    wind_speed_unit: 'kmh',
    timezone: 'UTC',
    forecast_days: '3',
  });

  const data = await fetchJSON<Record<string, unknown>>(`${endpoint}?${params}`);
  const hourly = data.hourly as Record<string, (number | null)[]>;
  const times = hourly.time as unknown as string[];

  // Find the time index closest to forecastHour
  const idx = Math.min(forecastHour, times.length - 1);

  const points: SoundingPoint[] = [];
  for (const lvl of SOUNDING_LEVELS) {
    const temp = hourly[`temperature_${lvl}hPa`]?.[idx];
    const rh   = hourly[`relative_humidity_${lvl}hPa`]?.[idx];
    const ws   = hourly[`wind_speed_${lvl}hPa`]?.[idx];
    const wd   = hourly[`wind_direction_${lvl}hPa`]?.[idx];
    const gh   = hourly[`geopotential_height_${lvl}hPa`]?.[idx];

    if (temp == null) continue;

    // Estimate dewpoint from RH and temp (Magnus formula)
    const dp =
      rh != null
        ? calcDewpoint(temp, rh)
        : temp - 10;

    points.push({
      pressure: lvl,
      temperature: temp,
      dewpoint: dp,
      windSpeed: ws ?? 0,
      windDir: wd ?? 0,
      height: gh ?? estimateHeight(lvl),
    });
  }

  return points;
}

function calcDewpoint(tempC: number, rh: number): number {
  // Magnus-Tetens approximation
  const a = 17.27;
  const b = 237.7;
  const gamma = (a * tempC) / (b + tempC) + Math.log(rh / 100);
  return (b * gamma) / (a - gamma);
}

function estimateHeight(pressure: number): number {
  // Approximate geopotential height using barometric formula
  return 44330 * (1 - Math.pow(pressure / 1013.25, 0.1903));
}

// ── Geocoding ───────────────────────────────────────────────────────

const GEO_BASE = 'https://geocoding-api.open-meteo.com/v1';

export async function searchLocations(query: string): Promise<GeoSearchResult[]> {
  if (query.length < 2) return [];
  const params = new URLSearchParams({
    name: query,
    count: '8',
    language: 'en',
    format: 'json',
  });
  const data = await fetchJSON<{ results?: GeoSearchResult[] }>(
    `${GEO_BASE}/search?${params}`,
  );
  return data.results ?? [];
}
