/* ------------------------------------------------------------------ */
/*  WeatherFlow – shared type definitions                             */
/* ------------------------------------------------------------------ */

// ── Geographic ──────────────────────────────────────────────────────
export interface LatLng {
  lat: number;
  lng: number;
}

export interface GeoSearchResult {
  id: number;
  name: string;
  country: string;
  admin1?: string;
  latitude: number;
  longitude: number;
  elevation?: number;
}

// ── Weather models ──────────────────────────────────────────────────
export type WeatherModel =
  | 'best_match'
  | 'gfs'
  | 'ecmwf'
  | 'hrrr'
  | 'icon'
  | 'gem'
  | 'jma'
  | 'meteofrance';

export interface ModelMeta {
  id: WeatherModel;
  label: string;
  provider: string;
  resolution: string;
  color: string;
}

export const MODELS: ModelMeta[] = [
  { id: 'best_match', label: 'Best Match', provider: 'Open-Meteo', resolution: 'varies', color: '#00d4ff' },
  { id: 'gfs',        label: 'GFS',        provider: 'NOAA',       resolution: '0.25°',  color: '#22c55e' },
  { id: 'ecmwf',      label: 'ECMWF IFS',  provider: 'ECMWF',      resolution: '0.1°',   color: '#f59e0b' },
  { id: 'hrrr',       label: 'HRRR',       provider: 'NOAA',       resolution: '3 km',   color: '#ef4444' },
  { id: 'icon',       label: 'ICON',       provider: 'DWD',        resolution: '0.125°', color: '#a855f7' },
  { id: 'gem',        label: 'GEM',        provider: 'ECCC',       resolution: '0.25°',  color: '#ec4899' },
  { id: 'jma',        label: 'JMA',        provider: 'JMA',        resolution: '0.25°',  color: '#14b8a6' },
  { id: 'meteofrance', label: 'Arpege',    provider: 'MeteoFrance', resolution: '0.25°', color: '#6366f1' },
];

// ── Surface variables ───────────────────────────────────────────────
export type SurfaceVariable =
  | 'temperature_2m'
  | 'relative_humidity_2m'
  | 'dewpoint_2m'
  | 'apparent_temperature'
  | 'precipitation'
  | 'rain'
  | 'snowfall'
  | 'snow_depth'
  | 'weather_code'
  | 'pressure_msl'
  | 'surface_pressure'
  | 'cloud_cover'
  | 'cloud_cover_low'
  | 'cloud_cover_mid'
  | 'cloud_cover_high'
  | 'wind_speed_10m'
  | 'wind_direction_10m'
  | 'wind_gusts_10m'
  | 'cape'
  | 'lifted_index'
  | 'visibility'
  | 'shortwave_radiation';

export interface VariableMeta {
  id: SurfaceVariable;
  label: string;
  unit: string;
  category: string;
}

export const SURFACE_VARIABLES: VariableMeta[] = [
  { id: 'temperature_2m',       label: 'Temperature',         unit: '°C',    category: 'Temperature' },
  { id: 'dewpoint_2m',          label: 'Dewpoint',            unit: '°C',    category: 'Temperature' },
  { id: 'apparent_temperature',  label: 'Feels Like',         unit: '°C',    category: 'Temperature' },
  { id: 'relative_humidity_2m', label: 'Relative Humidity',   unit: '%',     category: 'Moisture' },
  { id: 'precipitation',        label: 'Precipitation',       unit: 'mm',    category: 'Precipitation' },
  { id: 'rain',                 label: 'Rain',                unit: 'mm',    category: 'Precipitation' },
  { id: 'snowfall',             label: 'Snowfall',            unit: 'cm',    category: 'Precipitation' },
  { id: 'snow_depth',           label: 'Snow Depth',          unit: 'm',     category: 'Precipitation' },
  { id: 'pressure_msl',         label: 'Sea Level Pressure',  unit: 'hPa',   category: 'Pressure' },
  { id: 'surface_pressure',     label: 'Surface Pressure',    unit: 'hPa',   category: 'Pressure' },
  { id: 'cloud_cover',          label: 'Cloud Cover',         unit: '%',     category: 'Clouds' },
  { id: 'cloud_cover_low',      label: 'Low Clouds',          unit: '%',     category: 'Clouds' },
  { id: 'cloud_cover_mid',      label: 'Mid Clouds',          unit: '%',     category: 'Clouds' },
  { id: 'cloud_cover_high',     label: 'High Clouds',         unit: '%',     category: 'Clouds' },
  { id: 'wind_speed_10m',       label: 'Wind Speed',          unit: 'km/h',  category: 'Wind' },
  { id: 'wind_direction_10m',   label: 'Wind Direction',      unit: '°',     category: 'Wind' },
  { id: 'wind_gusts_10m',       label: 'Wind Gusts',          unit: 'km/h',  category: 'Wind' },
  { id: 'cape',                 label: 'CAPE',                unit: 'J/kg',  category: 'Instability' },
  { id: 'lifted_index',         label: 'Lifted Index',        unit: '°C',    category: 'Instability' },
  { id: 'visibility',           label: 'Visibility',          unit: 'm',     category: 'Other' },
  { id: 'shortwave_radiation',  label: 'Solar Radiation',     unit: 'W/m²',  category: 'Other' },
  { id: 'weather_code',         label: 'Weather Code',        unit: 'WMO',   category: 'Other' },
];

// ── Pressure-level variables (for soundings) ────────────────────────
export const PRESSURE_LEVELS = [
  1000, 975, 950, 925, 900, 875, 850, 800, 750, 700,
  650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
  150, 100, 70, 50, 30,
] as const;

export type PressureLevel = (typeof PRESSURE_LEVELS)[number];

export interface SoundingPoint {
  pressure: number;    // hPa
  temperature: number; // °C
  dewpoint: number;    // °C
  windSpeed: number;   // km/h
  windDir: number;     // degrees
  height: number;      // m (geopotential height)
}

// ── Forecast data ───────────────────────────────────────────────────
export interface HourlyForecast {
  time: string[];
  [key: string]: number[] | string[];
}

export interface PointForecast {
  latitude: number;
  longitude: number;
  elevation: number;
  timezone: string;
  model: WeatherModel;
  hourly: HourlyForecast;
  hourly_units: Record<string, string>;
}

// ── RainViewer ──────────────────────────────────────────────────────
export interface RainViewerFrame {
  time: number;     // unix timestamp
  path: string;     // tile URL path
}

export interface RainViewerData {
  version: string;
  generated: number;
  host: string;
  radar: {
    past: RainViewerFrame[];
    nowcast: RainViewerFrame[];
  };
  satellite: {
    infrared: RainViewerFrame[];
  };
}

// ── Map layers ──────────────────────────────────────────────────────
export type MapLayer = 'radar' | 'satellite' | 'temperature' | 'wind' | 'precipitation' | 'pressure' | 'clouds';

export type BaseMap = 'dark' | 'satellite_base' | 'terrain' | 'streets';

// ── Tropical cyclones ───────────────────────────────────────────────
export interface TropicalCyclone {
  id: string;
  name: string;
  basin: string;
  category: string;
  lat: number;
  lon: number;
  maxWind: number;    // knots
  minPressure: number; // hPa
  movement: string;
  timestamp: string;
  forecast?: CycloneForecastPoint[];
}

export interface CycloneForecastPoint {
  hour: number;
  lat: number;
  lon: number;
  maxWind: number;
  category: string;
}

// ── App state ───────────────────────────────────────────────────────
export type ViewMode = 'map' | 'models' | 'satellite' | 'radar' | 'soundings' | 'tropical';

export interface AppState {
  viewMode: ViewMode;
  selectedModel: WeatherModel;
  selectedVariable: SurfaceVariable;
  selectedLocation: LatLng | null;
  forecastHour: number;
  maxForecastHour: number;
  activeLayers: MapLayer[];
  baseMap: BaseMap;
  sidebarOpen: boolean;
  panelOpen: boolean;
}

// ── Weather code mapping ────────────────────────────────────────────
export const WMO_CODES: Record<number, { description: string; icon: string }> = {
  0:  { description: 'Clear sky',              icon: '01d' },
  1:  { description: 'Mainly clear',           icon: '02d' },
  2:  { description: 'Partly cloudy',          icon: '03d' },
  3:  { description: 'Overcast',               icon: '04d' },
  45: { description: 'Fog',                    icon: '50d' },
  48: { description: 'Depositing rime fog',    icon: '50d' },
  51: { description: 'Light drizzle',          icon: '09d' },
  53: { description: 'Moderate drizzle',       icon: '09d' },
  55: { description: 'Dense drizzle',          icon: '09d' },
  61: { description: 'Slight rain',            icon: '10d' },
  63: { description: 'Moderate rain',          icon: '10d' },
  65: { description: 'Heavy rain',             icon: '10d' },
  66: { description: 'Light freezing rain',    icon: '13d' },
  67: { description: 'Heavy freezing rain',    icon: '13d' },
  71: { description: 'Slight snow',            icon: '13d' },
  73: { description: 'Moderate snow',          icon: '13d' },
  75: { description: 'Heavy snow',             icon: '13d' },
  77: { description: 'Snow grains',            icon: '13d' },
  80: { description: 'Slight rain showers',    icon: '09d' },
  81: { description: 'Moderate rain showers',  icon: '09d' },
  82: { description: 'Violent rain showers',   icon: '09d' },
  85: { description: 'Slight snow showers',    icon: '13d' },
  86: { description: 'Heavy snow showers',     icon: '13d' },
  95: { description: 'Thunderstorm',           icon: '11d' },
  96: { description: 'T-storm w/ slight hail', icon: '11d' },
  99: { description: 'T-storm w/ heavy hail',  icon: '11d' },
};
