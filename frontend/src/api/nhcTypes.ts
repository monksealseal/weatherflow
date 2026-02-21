/**
 * TypeScript types for National Hurricane Center (NHC) products and data feeds.
 *
 * Data sources:
 *   - RSS/XML:  https://www.nhc.noaa.gov/aboutrss.shtml
 *   - GIS:      https://www.nhc.noaa.gov/gis/
 *   - Archives: https://www.nhc.noaa.gov/data/
 */

// ---------------------------------------------------------------------------
// Basin identifiers
// ---------------------------------------------------------------------------
export type Basin = 'atlantic' | 'eastern_pacific' | 'central_pacific';

export const BASIN_LABELS: Record<Basin, string> = {
  atlantic: 'Atlantic',
  eastern_pacific: 'Eastern North Pacific',
  central_pacific: 'Central North Pacific',
};

// ---------------------------------------------------------------------------
// Tropical Weather Outlook (TWO)
// ---------------------------------------------------------------------------
export interface OutlookArea {
  id: string;
  title: string;
  lat: number;
  lon: number;
  probability48h: number;   // 0‑100
  probability7d: number;    // 0‑100
  description: string;
  type: 'low' | 'medium' | 'high';
}

export interface TropicalWeatherOutlook {
  basin: Basin;
  issuedAt: string;         // ISO 8601
  text: string;             // full outlook prose
  areas: OutlookArea[];
  imageUrl: string;         // 2‑day / 5‑day graphic URL
}

// ---------------------------------------------------------------------------
// Active tropical cyclones
// ---------------------------------------------------------------------------
export type CycloneClassification =
  | 'tropical_depression'
  | 'tropical_storm'
  | 'hurricane'
  | 'major_hurricane'
  | 'subtropical_depression'
  | 'subtropical_storm'
  | 'post_tropical'
  | 'potential_tropical_cyclone'
  | 'remnant_low';

export interface ActiveCyclone {
  id: string;               // e.g. "AL052024"
  name: string;
  basin: Basin;
  classification: CycloneClassification;
  lat: number;
  lon: number;
  maxWindMph: number;
  maxWindKt: number;
  gustMph: number;
  gustKt: number;
  pressureMb: number;
  movementDir: string;      // e.g. "NNW"
  movementSpeedMph: number;
  lastUpdated: string;      // ISO 8601
  isActive: boolean;
}

// ---------------------------------------------------------------------------
// Forecast positions (used in track / cone)
// ---------------------------------------------------------------------------
export interface ForecastPosition {
  hour: number;             // forecast hour (0, 12, 24, …)
  lat: number;
  lon: number;
  maxWindKt: number;
  classification: CycloneClassification;
  dateTime: string;
}

// ---------------------------------------------------------------------------
// Public Advisory (TCP)
// ---------------------------------------------------------------------------
export interface PublicAdvisory {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  headline: string;
  summary: string;
  text: string;
  watches: WatchWarning[];
  warnings: WatchWarning[];
  nextAdvisoryAt: string;
}

export interface WatchWarning {
  type: 'hurricane_warning' | 'hurricane_watch' | 'tropical_storm_warning' | 'tropical_storm_watch' | 'storm_surge_warning' | 'storm_surge_watch';
  areas: string;
}

// ---------------------------------------------------------------------------
// Forecast / Advisory (TCM)
// ---------------------------------------------------------------------------
export interface ForecastAdvisory {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  currentLat: number;
  currentLon: number;
  maxWindKt: number;
  gustKt: number;
  pressureMb: number;
  movementDir: number;      // degrees
  movementSpeedKt: number;
  forecastPositions: ForecastPosition[];
  windRadii34: WindRadii | null;
  windRadii50: WindRadii | null;
  windRadii64: WindRadii | null;
  text: string;
}

export interface WindRadii {
  ne: number;
  se: number;
  sw: number;
  nw: number;
}

// ---------------------------------------------------------------------------
// Tropical Cyclone Discussion (TCD)
// ---------------------------------------------------------------------------
export interface TropicalCycloneDiscussion {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  forecaster: string;
  text: string;
}

// ---------------------------------------------------------------------------
// Wind Speed Probabilities (PWS)
// ---------------------------------------------------------------------------
export interface WindProbabilityLocation {
  location: string;
  lat: number;
  lon: number;
  prob34kt: number;
  prob50kt: number;
  prob64kt: number;
}

export interface WindSpeedProbabilities {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  validThrough: string;
  locations: WindProbabilityLocation[];
  text: string;
}

// ---------------------------------------------------------------------------
// Storm Surge
// ---------------------------------------------------------------------------
export interface StormSurgeData {
  stormId: string;
  stormName: string;
  issuedAt: string;
  watches: string[];
  warnings: string[];
  peakSurgeFt: number;
  surgeZones: SurgeZone[];
  inundationMapUrl: string;
  text: string;
}

export interface SurgeZone {
  area: string;
  minFt: number;
  maxFt: number;
  tidalAdjusted: boolean;
}

// ---------------------------------------------------------------------------
// Track Forecast Cone
// ---------------------------------------------------------------------------
export interface TrackForecastCone {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  coneImageUrl: string;
  animationUrl: string;
  forecastPositions: ForecastPosition[];
  watches: WatchWarning[];
  warnings: WatchWarning[];
}

// ---------------------------------------------------------------------------
// Marine Forecasts
// ---------------------------------------------------------------------------
export type MarineZone = 'offshore' | 'high_seas' | 'coastal';

export interface MarineForecast {
  zone: MarineZone;
  region: string;
  issuedAt: string;
  synopsis: string;
  forecast: string;
  text: string;
}

// ---------------------------------------------------------------------------
// Tropical Cyclone Report (post-season TCR)
// ---------------------------------------------------------------------------
export interface TropicalCycloneReport {
  stormId: string;
  stormName: string;
  year: number;
  basin: Basin;
  classification: CycloneClassification;
  formationDate: string;
  dissipationDate: string;
  peakWindKt: number;
  minPressureMb: number;
  deaths: number;
  damageUsd: string;        // e.g. "$50 billion"
  summary: string;
  reportUrl: string;
}

// ---------------------------------------------------------------------------
// Graphicast / Key Messages
// ---------------------------------------------------------------------------
export interface KeyMessages {
  stormId: string;
  stormName: string;
  issuedAt: string;
  messages: string[];
  graphicUrl: string;
}

// ---------------------------------------------------------------------------
// Aviation Advisory (TCA)
// ---------------------------------------------------------------------------
export interface AviationAdvisory {
  stormId: string;
  stormName: string;
  advisoryNumber: string;
  issuedAt: string;
  text: string;
}

// ---------------------------------------------------------------------------
// Tropical Cyclone Update (TCU)
// ---------------------------------------------------------------------------
export interface TropicalCycloneUpdate {
  stormId: string;
  stormName: string;
  issuedAt: string;
  text: string;
}

// ---------------------------------------------------------------------------
// Aggregated view for a single storm
// ---------------------------------------------------------------------------
export interface StormProducts {
  cyclone: ActiveCyclone;
  publicAdvisory: PublicAdvisory | null;
  forecastAdvisory: ForecastAdvisory | null;
  discussion: TropicalCycloneDiscussion | null;
  windProbabilities: WindSpeedProbabilities | null;
  stormSurge: StormSurgeData | null;
  trackCone: TrackForecastCone | null;
  keyMessages: KeyMessages | null;
  aviationAdvisory: AviationAdvisory | null;
  updates: TropicalCycloneUpdate[];
}

// ---------------------------------------------------------------------------
// RSS feed item (generic wrapper)
// ---------------------------------------------------------------------------
export interface NHCRssFeedItem {
  title: string;
  link: string;
  description: string;
  pubDate: string;
  guid: string;
}

export interface NHCRssFeed {
  title: string;
  link: string;
  description: string;
  lastBuildDate: string;
  items: NHCRssFeedItem[];
}
