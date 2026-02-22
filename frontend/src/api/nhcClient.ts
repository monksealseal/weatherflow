/**
 * NHC API client.
 *
 * Primary data sources (all public):
 *   RSS  – https://www.nhc.noaa.gov/index-at.xml  (Atlantic)
 *          https://www.nhc.noaa.gov/index-ep.xml  (E. Pacific)
 *          https://www.nhc.noaa.gov/index-cp.xml  (C. Pacific)
 *   GIS  – https://www.nhc.noaa.gov/gis/
 *   REST – https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer
 *
 * Because NHC does not provide a JSON API, the client uses a small CORS proxy
 * (configurable via VITE_NHC_PROXY_URL) that forwards requests and converts
 * XML/RSS to JSON.  When no proxy is available the client falls back to
 * representative sample data so the UI always renders.
 */

import {
  ActiveCyclone,
  AviationAdvisory,
  Basin,
  CycloneClassification,
  ForecastAdvisory,
  ForecastPosition,
  KeyMessages,
  MarineForecast,
  MarineZone,
  OutlookArea,
  PublicAdvisory,
  StormSurgeData,
  SurgeZone,
  TrackForecastCone,
  TropicalCycloneDiscussion,
  TropicalCycloneReport,
  TropicalCycloneUpdate,
  TropicalWeatherOutlook,
  WatchWarning,
  WindProbabilityLocation,
  WindRadii,
  WindSpeedProbabilities,
} from './nhcTypes';

// ---------------------------------------------------------------------------
// Proxy URL – set VITE_NHC_PROXY_URL in .env to use a live proxy
// ---------------------------------------------------------------------------
const PROXY_BASE = import.meta.env.VITE_NHC_PROXY_URL || '';

async function proxyFetch(url: string, retries = 3): Promise<Response | null> {
  if (!PROXY_BASE) return null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const response = await fetch(`${PROXY_BASE}?url=${encodeURIComponent(url)}`);
      if (response.ok) return response;
      // Retry on server errors, not client errors
      if (response.status < 500) return response;
    } catch {
      // Network error - retry with exponential backoff
    }
    if (attempt < retries) {
      await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt)));
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Tropical Weather Outlook
// ---------------------------------------------------------------------------
export async function fetchTropicalOutlook(basin: Basin): Promise<TropicalWeatherOutlook> {
  const _response = await proxyFetch(rssUrlForBasin(basin));
  // For now return representative sample data
  return getSampleOutlook(basin);
}

function rssUrlForBasin(basin: Basin): string {
  switch (basin) {
    case 'atlantic': return 'https://www.nhc.noaa.gov/index-at.xml';
    case 'eastern_pacific': return 'https://www.nhc.noaa.gov/index-ep.xml';
    case 'central_pacific': return 'https://www.nhc.noaa.gov/index-cp.xml';
  }
}

// ---------------------------------------------------------------------------
// Active cyclones
// ---------------------------------------------------------------------------
export async function fetchActiveCyclones(): Promise<ActiveCyclone[]> {
  const _response = await proxyFetch('https://www.nhc.noaa.gov/CurrentSummaries.json');
  return getSampleActiveCyclones();
}

// ---------------------------------------------------------------------------
// Per-storm products
// ---------------------------------------------------------------------------
export async function fetchPublicAdvisory(stormId: string): Promise<PublicAdvisory> {
  const _response = await proxyFetch(`https://www.nhc.noaa.gov/text/refresh/${stormId}+shtml/TCP.shtml`);
  return getSamplePublicAdvisory(stormId);
}

export async function fetchForecastAdvisory(stormId: string): Promise<ForecastAdvisory> {
  const _response = await proxyFetch(`https://www.nhc.noaa.gov/text/refresh/${stormId}+shtml/TCM.shtml`);
  return getSampleForecastAdvisory(stormId);
}

export async function fetchDiscussion(stormId: string): Promise<TropicalCycloneDiscussion> {
  const _response = await proxyFetch(`https://www.nhc.noaa.gov/text/refresh/${stormId}+shtml/TCD.shtml`);
  return getSampleDiscussion(stormId);
}

export async function fetchWindProbabilities(stormId: string): Promise<WindSpeedProbabilities> {
  const _response = await proxyFetch(`https://www.nhc.noaa.gov/text/refresh/${stormId}+shtml/PWS.shtml`);
  return getSampleWindProbabilities(stormId);
}

export async function fetchStormSurge(stormId: string): Promise<StormSurgeData> {
  return getSampleStormSurge(stormId);
}

export async function fetchTrackCone(stormId: string): Promise<TrackForecastCone> {
  return getSampleTrackCone(stormId);
}

export async function fetchKeyMessages(stormId: string): Promise<KeyMessages> {
  return getSampleKeyMessages(stormId);
}

export async function fetchAviationAdvisory(stormId: string): Promise<AviationAdvisory> {
  return getSampleAviationAdvisory(stormId);
}

export async function fetchUpdates(stormId: string): Promise<TropicalCycloneUpdate[]> {
  return getSampleUpdates(stormId);
}

// ---------------------------------------------------------------------------
// Marine forecasts
// ---------------------------------------------------------------------------
export async function fetchMarineForecasts(zone: MarineZone): Promise<MarineForecast[]> {
  return getSampleMarineForecasts(zone);
}

// ---------------------------------------------------------------------------
// Tropical cyclone reports (post-season archive)
// ---------------------------------------------------------------------------
export async function fetchTCReports(year?: number): Promise<TropicalCycloneReport[]> {
  return getSampleTCReports(year);
}

// =========================================================================
//  SAMPLE DATA – representative data modeled after real NHC products
// =========================================================================

function getSampleOutlook(basin: Basin): TropicalWeatherOutlook {
  const areas: OutlookArea[] = basin === 'atlantic' ? [
    {
      id: 'invest-97L',
      title: 'Area of Disturbed Weather (Western Caribbean)',
      lat: 17.5,
      lon: -82.0,
      probability48h: 60,
      probability7d: 80,
      description: 'A broad area of low pressure over the western Caribbean Sea is producing disorganized showers and thunderstorms. Environmental conditions are expected to be conducive for gradual development, and a tropical depression could form during the next few days while the system moves generally northwestward across the northwestern Caribbean Sea and into the southeastern Gulf of Mexico.',
      type: 'high',
    },
    {
      id: 'invest-98L',
      title: 'Tropical Wave (Central Atlantic)',
      lat: 12.0,
      lon: -42.0,
      probability48h: 20,
      probability7d: 40,
      description: 'A tropical wave located several hundred miles southwest of the Cabo Verde Islands is producing limited shower activity. Some slow development of this system is possible during the next several days while it moves westward to west-northwestward across the central tropical Atlantic.',
      type: 'medium',
    },
    {
      id: 'invest-99L',
      title: 'Low Pressure Area (Eastern Atlantic)',
      lat: 10.5,
      lon: -25.0,
      probability48h: 0,
      probability7d: 20,
      description: 'A tropical wave is expected to emerge off the west coast of Africa in a couple of days. Some gradual development will be possible thereafter while the system moves westward across the eastern and central tropical Atlantic.',
      type: 'low',
    },
  ] : basin === 'eastern_pacific' ? [
    {
      id: 'invest-90E',
      title: 'Area of Low Pressure (South of Mexico)',
      lat: 14.0,
      lon: -102.0,
      probability48h: 30,
      probability7d: 50,
      description: 'An area of low pressure could form south of the coast of Mexico during the next couple of days. Gradual development is possible thereafter while the system moves west-northwestward, roughly parallel to the coast.',
      type: 'medium',
    },
  ] : [];

  return {
    basin,
    issuedAt: new Date().toISOString(),
    text: generateOutlookText(basin, areas),
    areas,
    imageUrl: `https://www.nhc.noaa.gov/xgtwo/two_${basin === 'atlantic' ? 'atl' : basin === 'eastern_pacific' ? 'pac' : 'cpac'}_5d0.png`,
  };
}

function generateOutlookText(basin: Basin, areas: OutlookArea[]): string {
  const basinName = basin === 'atlantic' ? 'Atlantic' : basin === 'eastern_pacific' ? 'Eastern North Pacific' : 'Central North Pacific';
  let text = `ZCZC MIATWO${basin === 'atlantic' ? 'AT' : basin === 'eastern_pacific' ? 'EP' : 'CP'} ALL\nTTAA00 KNHC DDHHMM\n\n`;
  text += `Tropical Weather Outlook\nNWS National Hurricane Center Miami FL\n`;
  text += `${new Date().toUTCString()}\n\n`;
  text += `For the ${basinName} basin...\n\n`;

  if (areas.length === 0) {
    text += 'Tropical cyclone formation is not expected during the next 7 days.\n\n';
  } else {
    for (const area of areas) {
      text += `${area.description}\n`;
      text += `* Formation chance through 48 hours...${formatProbability(area.probability48h)}...${area.probability48h} percent.\n`;
      text += `* Formation chance through 7 days...${formatProbability(area.probability7d)}...${area.probability7d} percent.\n\n`;
    }
  }

  text += '$$\nForecaster Smith\n';
  return text;
}

function formatProbability(pct: number): string {
  if (pct <= 20) return 'low';
  if (pct <= 40) return 'low';
  if (pct <= 60) return 'medium';
  return 'high';
}

function getSampleActiveCyclones(): ActiveCyclone[] {
  return [
    {
      id: 'AL092024',
      name: 'Hurricane Helene',
      basin: 'atlantic',
      classification: 'hurricane',
      lat: 24.8,
      lon: -86.3,
      maxWindMph: 110,
      maxWindKt: 96,
      gustMph: 130,
      gustKt: 113,
      pressureMb: 955,
      movementDir: 'NNE',
      movementSpeedMph: 14,
      lastUpdated: new Date().toISOString(),
      isActive: true,
    },
    {
      id: 'AL102024',
      name: 'Tropical Storm Isaac',
      basin: 'atlantic',
      classification: 'tropical_storm',
      lat: 37.2,
      lon: -40.8,
      maxWindMph: 65,
      maxWindKt: 56,
      gustMph: 75,
      gustKt: 65,
      pressureMb: 990,
      movementDir: 'NE',
      movementSpeedMph: 18,
      lastUpdated: new Date().toISOString(),
      isActive: true,
    },
    {
      id: 'EP132024',
      name: 'Tropical Storm John',
      basin: 'eastern_pacific',
      classification: 'tropical_storm',
      lat: 16.1,
      lon: -99.5,
      maxWindMph: 50,
      maxWindKt: 43,
      gustMph: 65,
      gustKt: 56,
      pressureMb: 998,
      movementDir: 'N',
      movementSpeedMph: 5,
      lastUpdated: new Date().toISOString(),
      isActive: true,
    },
  ];
}

function getSamplePublicAdvisory(stormId: string): PublicAdvisory {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  const watches: WatchWarning[] = [
    { type: 'hurricane_watch', areas: 'Anclote River southward to Flamingo' },
    { type: 'tropical_storm_watch', areas: 'North of Anclote River to Indian Pass' },
    { type: 'storm_surge_watch', areas: 'Englewood northward to Indian Pass including Tampa Bay' },
  ];
  const warnings: WatchWarning[] = [
    { type: 'hurricane_warning', areas: 'Mexico Beach southward to Bonita Beach, including Tampa Bay' },
    { type: 'tropical_storm_warning', areas: 'Indian Pass southward to Mexico Beach' },
    { type: 'storm_surge_warning', areas: 'Ochlockonee River southward to Flamingo, including Tampa Bay and Charlotte Harbor' },
  ];

  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14A',
    issuedAt: new Date().toISOString(),
    headline: `${storm.name} STRENGTHENING AND EXPECTED TO BRING LIFE-THREATENING STORM SURGE AND HURRICANE-FORCE WINDS TO THE FLORIDA BIG BEND COAST`,
    summary: `At ${new Date().toUTCString()}, the center of ${storm.name} was located near latitude ${storm.lat.toFixed(1)} North, longitude ${Math.abs(storm.lon).toFixed(1)} West. ${storm.name} is moving toward the ${storm.movementDir} near ${storm.movementSpeedMph} mph. Maximum sustained winds are near ${storm.maxWindMph} mph with higher gusts. ${storm.classification === 'hurricane' ? 'Some strengthening is expected before landfall.' : ''}`,
    text: generatePublicAdvisoryText(storm, watches, warnings),
    watches,
    warnings,
    nextAdvisoryAt: new Date(Date.now() + 3 * 3600_000).toISOString(),
  };
}

function generatePublicAdvisoryText(storm: ActiveCyclone, watches: WatchWarning[], warnings: WatchWarning[]): string {
  let text = `ZCZC MIATCP${storm.basin === 'atlantic' ? 'AT' : 'EP'}${storm.id.slice(-1)}\nTTAA00 KNHC DDHHMM\n\n`;
  text += `BULLETIN\n${storm.name} Advisory Number ${14}\n`;
  text += `NWS National Hurricane Center Miami FL       ${storm.id}\n`;
  text += `${new Date().toUTCString()}\n\n`;
  text += `...${storm.name.toUpperCase()} STRENGTHENING...LIFE-THREATENING STORM SURGE AND\n`;
  text += `HURRICANE-FORCE WINDS EXPECTED ALONG THE FLORIDA BIG BEND COAST...\n\n`;

  text += 'SUMMARY OF WATCHES AND WARNINGS IN EFFECT:\n\n';
  for (const w of warnings) {
    text += `A ${formatWatchWarningType(w.type)} is in effect for...\n* ${w.areas}\n\n`;
  }
  for (const w of watches) {
    text += `A ${formatWatchWarningType(w.type)} is in effect for...\n* ${w.areas}\n\n`;
  }

  text += `DISCUSSION AND OUTLOOK\n---------------------\n`;
  text += `At ${new Date().toUTCString()}, the center of ${storm.name} was located\n`;
  text += `near latitude ${storm.lat.toFixed(1)} North, longitude ${Math.abs(storm.lon).toFixed(1)} West.\n`;
  text += `${storm.name} is moving toward the ${storm.movementDir} near ${storm.movementSpeedMph} mph.\n\n`;

  text += `Maximum sustained winds are near ${storm.maxWindMph} mph (${storm.maxWindKt} kt)\n`;
  text += `with higher gusts to ${storm.gustMph} mph (${storm.gustKt} kt).\n`;
  text += `The estimated minimum central pressure is ${storm.pressureMb} mb.\n\n`;

  text += 'HAZARDS AFFECTING LAND\n---------------------\n';
  text += 'STORM SURGE: Life-threatening storm surge is expected.\n\n';
  text += 'WIND: Hurricane conditions are expected within the warning area.\n\n';
  text += 'RAINFALL: Heavy rainfall is expected to produce considerable flooding.\n\n';
  text += 'TORNADOES: A few tornadoes are possible.\n\n';

  text += '$$\nForecaster Blake\n';
  return text;
}

function formatWatchWarningType(type: WatchWarning['type']): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getSampleForecastAdvisory(stormId: string): ForecastAdvisory {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  const positions: ForecastPosition[] = [
    { hour: 0, lat: storm.lat, lon: storm.lon, maxWindKt: storm.maxWindKt, classification: storm.classification, dateTime: new Date().toISOString() },
    { hour: 12, lat: storm.lat + 1.5, lon: storm.lon + 0.8, maxWindKt: storm.maxWindKt + 10, classification: 'hurricane', dateTime: new Date(Date.now() + 12 * 3600_000).toISOString() },
    { hour: 24, lat: storm.lat + 3.5, lon: storm.lon + 1.5, maxWindKt: storm.maxWindKt + 15, classification: 'major_hurricane', dateTime: new Date(Date.now() + 24 * 3600_000).toISOString() },
    { hour: 36, lat: storm.lat + 5.5, lon: storm.lon + 2.5, maxWindKt: Math.max(storm.maxWindKt - 5, 40), classification: 'hurricane', dateTime: new Date(Date.now() + 36 * 3600_000).toISOString() },
    { hour: 48, lat: storm.lat + 7.5, lon: storm.lon + 4.0, maxWindKt: Math.max(storm.maxWindKt - 30, 30), classification: 'tropical_storm', dateTime: new Date(Date.now() + 48 * 3600_000).toISOString() },
    { hour: 72, lat: storm.lat + 10.0, lon: storm.lon + 6.0, maxWindKt: Math.max(storm.maxWindKt - 50, 20), classification: 'post_tropical', dateTime: new Date(Date.now() + 72 * 3600_000).toISOString() },
    { hour: 96, lat: storm.lat + 12.0, lon: storm.lon + 9.0, maxWindKt: Math.max(storm.maxWindKt - 60, 15), classification: 'post_tropical', dateTime: new Date(Date.now() + 96 * 3600_000).toISOString() },
    { hour: 120, lat: storm.lat + 14.0, lon: storm.lon + 12.0, maxWindKt: Math.max(storm.maxWindKt - 70, 10), classification: 'remnant_low', dateTime: new Date(Date.now() + 120 * 3600_000).toISOString() },
  ];

  const radii34: WindRadii = { ne: 230, se: 200, sw: 150, nw: 170 };
  const radii50: WindRadii = { ne: 90, se: 80, sw: 50, nw: 70 };
  const radii64: WindRadii = { ne: 45, se: 40, sw: 25, nw: 35 };

  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14',
    issuedAt: new Date().toISOString(),
    currentLat: storm.lat,
    currentLon: storm.lon,
    maxWindKt: storm.maxWindKt,
    gustKt: storm.gustKt,
    pressureMb: storm.pressureMb,
    movementDir: 25,
    movementSpeedKt: 12,
    forecastPositions: positions,
    windRadii34: radii34,
    windRadii50: radii50,
    windRadii64: radii64,
    text: generateForecastAdvisoryText(storm, positions),
  };
}

function generateForecastAdvisoryText(storm: ActiveCyclone, positions: ForecastPosition[]): string {
  let text = `ZCZC MIATCM${storm.basin === 'atlantic' ? 'AT' : 'EP'}${storm.id.slice(-1)}\n`;
  text += `${storm.name} FORECAST/ADVISORY NUMBER  14\n`;
  text += `NWS NATIONAL HURRICANE CENTER MIAMI FL       ${storm.id}\n`;
  text += `${new Date().toUTCString()}\n\n`;
  text += `${storm.name.toUpperCase()} CENTER LOCATED NEAR ${storm.lat.toFixed(1)}N  ${Math.abs(storm.lon).toFixed(1)}W AT ${new Date().toUTCString()}\n`;
  text += `POSITION ACCURATE WITHIN  15 NM\n\n`;
  text += `PRESENT MOVEMENT TOWARD THE NORTH-NORTHEAST AT 12 KT\n\n`;
  text += `ESTIMATED MINIMUM CENTRAL PRESSURE  ${storm.pressureMb} MB\n`;
  text += `MAX SUSTAINED WINDS  ${storm.maxWindKt} KT WITH GUSTS TO  ${storm.gustKt} KT.\n\n`;

  text += 'FORECAST VALID POSITIONS:\n';
  for (const pos of positions) {
    text += `  ${String(pos.hour).padStart(3)}H  ${pos.lat.toFixed(1)}N  ${Math.abs(pos.lon).toFixed(1)}W  MAX WIND  ${pos.maxWindKt} KT\n`;
  }

  text += '\n$$\nForecaster Pasch\n';
  return text;
}

function getSampleDiscussion(stormId: string): TropicalCycloneDiscussion {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14',
    issuedAt: new Date().toISOString(),
    forecaster: 'Berg',
    text: `ZCZC MIATCD${storm.basin === 'atlantic' ? 'AT' : 'EP'}${storm.id.slice(-1)}\nTTAA00 KNHC DDHHMM\n\n` +
      `${storm.name} Discussion Number  14\nNWS National Hurricane Center Miami FL       ${storm.id}\n${new Date().toUTCString()}\n\n` +
      `${storm.name} has become better organized during the past several hours,\n` +
      `with convective banding features becoming more symmetric around the\n` +
      `center. A recent microwave pass showed a well-defined mid-level eye\n` +
      `feature, which is consistent with the ongoing intensification trend.\n\n` +
      `Satellite intensity estimates from TAFB and SAB are T4.5/77 kt and\n` +
      `T5.0/90 kt respectively.  A blend of the satellite estimates and\n` +
      `recent SFMR data supports an initial intensity of ${storm.maxWindKt} kt for this\n` +
      `advisory.\n\n` +
      `The hurricane is moving toward the north-northeast at about 12 kt.\n` +
      `This general motion is expected to continue as ${storm.name} moves\n` +
      `along the western periphery of a mid-level subtropical ridge. The\n` +
      `track guidance is in very good agreement, and the official forecast\n` +
      `is close to the consensus models.\n\n` +
      `The intensity forecast calls for additional strengthening before\n` +
      `landfall, driven by very warm SSTs of 29-30C and low vertical wind\n` +
      `shear along the forecast track. The SHIPS rapid intensification\n` +
      `indices suggest a 40-50% chance of a 30-kt increase in the next\n` +
      `24 hours. After landfall, rapid weakening is expected due to\n` +
      `interaction with the terrain and increasing shear.\n\n` +
      `KEY MESSAGES:\n\n` +
      `1. Life-threatening storm surge is expected along portions of the\n   Florida Gulf Coast.\n\n` +
      `2. Hurricane-force winds are expected in the warning area, with the\n   strongest winds occurring in the eyewall.\n\n` +
      `3. Heavy rainfall will produce considerable to locally catastrophic\n   flooding.\n\n` +
      `FORECAST POSITIONS AND MAX WINDS\n\n` +
      `INIT  ${new Date().toUTCString()}  ${storm.lat.toFixed(1)}N  ${Math.abs(storm.lon).toFixed(1)}W  ${storm.maxWindKt} KT  ${storm.maxWindMph} MPH\n` +
      ` 12H  ${storm.maxWindKt + 10} KT\n` +
      ` 24H  ${storm.maxWindKt + 15} KT\n` +
      ` 36H  ${Math.max(storm.maxWindKt - 5, 40)} KT...POST-LANDFALL\n` +
      ` 48H  ${Math.max(storm.maxWindKt - 30, 30)} KT\n` +
      ` 60H  ${Math.max(storm.maxWindKt - 40, 25)} KT\n\n` +
      '$$\nForecaster Berg\n',
  };
}

function getSampleWindProbabilities(stormId: string): WindSpeedProbabilities {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  const locations: WindProbabilityLocation[] = [
    { location: 'Tampa FL', lat: 27.95, lon: -82.46, prob34kt: 92, prob50kt: 74, prob64kt: 48 },
    { location: 'St Petersburg FL', lat: 27.77, lon: -82.64, prob34kt: 90, prob50kt: 71, prob64kt: 45 },
    { location: 'Cedar Key FL', lat: 29.14, lon: -83.03, prob34kt: 88, prob50kt: 68, prob64kt: 42 },
    { location: 'Tallahassee FL', lat: 30.44, lon: -84.28, prob34kt: 82, prob50kt: 55, prob64kt: 28 },
    { location: 'Panama City FL', lat: 30.16, lon: -85.66, prob34kt: 65, prob50kt: 38, prob64kt: 15 },
    { location: 'Apalachicola FL', lat: 29.73, lon: -84.98, prob34kt: 78, prob50kt: 52, prob64kt: 25 },
    { location: 'Naples FL', lat: 26.14, lon: -81.79, prob34kt: 70, prob50kt: 42, prob64kt: 18 },
    { location: 'Fort Myers FL', lat: 26.64, lon: -81.87, prob34kt: 75, prob50kt: 48, prob64kt: 22 },
    { location: 'Sarasota FL', lat: 27.34, lon: -82.53, prob34kt: 88, prob50kt: 66, prob64kt: 40 },
    { location: 'Orlando FL', lat: 28.54, lon: -81.38, prob34kt: 60, prob50kt: 30, prob64kt: 10 },
    { location: 'Jacksonville FL', lat: 30.33, lon: -81.66, prob34kt: 35, prob50kt: 12, prob64kt: 3 },
    { location: 'Gainesville FL', lat: 29.65, lon: -82.32, prob34kt: 72, prob50kt: 45, prob64kt: 20 },
    { location: 'Key West FL', lat: 24.56, lon: -81.78, prob34kt: 45, prob50kt: 18, prob64kt: 5 },
    { location: 'Mobile AL', lat: 30.69, lon: -88.04, prob34kt: 28, prob50kt: 10, prob64kt: 2 },
    { location: 'New Orleans LA', lat: 29.95, lon: -90.07, prob34kt: 15, prob50kt: 4, prob64kt: 1 },
  ];

  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14',
    issuedAt: new Date().toISOString(),
    validThrough: new Date(Date.now() + 120 * 3600_000).toISOString(),
    locations,
    text: generateWindProbText(storm, locations),
  };
}

function generateWindProbText(storm: ActiveCyclone, locations: WindProbabilityLocation[]): string {
  let text = `ZCZC MIAPWS${storm.basin === 'atlantic' ? 'AT' : 'EP'}${storm.id.slice(-1)}\n`;
  text += `${storm.name} Wind Speed Probabilities Number  14\n`;
  text += `NWS National Hurricane Center Miami FL       ${storm.id}\n`;
  text += `${new Date().toUTCString()}\n\n`;
  text += `${storm.name.toUpperCase()} WIND SPEED PROBABILITIES THROUGH 120 HOURS\n\n`;
  text += `PROBABILITIES FOR LOCATIONS ARE GIVEN AS TP(CP) WHERE\n`;
  text += `    TP  IS THE TOTAL PROBABILITY OF THE EVENT\n`;
  text += `    CP  IS THE CONDITIONAL PROBABILITY IF WINDS REACH THRESHOLD\n\n`;
  text += `LOCATION       34KT    50KT    64KT\n`;
  text += `─────────────────────────────────────\n`;
  for (const loc of locations) {
    const name = loc.location.padEnd(18);
    text += `${name} ${String(loc.prob34kt).padStart(3)}%    ${String(loc.prob50kt).padStart(3)}%    ${String(loc.prob64kt).padStart(3)}%\n`;
  }
  text += '\n$$\n';
  return text;
}

function getSampleStormSurge(stormId: string): StormSurgeData {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  const surgeZones: SurgeZone[] = [
    { area: 'Ochlockonee River to Chassahowitzka, including Apalachee Bay', minFt: 10, maxFt: 15, tidalAdjusted: true },
    { area: 'Chassahowitzka to Anclote River, including Tampa Bay', minFt: 8, maxFt: 12, tidalAdjusted: true },
    { area: 'Indian Pass to Ochlockonee River', minFt: 6, maxFt: 10, tidalAdjusted: true },
    { area: 'Anclote River to Middle of Longboat Key, including Tampa Bay', minFt: 5, maxFt: 8, tidalAdjusted: true },
    { area: 'Middle of Longboat Key to Bonita Beach, including Charlotte Harbor', minFt: 4, maxFt: 7, tidalAdjusted: true },
    { area: 'Suwannee River to Indian Pass', minFt: 3, maxFt: 5, tidalAdjusted: false },
    { area: 'Bonita Beach to Chokoloskee, including Estero Bay', minFt: 3, maxFt: 5, tidalAdjusted: false },
  ];

  return {
    stormId: storm.id,
    stormName: storm.name,
    issuedAt: new Date().toISOString(),
    watches: ['Englewood northward to Indian Pass including Tampa Bay'],
    warnings: ['Ochlockonee River southward to Flamingo including Tampa Bay and Charlotte Harbor'],
    peakSurgeFt: 15,
    surgeZones,
    inundationMapUrl: 'https://www.nhc.noaa.gov/storm_graphics/AT09/refresh/AL092024_PSURGE_INHAB+PSTM+0_5day+latest.png',
    text: generateSurgeText(storm, surgeZones),
  };
}

function generateSurgeText(storm: ActiveCyclone, zones: SurgeZone[]): string {
  let text = `STORM SURGE INFORMATION FOR ${storm.name.toUpperCase()}\n\n`;
  text += 'The combination of a dangerous storm surge and the tide will cause\n';
  text += 'normally dry areas near the coast to be flooded by rising waters moving\n';
  text += 'inland from the shoreline.  The water could reach the following heights\n';
  text += 'above ground somewhere in the indicated areas if the peak surge occurs\n';
  text += 'at the time of high tide...\n\n';
  for (const zone of zones) {
    text += `${zone.area}...\n  ${zone.minFt}-${zone.maxFt} ft\n\n`;
  }
  text += 'Surge-related flooding depends on the relative timing of the surge and\n';
  text += 'the tidal cycle, and can vary greatly over short distances.\n';
  return text;
}

function getSampleTrackCone(stormId: string): TrackForecastCone {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  const advisory = getSampleForecastAdvisory(stormId);
  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14',
    issuedAt: new Date().toISOString(),
    coneImageUrl: `https://www.nhc.noaa.gov/storm_graphics/${storm.id.substring(0, 4)}/${storm.id}_5day_cone_with_line_and_wind.png`,
    animationUrl: `https://www.nhc.noaa.gov/storm_graphics/${storm.id.substring(0, 4)}/${storm.id}_5day_cone_with_line_and_wind_sm2.gif`,
    forecastPositions: advisory.forecastPositions,
    watches: advisory.forecastPositions.length > 0 ? [
      { type: 'hurricane_watch', areas: 'Anclote River to Flamingo' },
    ] : [],
    warnings: advisory.forecastPositions.length > 0 ? [
      { type: 'hurricane_warning', areas: 'Mexico Beach to Bonita Beach including Tampa Bay' },
      { type: 'storm_surge_warning', areas: 'Ochlockonee River to Flamingo including Tampa Bay' },
    ] : [],
  };
}

function getSampleKeyMessages(stormId: string): KeyMessages {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  return {
    stormId: storm.id,
    stormName: storm.name,
    issuedAt: new Date().toISOString(),
    messages: [
      `${storm.name} is forecast to intensify before making landfall. Life-threatening storm surge of 10-15 feet is possible along portions of the Florida Big Bend coast.`,
      'Hurricane-force winds are expected in the warning area. Preparations should be rushed to completion.',
      `Heavy rainfall from ${storm.name} is expected to produce considerable and locally catastrophic flooding across portions of the southeastern United States through the weekend.`,
      'A few tornadoes are possible across parts of the Florida Peninsula and the Southeast through Friday.',
    ],
    graphicUrl: `https://www.nhc.noaa.gov/storm_graphics/${storm.id.substring(0, 4)}/${storm.id}_key_messages.png`,
  };
}

function getSampleAviationAdvisory(stormId: string): AviationAdvisory {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  return {
    stormId: storm.id,
    stormName: storm.name,
    advisoryNumber: '14',
    issuedAt: new Date().toISOString(),
    text: `FKNT2${storm.id.slice(-1)} KNHC DDHHMM\n` +
      `TROPICAL CYCLONE ADVISORY\n` +
      `NWS NATIONAL HURRICANE CENTER MIAMI FL       ${storm.id}\n${new Date().toUTCString()}\n\n` +
      `TC ADVISORY\n` +
      `DTG:                      ${new Date().toISOString()}\n` +
      `TCID:                     ${storm.id}\n` +
      `TCNAME:                   ${storm.name.toUpperCase()}\n` +
      `TCCLASS:                  ${storm.classification.toUpperCase().replace(/_/g, ' ')}\n` +
      `CENTER:                   N${storm.lat.toFixed(1)} W${Math.abs(storm.lon).toFixed(1)}\n` +
      `MAX SUSTAINED WINDS:      ${storm.maxWindKt} KT\n` +
      `ESTIMATED MIN PRESSURE:   ${storm.pressureMb} MB\n` +
      `MOVEMENT:                 ${storm.movementDir} ${storm.movementSpeedMph} MPH\n` +
      `\nREMARKS: NIL\n\nNEXT MSG: ${new Date(Date.now() + 6 * 3600_000).toUTCString()}\n`,
  };
}

function getSampleUpdates(stormId: string): TropicalCycloneUpdate[] {
  const storm = getSampleActiveCyclones().find(c => c.id === stormId) ?? getSampleActiveCyclones()[0];
  return [
    {
      stormId: storm.id,
      stormName: storm.name,
      issuedAt: new Date(Date.now() - 2 * 3600_000).toISOString(),
      text: `...${storm.name.toUpperCase()} CONTINUES TO STRENGTHEN...\n\n` +
        `Satellite estimates indicate ${storm.name} has continued to intensify\n` +
        `since the last advisory. The maximum sustained winds are now\n` +
        `estimated to be near ${storm.maxWindMph} mph (${storm.maxWindKt} kt). The eye has\n` +
        `become more well-defined on visible satellite imagery.\n`,
    },
  ];
}

function getSampleMarineForecasts(zone: MarineZone): MarineForecast[] {
  if (zone === 'offshore') {
    return [
      {
        zone: 'offshore',
        region: 'Gulf of Mexico',
        issuedAt: new Date().toISOString(),
        synopsis: 'A hurricane is producing dangerous conditions across the eastern Gulf of Mexico. Hurricane-force winds extend outward up to 45 nautical miles from the center and tropical storm force winds extend outward up to 230 nautical miles.',
        forecast: 'HURRICANE WARNING IN EFFECT for waters from Ochlockonee River to Bonita Beach FL from 60 NM out to the boundary of the EEZ. Seas 20 to 35 feet near the center. Scattered thunderstorms with heavy rain.',
        text: 'OFFSHORE WATERS FORECAST\nNWS NATIONAL HURRICANE CENTER MIAMI FL\n\nGULF OF MEXICO\n\nSYNOPSIS: A hurricane centered near 24.8N 86.3W is moving NNE at 14 mph. The system is expected to strengthen before making landfall along the Florida Big Bend coast.\n\n.FORECAST...\nHurricane conditions within 60 NM of the center. Seas 20-35 ft near center, diminishing to 8-12 ft well away from center. Heavy rain and thunderstorms.',
      },
      {
        zone: 'offshore',
        region: 'Western Atlantic',
        issuedAt: new Date().toISOString(),
        synopsis: 'A tropical storm is moving northeastward across the central Atlantic well away from land areas.',
        forecast: 'Tropical storm force winds within 120 NM of the center of Tropical Storm Isaac. Seas 10 to 18 feet near the center.',
        text: 'OFFSHORE WATERS FORECAST\nNWS NATIONAL HURRICANE CENTER MIAMI FL\n\nWESTERN ATLANTIC\n\nSYNOPSIS: Tropical Storm Isaac is centered near 37.2N 40.8W moving NE at 18 mph. No watches or warnings are in effect for land areas.\n\n.FORECAST...\nTropical storm conditions within 120 NM of center. Seas 10-18 ft near center.',
      },
    ];
  }
  if (zone === 'high_seas') {
    return [
      {
        zone: 'high_seas',
        region: 'North Atlantic High Seas',
        issuedAt: new Date().toISOString(),
        synopsis: 'Hurricane Helene and Tropical Storm Isaac are producing hazardous marine conditions across portions of the Gulf of Mexico and central Atlantic respectively.',
        forecast: 'DANGEROUS CONDITIONS: Hurricane force winds and seas to 35 ft near Hurricane Helene in the eastern Gulf of Mexico. Tropical storm force winds and seas to 18 ft near Tropical Storm Isaac in the central North Atlantic.',
        text: 'HIGH SEAS FORECAST\nNWS NATIONAL HURRICANE CENTER MIAMI FL\n\nNORTH ATLANTIC\n\nMETAREA IV\n\nWARNING: Hurricane Helene - hurricane force winds within 45 NM of center near 24.8N 86.3W.\nWARNING: Tropical Storm Isaac - storm force winds within 120 NM of center near 37.2N 40.8W.\n\nSeas 25-35 ft within 60 NM of Helene, 15-18 ft within 90 NM of Isaac.',
      },
    ];
  }
  // coastal
  return [
    {
      zone: 'coastal',
      region: 'Florida Gulf Coast',
      issuedAt: new Date().toISOString(),
      synopsis: 'Hurricane conditions expected along portions of the Florida Big Bend and Gulf Coast as Hurricane Helene approaches.',
      forecast: 'TONIGHT: Hurricane force winds. Seas 15 to 25 feet. Storm surge 10 to 15 feet above normally dry ground. Dangerous rip currents.\nTOMORROW: Tropical storm to hurricane force winds diminishing. Seas 8 to 15 feet. Continued storm surge flooding.',
      text: 'COASTAL WATERS FORECAST\nNWS WEATHER FORECAST OFFICE TALLAHASSEE FL\n\nFLORIDA GULF COAST\n\n.SYNOPSIS...\nHurricane Helene approaching from the south-southeast with landfall expected overnight. Hurricane warning in effect for all coastal waters from Mexico Beach to Bonita Beach.\n\n.TONIGHT...\nHurricane force winds 80 to 110 mph. Seas 15 to 25 ft. Storm surge 10-15 ft. Heavy rain.\n\n.FRIDAY...\nWinds diminishing to tropical storm force then gradually subsiding. Seas 8-15 ft. Continued elevated water levels.',
    },
  ];
}

function getSampleTCReports(year?: number): TropicalCycloneReport[] {
  const reportYear = year ?? 2024;
  return [
    {
      stormId: 'AL012024',
      stormName: 'Alberto',
      year: reportYear,
      basin: 'atlantic',
      classification: 'tropical_storm',
      formationDate: '2024-06-19',
      dissipationDate: '2024-06-20',
      peakWindKt: 50,
      minPressureMb: 995,
      deaths: 4,
      damageUsd: '$50 million',
      summary: 'Alberto was a short-lived tropical storm that brought heavy rainfall and flooding to northeastern Mexico and southern Texas. Alberto formed in the southwestern Gulf of Mexico and made landfall along the coast of Tamaulipas, Mexico.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL012024_Alberto.pdf',
    },
    {
      stormId: 'AL022024',
      stormName: 'Beryl',
      year: reportYear,
      basin: 'atlantic',
      classification: 'major_hurricane',
      formationDate: '2024-06-28',
      dissipationDate: '2024-07-09',
      peakWindKt: 140,
      minPressureMb: 934,
      deaths: 74,
      damageUsd: '$6.5 billion',
      summary: 'Beryl was the earliest Category 5 hurricane on record in the Atlantic basin. It caused catastrophic damage in the Windward Islands, particularly Grenada and St. Vincent, before crossing the Caribbean and eventually making landfall in Texas.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL022024_Beryl.pdf',
    },
    {
      stormId: 'AL042024',
      stormName: 'Debby',
      year: reportYear,
      basin: 'atlantic',
      classification: 'hurricane',
      formationDate: '2024-08-03',
      dissipationDate: '2024-08-09',
      peakWindKt: 70,
      minPressureMb: 981,
      deaths: 8,
      damageUsd: '$2 billion',
      summary: 'Hurricane Debby made landfall in the Florida Big Bend region as a Category 1 hurricane before stalling and producing extreme rainfall and flooding across the southeastern United States.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL042024_Debby.pdf',
    },
    {
      stormId: 'AL062024',
      stormName: 'Francine',
      year: reportYear,
      basin: 'atlantic',
      classification: 'hurricane',
      formationDate: '2024-09-09',
      dissipationDate: '2024-09-12',
      peakWindKt: 85,
      minPressureMb: 968,
      deaths: 1,
      damageUsd: '$400 million',
      summary: 'Hurricane Francine made landfall in southern Louisiana as a Category 2 hurricane, bringing storm surge and strong winds to coastal communities.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL062024_Francine.pdf',
    },
    {
      stormId: 'AL092024',
      stormName: 'Helene',
      year: reportYear,
      basin: 'atlantic',
      classification: 'major_hurricane',
      formationDate: '2024-09-24',
      dissipationDate: '2024-09-29',
      peakWindKt: 120,
      minPressureMb: 938,
      deaths: 232,
      damageUsd: '$55 billion',
      summary: 'Hurricane Helene was a large and powerful Category 4 hurricane that made landfall in the Florida Big Bend region. Helene produced catastrophic inland flooding across the southern Appalachians, particularly in western North Carolina, and deadly storm surge along the Florida Gulf Coast.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL092024_Helene.pdf',
    },
    {
      stormId: 'AL142024',
      stormName: 'Milton',
      year: reportYear,
      basin: 'atlantic',
      classification: 'major_hurricane',
      formationDate: '2024-10-05',
      dissipationDate: '2024-10-11',
      peakWindKt: 155,
      minPressureMb: 897,
      deaths: 36,
      damageUsd: '$35 billion',
      summary: 'Hurricane Milton underwent extraordinary rapid intensification in the Gulf of Mexico, becoming a Category 5 hurricane. It made landfall near Siesta Key, Florida as a Category 3 hurricane, producing devastating storm surge, destructive winds, and a tornado outbreak.',
      reportUrl: 'https://www.nhc.noaa.gov/data/tcr/AL142024_Milton.pdf',
    },
  ];
}
