/* ------------------------------------------------------------------ */
/*  RainViewer API client                                             */
/*  Free radar & satellite tiles – https://www.rainviewer.com/api     */
/* ------------------------------------------------------------------ */

import type { RainViewerData, RainViewerFrame } from '../types/weather';

const API_URL = 'https://api.rainviewer.com/public/weather-maps.json';

let cachedData: RainViewerData | null = null;
let cacheTime = 0;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export async function fetchRainViewerData(): Promise<RainViewerData> {
  const now = Date.now();
  if (cachedData && now - cacheTime < CACHE_TTL) {
    return cachedData;
  }
  const res = await fetch(API_URL);
  if (!res.ok) throw new Error(`RainViewer error: ${res.status}`);
  cachedData = (await res.json()) as RainViewerData;
  cacheTime = now;
  return cachedData;
}

/** Build a Leaflet-compatible tile URL for a given frame. */
export function radarTileUrl(frame: RainViewerFrame, colorScheme = 4, smooth = 1, snow = 1): string {
  // https://tilecache.rainviewer.com{path}/{size}/{z}/{x}/{y}/{color}/{smooth}_{snow}.png
  return `https://tilecache.rainviewer.com${frame.path}/256/{z}/{x}/{y}/${colorScheme}/${smooth}_${snow}.png`;
}

export function satelliteTileUrl(frame: RainViewerFrame, colorScheme = 0, smooth = 0, snow = 0): string {
  return `https://tilecache.rainviewer.com${frame.path}/256/{z}/{x}/{y}/${colorScheme}/${smooth}_${snow}.png`;
}

/** Get all available radar frames (past + nowcast). */
export async function getRadarFrames(): Promise<RainViewerFrame[]> {
  const data = await fetchRainViewerData();
  return [...data.radar.past, ...data.radar.nowcast];
}

/** Get all available satellite frames. */
export async function getSatelliteFrames(): Promise<RainViewerFrame[]> {
  const data = await fetchRainViewerData();
  return data.satellite.infrared;
}

/** Format a unix timestamp to a locale time string. */
export function formatFrameTime(unixTs: number): string {
  const d = new Date(unixTs * 1000);
  return d.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}
