/* ------------------------------------------------------------------ */
/*  Cascade Explorer — synthetic data                                  */
/* ------------------------------------------------------------------ */
/*  Three levels:                                                      */
/*    1. world      → data centers (geocoded)                          */
/*    2. datacenter → servers (rack/slot, NOT geographic)              */
/*    3. server     → 24h time series + recent alerts                  */
/*                                                                     */
/*  All data is deterministic for a given (seed, key) so the same      */
/*  click always lands on the same world. Replace these functions      */
/*  with real API calls — the return shapes are the contract.          */
/* ------------------------------------------------------------------ */

const SEED = 7;

export type Health = 'healthy' | 'warning' | 'critical';

export const HEALTH_COLOR: Record<Health, string> = {
  healthy: '#2ecc71',
  warning: '#f39c12',
  critical: '#e74c3c',
};

export interface Datacenter {
  id: string;
  name: string;
  city: string;
  country: string;
  region: string;
  lat: number;
  lon: number;
  servers: number;
  critical: number;
  warning: number;
  healthy: number;
  avgCpu: number;
  avgMem: number;
  healthScore: number;
}

export interface Server {
  id: string;
  dcId: string;
  role: 'web' | 'db' | 'cache' | 'batch' | 'ml-train';
  rack: number;
  slot: number;
  cpuPct: number;
  memPct: number;
  netMbps: number;
  uptimeDays: number;
  status: Health;
}

export interface SeriesPoint {
  t: number; // unix ms
  cpu: number;
  mem: number;
  net: number;
}

export interface Alert {
  t: number;
  severity: 'info' | 'warn' | 'crit';
  message: string;
}

interface DCSeed {
  id: string;
  name: string;
  city: string;
  country: string;
  region: string;
  lat: number;
  lon: number;
}

const DC_SEEDS: DCSeed[] = [
  { id: 'dc-iad', name: 'US-East (IAD)',     city: 'Ashburn',   country: 'USA',       region: 'us-east',     lat: 39.04,  lon: -77.49  },
  { id: 'dc-sjc', name: 'US-West (SJC)',     city: 'San Jose',  country: 'USA',       region: 'us-west',     lat: 37.34,  lon: -121.89 },
  { id: 'dc-dfw', name: 'US-Central (DFW)',  city: 'Dallas',    country: 'USA',       region: 'us-central',  lat: 32.90,  lon: -97.04  },
  { id: 'dc-fra', name: 'EU-Central (FRA)',  city: 'Frankfurt', country: 'Germany',   region: 'eu-central',  lat: 50.11,  lon: 8.68    },
  { id: 'dc-lhr', name: 'EU-West (LHR)',     city: 'London',    country: 'UK',        region: 'eu-west',     lat: 51.47,  lon: -0.45   },
  { id: 'dc-sin', name: 'AP-Southeast (SIN)', city: 'Singapore', country: 'Singapore', region: 'ap-southeast', lat: 1.36,  lon: 103.99 },
  { id: 'dc-nrt', name: 'AP-Northeast (NRT)', city: 'Tokyo',    country: 'Japan',     region: 'ap-northeast', lat: 35.55, lon: 139.78 },
  { id: 'dc-syd', name: 'AP-South (SYD)',    city: 'Sydney',    country: 'Australia', region: 'ap-south',    lat: -33.94, lon: 151.18  },
];

const ROLES: Server['role'][] = ['web', 'db', 'cache', 'batch', 'ml-train'];
const ROLE_BASELINE: Record<Server['role'], { cpu: number; mem: number }> = {
  web:        { cpu: 45, mem: 50 },
  db:         { cpu: 60, mem: 75 },
  cache:      { cpu: 30, mem: 70 },
  batch:      { cpu: 75, mem: 55 },
  'ml-train': { cpu: 85, mem: 80 },
};

// ── deterministic RNG ──────────────────────────────────────────────
//   xmur3 -> sfc32, both from public-domain JS implementations.
//   We re-seed per (key) so the same key always gives the same stream.

function xmur3(str: string): () => number {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0;
  };
}

function sfc32(a: number, b: number, c: number, d: number): () => number {
  return () => {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = ((a + b) | 0) + d | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

function rngFor(...parts: (string | number)[]): () => number {
  const key = `${SEED}|${parts.join('|')}`;
  const seed = xmur3(key);
  return sfc32(seed(), seed(), seed(), seed());
}

function gauss(rng: () => number, mean = 0, sd = 1): number {
  // Box–Muller
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

function classify(cpu: number, mem: number): Health {
  const worst = Math.max(cpu, mem);
  if (worst >= 90) return 'critical';
  if (worst >= 70) return 'warning';
  return 'healthy';
}

// ── memoized loaders ───────────────────────────────────────────────

const dcCache = new Map<string, Server[]>();
const serverCache = new Map<string, { series: SeriesPoint[]; alerts: Alert[] }>();
let worldCache: Datacenter[] | null = null;

export function loadDatacenter(dcId: string): Server[] {
  const cached = dcCache.get(dcId);
  if (cached) return cached;

  const rng = rngFor('dc', dcId);
  const nServers = 28 + Math.floor(rng() * 36);
  const nRacks = Math.max(4, Math.floor(nServers / 8));
  const roleWeights = [0.35, 0.55, 0.70, 0.90, 1.0];

  const servers: Server[] = [];
  for (let i = 0; i < nServers; i++) {
    const r = rng();
    const roleIdx = roleWeights.findIndex((w) => r < w);
    const role = ROLES[roleIdx === -1 ? ROLES.length - 1 : roleIdx];
    const baseline = ROLE_BASELINE[role];
    const cpu = clamp(baseline.cpu + gauss(rng, 0, 12), 2, 99.5);
    const mem = clamp(baseline.mem + gauss(rng, 0, 10), 5, 99.5);
    servers.push({
      id: `${dcId}-${role}-${String(i).padStart(3, '0')}`,
      dcId,
      role,
      rack: Math.floor(rng() * nRacks),
      slot: Math.floor(rng() * 8),
      cpuPct: round1(cpu),
      memPct: round1(mem),
      netMbps: Math.round(50 + rng() * 900),
      uptimeDays: 1 + Math.floor(rng() * 720),
      status: classify(cpu, mem),
    });
  }

  // Worst-first ordering — same principle as the table.
  const rank: Record<Health, number> = { critical: 0, warning: 1, healthy: 2 };
  servers.sort((a, b) => rank[a.status] - rank[b.status] || b.cpuPct - a.cpuPct);
  dcCache.set(dcId, servers);
  return servers;
}

export function loadWorld(): Datacenter[] {
  if (worldCache) return worldCache;

  worldCache = DC_SEEDS.map((seed) => {
    const servers = loadDatacenter(seed.id);
    const n = servers.length;
    const critical = servers.filter((s) => s.status === 'critical').length;
    const warning = servers.filter((s) => s.status === 'warning').length;
    const healthy = n - critical - warning;
    const avgCpu = servers.reduce((a, s) => a + s.cpuPct, 0) / n;
    const avgMem = servers.reduce((a, s) => a + s.memPct, 0) / n;
    const healthScore = (100 * (healthy + 0.5 * warning)) / n;
    return {
      ...seed,
      servers: n,
      critical,
      warning,
      healthy,
      avgCpu: round1(avgCpu),
      avgMem: round1(avgMem),
      healthScore: round1(healthScore),
    };
  });
  return worldCache;
}

export function loadServer(serverId: string) {
  const cached = serverCache.get(serverId);
  if (cached) return cached;

  const rng = rngFor('server', serverId);
  const now = Date.UTC(2026, 3, 29, 12, 0); // 2026-04-29 12:00 UTC
  const points = 288; // 24h × 5min
  const stepMs = 5 * 60 * 1000;

  const walk = (start: number, vol: number, lo: number, hi: number): number[] => {
    const out = new Array<number>(points);
    out[0] = start;
    for (let k = 1; k < points; k++) {
      const x = 0.92 * out[k - 1] + 0.08 * start + gauss(rng, 0, vol);
      out[k] = clamp(x, lo, hi);
    }
    return out;
  };

  const cpu = walk(20 + rng() * 60, 4.0, 1, 100);
  const mem = walk(30 + rng() * 55, 2.5, 5, 100);
  const net = walk(100 + rng() * 600, 30.0, 10, 1000);

  const series: SeriesPoint[] = [];
  for (let i = 0; i < points; i++) {
    series.push({
      t: now - (points - 1 - i) * stepMs,
      cpu: cpu[i],
      mem: mem[i],
      net: net[i],
    });
  }

  const alertPool: { severity: Alert['severity']; message: string }[] = [
    { severity: 'warn', message: 'CPU > 80% sustained 5m' },
    { severity: 'warn', message: 'memory pressure' },
    { severity: 'info', message: 'kernel updated, reboot pending' },
    { severity: 'crit', message: 'disk I/O wait spike' },
    { severity: 'info', message: 'package security patches available' },
    { severity: 'warn', message: 'swap usage rising' },
    { severity: 'crit', message: 'OOM killer invoked' },
  ];
  const nAlerts = 2 + Math.floor(rng() * 5);
  const indices = new Set<number>();
  while (indices.size < nAlerts) indices.add(Math.floor(rng() * alertPool.length));
  const alerts: Alert[] = [...indices]
    .map((i) => ({
      t: now - Math.floor(rng() * 24 * 60) * 60 * 1000,
      ...alertPool[i],
    }))
    .sort((a, b) => b.t - a.t);

  const bundle = { series, alerts };
  serverCache.set(serverId, bundle);
  return bundle;
}

function round1(x: number): number {
  return Math.round(x * 10) / 10;
}
