/* ------------------------------------------------------------------ */
/*  Cascade Explorer                                                   */
/* ------------------------------------------------------------------ */
/*  Three linked levels — each one answers exactly one question:       */
/*    World      → "where should I focus right now?"                   */
/*    Datacenter → "which servers in this building need help?"         */
/*    Server     → "what is happening on this machine?"                */
/*                                                                     */
/*  The shape of the visualization changes per level: world map →      */
/*  rack/slot grid → time series. That shape change is the point —     */
/*  it tells the user "you have arrived somewhere new." Map and table  */
/*  are bidirectionally synced; clicking either drills.                */
/* ------------------------------------------------------------------ */

import { useMemo, useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import {
  HEALTH_COLOR,
  type Datacenter,
  type Health,
  type Server,
  loadDatacenter,
  loadServer,
  loadWorld,
} from './data';
import './CascadeExplorer.css';

type Level =
  | { kind: 'world' }
  | { kind: 'datacenter'; dcId: string }
  | { kind: 'server'; dcId: string; serverId: string };

const HEALTH_LIGHT: Record<Health, string> = {
  healthy: '🟢',
  warning: '🟠',
  critical: '🔴',
};

export default function CascadeExplorer() {
  const [level, setLevel] = useState<Level>({ kind: 'world' });

  // Browser back/forward integration — the trail you walked is real history.
  useEffect(() => {
    const onPop = (ev: PopStateEvent) => {
      if (ev.state && typeof ev.state === 'object' && 'cascade' in ev.state) {
        setLevel(ev.state.cascade as Level);
      } else {
        setLevel({ kind: 'world' });
      }
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const go = (next: Level) => {
    window.history.pushState({ cascade: next }, '');
    setLevel(next);
  };

  return (
    <div className="cascade">
      <Breadcrumb level={level} onGo={go} />
      {level.kind === 'world' && <WorldLevel onDrill={(dcId) => go({ kind: 'datacenter', dcId })} />}
      {level.kind === 'datacenter' && (
        <DatacenterLevel
          dcId={level.dcId}
          onDrill={(serverId) => go({ kind: 'server', dcId: level.dcId, serverId })}
        />
      )}
      {level.kind === 'server' && (
        <ServerLevel dcId={level.dcId} serverId={level.serverId} />
      )}
    </div>
  );
}

/* ── breadcrumb ───────────────────────────────────────────────────── */

function Breadcrumb({ level, onGo }: { level: Level; onGo: (l: Level) => void }) {
  const crumbs: { label: string; level: Level | null }[] = [
    { label: '🌍  World', level: { kind: 'world' } },
  ];
  if (level.kind !== 'world') {
    crumbs.push({
      label: `🏢  ${level.dcId}`,
      level: { kind: 'datacenter', dcId: level.dcId },
    });
  }
  if (level.kind === 'server') {
    crumbs.push({ label: `🖥  ${level.serverId}`, level: null });
  }

  return (
    <div className="cascade__crumbs">
      {crumbs.map((c, i) => (
        <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <button
            type="button"
            className={`cascade__crumb ${i === crumbs.length - 1 ? 'cascade__crumb--active' : ''}`}
            onClick={() => c.level && onGo(c.level)}
            disabled={i === crumbs.length - 1}
          >
            {c.label}
          </button>
          {i < crumbs.length - 1 && <span className="cascade__crumb-sep">›</span>}
        </span>
      ))}
    </div>
  );
}

/* ── level 1: world ───────────────────────────────────────────────── */

function WorldLevel({ onDrill }: { onDrill: (dcId: string) => void }) {
  const dcs = useMemo(() => {
    return [...loadWorld()].sort((a, b) => a.healthScore - b.healthScore);
  }, []);
  const [hoverId, setHoverId] = useState<string | null>(null);

  return (
    <>
      <div className="cascade__header">
        <h2 className="cascade__title">Global fleet</h2>
        <p className="cascade__question">Where should I focus my attention right now?</p>
      </div>

      <div className="cascade__body">
        <div className="cascade__viz">
          <div className="cascade__map">
            <MapContainer
              center={[20, 0]}
              zoom={2}
              minZoom={2}
              worldCopyJump
              scrollWheelZoom
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
                attribution='© OSM © CARTO'
              />
              {dcs.map((d) => (
                <CircleMarker
                  key={d.id}
                  center={[d.lat, d.lon]}
                  radius={8 + Math.sqrt(d.servers) * 1.4}
                  pathOptions={{
                    color: '#000',
                    weight: 1,
                    fillColor: scoreColor(d.healthScore),
                    fillOpacity: hoverId === d.id ? 1.0 : 0.85,
                  }}
                  eventHandlers={{
                    mouseover: () => setHoverId(d.id),
                    mouseout: () => setHoverId(null),
                    click: () => onDrill(d.id),
                  }}
                >
                  <Tooltip direction="top" offset={[0, -4]} opacity={0.95}>
                    <strong>{d.name}</strong>
                    <br />
                    {d.city}, {d.country}
                    <br />
                    {d.servers} servers · health {d.healthScore}
                  </Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>
          <div className="cascade__viz-caption">
            Bubble size = server count. Color = health score (red worst). Click any marker to enter the building.
          </div>
        </div>

        <div className="cascade__panel">
          <div className="cascade__panel-section" style={{ flex: '1 1 auto' }}>
            <div className="cascade__panel-title">Worst-first · click to drill</div>
            <div className="cascade__table-wrap">
              <table className="cascade__table">
                <thead>
                  <tr>
                    <th>Data center</th>
                    <th>Servers</th>
                    <th>Crit</th>
                    <th>Warn</th>
                    <th>Health</th>
                  </tr>
                </thead>
                <tbody>
                  {dcs.map((d) => (
                    <tr
                      key={d.id}
                      className={hoverId === d.id ? 'is-active' : ''}
                      onMouseEnter={() => setHoverId(d.id)}
                      onMouseLeave={() => setHoverId(null)}
                      onClick={() => onDrill(d.id)}
                    >
                      <td>
                        {HEALTH_LIGHT[summarize(d)]}  {d.name}
                      </td>
                      <td>{d.servers}</td>
                      <td style={{ color: d.critical > 0 ? '#e74c3c' : '#5b6473' }}>{d.critical}</td>
                      <td style={{ color: d.warning > 0 ? '#f39c12' : '#5b6473' }}>{d.warning}</td>
                      <td>{d.healthScore}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function summarize(d: Datacenter): Health {
  if (d.critical > 0) return 'critical';
  if (d.warning > 0) return 'warning';
  return 'healthy';
}

function scoreColor(s: number): string {
  // 0 → red, 50 → orange, 100 → green
  if (s >= 80) return HEALTH_COLOR.healthy;
  if (s >= 65) return HEALTH_COLOR.warning;
  return HEALTH_COLOR.critical;
}

/* ── level 2: data center ─────────────────────────────────────────── */

function DatacenterLevel({
  dcId,
  onDrill,
}: {
  dcId: string;
  onDrill: (serverId: string) => void;
}) {
  const meta = useMemo(() => loadWorld().find((d) => d.id === dcId), [dcId]);
  const servers = useMemo(() => loadDatacenter(dcId), [dcId]);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  if (!meta) return <div className="cascade__empty">Unknown data center.</div>;

  const nRacks = Math.max(...servers.map((s) => s.rack)) + 1;
  // grid[slot][rack] = server | null
  const grid: (Server | null)[][] = Array.from({ length: 8 }, () =>
    Array(nRacks).fill(null),
  );
  for (const s of servers) grid[s.slot][s.rack] = s;

  const filtered = search
    ? servers.filter((s) => s.id.toLowerCase().includes(search.toLowerCase()))
    : servers;

  return (
    <>
      <div className="cascade__header">
        <h2 className="cascade__title">
          {meta.name} — {meta.city}, {meta.country}
        </h2>
        <p className="cascade__question">
          Which servers in this building need a human in the loop?
        </p>
      </div>

      <div className="cascade__kpis">
        <Kpi label="Servers" value={meta.servers} />
        <Kpi label="Critical" value={meta.critical} tone={meta.critical > 0 ? 'crit' : undefined} />
        <Kpi label="Warning" value={meta.warning} tone={meta.warning > 0 ? 'warn' : undefined} />
        <Kpi label="Avg CPU" value={`${meta.avgCpu}%`} />
      </div>

      <div className="cascade__body">
        <div className="cascade__viz">
          <div
            className="cascade__racks"
            style={{
              gridTemplateColumns: undefined,
            }}
          >
            {Array.from({ length: 8 }).map((_, slot) => (
              <div
                key={slot}
                className="cascade__rack-row"
                style={{
                  gridTemplateColumns: `28px repeat(${nRacks}, minmax(20px, 1fr))`,
                }}
              >
                <div className="cascade__rack-label">U{slot + 1}</div>
                {Array.from({ length: nRacks }).map((__, rack) => {
                  const s = grid[slot][rack];
                  if (!s) {
                    return <div key={rack} className="cascade__cell" />;
                  }
                  return (
                    <div
                      key={rack}
                      className={`cascade__cell cascade__cell--filled ${
                        hoverId === s.id ? 'is-active' : ''
                      }`}
                      style={{ background: HEALTH_COLOR[s.status] }}
                      title={`${s.id}\n${s.role}\nCPU ${s.cpuPct}%  MEM ${s.memPct}%`}
                      onMouseEnter={() => setHoverId(s.id)}
                      onMouseLeave={() => setHoverId(null)}
                      onClick={() => onDrill(s.id)}
                    />
                  );
                })}
              </div>
            ))}
          </div>
          <div className="cascade__viz-caption">
            Physical layout — each cell is one rack-mounted server, rows are rack units. Click any cell to inspect.
          </div>
        </div>

        <div className="cascade__panel">
          <div className="cascade__panel-section" style={{ flex: '1 1 auto', minHeight: 0 }}>
            <div className="cascade__panel-title">Worst-first</div>
            <div className="cascade__table-wrap">
              <table className="cascade__table">
                <thead>
                  <tr>
                    <th>Server</th>
                    <th>Role</th>
                    <th>CPU</th>
                    <th>MEM</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((s) => (
                    <tr
                      key={s.id}
                      className={hoverId === s.id ? 'is-active' : ''}
                      onMouseEnter={() => setHoverId(s.id)}
                      onMouseLeave={() => setHoverId(null)}
                      onClick={() => onDrill(s.id)}
                    >
                      <td>
                        <span
                          className="cascade__status-dot"
                          style={{ background: HEALTH_COLOR[s.status] }}
                        />
                        {s.id.split('-').slice(-2).join('-')}
                      </td>
                      <td>{s.role}</td>
                      <td>{s.cpuPct}%</td>
                      <td>{s.memPct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="cascade__search">
              <input
                placeholder="search server id (e.g. web-007)"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Kpi({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone?: 'crit' | 'warn';
}) {
  return (
    <div className="cascade__kpi">
      <span className="cascade__kpi-label">{label}</span>
      <span
        className={`cascade__kpi-value ${
          tone ? `cascade__kpi-value--${tone}` : ''
        }`}
      >
        {value}
      </span>
    </div>
  );
}

/* ── level 3: server ──────────────────────────────────────────────── */

function ServerLevel({ dcId, serverId }: { dcId: string; serverId: string }) {
  const server = useMemo(
    () => loadDatacenter(dcId).find((s) => s.id === serverId),
    [dcId, serverId],
  );
  const bundle = useMemo(() => loadServer(serverId), [serverId]);

  if (!server) return <div className="cascade__empty">Unknown server.</div>;

  return (
    <>
      <div className="cascade__header">
        <h2 className="cascade__title">{server.id}</h2>
        <p className="cascade__question">
          What is happening on this machine, and is it getting worse?
        </p>
      </div>

      <div className="cascade__kpis">
        <Kpi label="Role" value={server.role} />
        <Kpi
          label="CPU now"
          value={`${server.cpuPct}%`}
          tone={server.cpuPct >= 90 ? 'crit' : server.cpuPct >= 70 ? 'warn' : undefined}
        />
        <Kpi
          label="Memory now"
          value={`${server.memPct}%`}
          tone={server.memPct >= 90 ? 'crit' : server.memPct >= 70 ? 'warn' : undefined}
        />
        <Kpi label="Uptime" value={`${server.uptimeDays} d`} />
      </div>

      <div className="cascade__body" style={{ gridTemplateColumns: '1.6fr 1fr' }}>
        <div className="cascade__viz">
          <div className="cascade__sparks">
            <Sparkline
              label="CPU %"
              series={bundle.series.map((p) => p.cpu)}
              times={bundle.series.map((p) => p.t)}
              latest={`${server.cpuPct}%`}
              yMin={0}
              yMax={100}
              warn={70}
              crit={90}
              color="#4ea1ff"
            />
            <Sparkline
              label="Memory %"
              series={bundle.series.map((p) => p.mem)}
              times={bundle.series.map((p) => p.t)}
              latest={`${server.memPct}%`}
              yMin={0}
              yMax={100}
              warn={70}
              crit={90}
              color="#a78bfa"
            />
            <Sparkline
              label="Network Mbps"
              series={bundle.series.map((p) => p.net)}
              times={bundle.series.map((p) => p.t)}
              latest={`${server.netMbps} Mbps`}
              yMin={0}
              yMax={1000}
              color="#34d399"
            />
          </div>
          <div className="cascade__viz-caption">
            24h history at 5-minute resolution. Dashed lines mark warn/critical thresholds.
          </div>
        </div>

        <div className="cascade__panel">
          <div className="cascade__panel-section" style={{ flex: '1 1 auto', minHeight: 0 }}>
            <div className="cascade__panel-title">Recent alerts (24h)</div>
            <div className="cascade__alerts">
              {bundle.alerts.length === 0 && (
                <div className="cascade__empty">No alerts in the last 24 hours.</div>
              )}
              {bundle.alerts.map((a, i) => (
                <div key={i} className={`cascade__alert cascade__alert--${a.severity}`}>
                  <div className="cascade__alert-time">
                    {new Date(a.t).toISOString().slice(11, 16)}
                  </div>
                  <div className="cascade__alert-msg">{a.message}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── sparkline ────────────────────────────────────────────────────── */

interface SparklineProps {
  label: string;
  latest: string;
  series: number[];
  times: number[];
  yMin: number;
  yMax: number;
  warn?: number;
  crit?: number;
  color: string;
}

function Sparkline({ label, latest, series, yMin, yMax, warn, crit, color }: SparklineProps) {
  const w = 600;
  const h = 70;
  const padX = 2;
  const padY = 4;
  const innerW = w - padX * 2;
  const innerH = h - padY * 2;

  const path = useMemo(() => {
    if (series.length === 0) return '';
    return series
      .map((v, i) => {
        const x = padX + (i / (series.length - 1)) * innerW;
        const y =
          padY +
          (1 - (v - yMin) / (yMax - yMin || 1)) * innerH;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [series, yMin, yMax, innerW, innerH]);

  const yLine = (v: number) => padY + (1 - (v - yMin) / (yMax - yMin || 1)) * innerH;

  return (
    <div className="cascade__spark">
      <div className="cascade__spark-head">
        <span>{label}</span>
        <span className="cascade__spark-value">{latest}</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {warn !== undefined && (
          <line
            x1={0}
            x2={w}
            y1={yLine(warn)}
            y2={yLine(warn)}
            stroke="#f39c12"
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.5}
          />
        )}
        {crit !== undefined && (
          <line
            x1={0}
            x2={w}
            y1={yLine(crit)}
            y2={yLine(crit)}
            stroke="#e74c3c"
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.5}
          />
        )}
        <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
      </svg>
    </div>
  );
}
