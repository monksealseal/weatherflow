import { useEffect, useRef, useState, useCallback } from 'react';
import type { LatLng, PointForecast, WeatherModel, SurfaceVariable } from '../types/weather';
import { MODELS, SURFACE_VARIABLES } from '../types/weather';
import { fetchMultiModelForecast } from '../api/openMeteo';

interface ModelComparisonProps {
  location: LatLng;
  onClose: () => void;
}

const COMPARE_MODELS: WeatherModel[] = ['gfs', 'ecmwf', 'icon', 'gem'];
const COMPARE_VAR: SurfaceVariable = 'temperature_2m';
const MIN_HEIGHT = 250;
const DEFAULT_HEIGHT = 360;
const MAX_HEIGHT_RATIO = 0.85; // max 85% of viewport

export default function ModelComparison({ location, onClose }: ModelComparisonProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [forecasts, setForecasts] = useState<PointForecast[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedVar, setSelectedVar] = useState<SurfaceVariable>(COMPARE_VAR);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [panelHeight, setPanelHeight] = useState(DEFAULT_HEIGHT);
  const [expanded, setExpanded] = useState(false);

  const toggleExpand = useCallback(() => {
    if (expanded) {
      setPanelHeight(DEFAULT_HEIGHT);
      setExpanded(false);
    } else {
      setPanelHeight(Math.floor(window.innerHeight * MAX_HEIGHT_RATIO));
      setExpanded(true);
    }
  }, [expanded]);

  // Drag-to-resize from the top edge
  const handleDragStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    const startY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const startHeight = panelRef.current?.getBoundingClientRect().height ?? panelHeight;

    const onMove = (ev: MouseEvent | TouchEvent) => {
      const currentY = 'touches' in ev ? ev.touches[0].clientY : ev.clientY;
      const delta = startY - currentY;
      const maxH = Math.floor(window.innerHeight * MAX_HEIGHT_RATIO);
      const newHeight = Math.max(MIN_HEIGHT, Math.min(maxH, startHeight + delta));
      setPanelHeight(newHeight);
      setExpanded(newHeight >= maxH - 20);
    };

    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.removeEventListener('touchmove', onMove);
      document.removeEventListener('touchend', onUp);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    document.addEventListener('touchmove', onMove);
    document.addEventListener('touchend', onUp);
  }, [panelHeight]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchMultiModelForecast(location.lat, location.lng, COMPARE_MODELS, [
      selectedVar,
      'precipitation',
    ])
      .then((data) => { if (!cancelled) setForecasts(data); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [location.lat, location.lng, selectedVar]);

  useEffect(() => {
    if (!forecasts.length || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    const pad = { top: 50, right: 30, bottom: 60, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // Gather all data for range calculation
    let allVals: number[] = [];
    let maxN = 0;
    for (const f of forecasts) {
      const vals = f.hourly[selectedVar] as number[] | undefined;
      if (vals) {
        allVals = allVals.concat(vals.filter((v) => v != null));
        maxN = Math.max(maxN, vals.length);
      }
    }
    if (allVals.length === 0 || maxN === 0) return;

    const vMin = Math.floor(Math.min(...allVals) - 2);
    const vMax = Math.ceil(Math.max(...allVals) + 2);

    const xScale = (i: number) => pad.left + (i / (maxN - 1)) * plotW;
    const yScale = (v: number) => pad.top + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 0.8;
    for (let i = 0; i <= 5; i++) {
      const val = vMin + (i / 5) * (vMax - vMin);
      const y = yScale(val);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${val.toFixed(0)}`, pad.left - 8, y + 4);
    }

    // Time labels from first forecast
    const times = forecasts[0]?.hourly.time as string[] | undefined;
    if (times) {
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      const labelInterval = Math.max(1, Math.floor(maxN / 10));
      for (let i = 0; i < maxN; i += labelInterval) {
        const x = xScale(i);
        const d = new Date(times[i]);
        ctx.fillText(
          d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
          x,
          H - pad.bottom + 20,
        );
        ctx.fillText(
          d.toLocaleTimeString(undefined, { hour: '2-digit', hour12: false }),
          x,
          H - pad.bottom + 34,
        );
      }
    }

    // Draw each model's line
    for (const f of forecasts) {
      const modelMeta = MODELS.find((m) => m.id === f.model);
      const color = modelMeta?.color ?? '#94a3b8';
      const vals = f.hourly[selectedVar] as number[] | undefined;
      if (!vals) continue;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < vals.length; i++) {
        if (vals[i] != null) {
          if (!started) { ctx.moveTo(xScale(i), yScale(vals[i])); started = true; }
          else ctx.lineTo(xScale(i), yScale(vals[i]));
        }
      }
      ctx.stroke();
    }

    // Hover
    if (hoverIndex != null && hoverIndex >= 0 && hoverIndex < maxN) {
      const x = xScale(hoverIndex);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, H - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Values box
      let boxY = pad.top + 10;
      for (const f of forecasts) {
        const modelMeta = MODELS.find((m) => m.id === f.model);
        const vals = f.hourly[selectedVar] as number[] | undefined;
        const val = vals?.[hoverIndex];
        if (val != null) {
          // Dot on line
          ctx.fillStyle = modelMeta?.color ?? '#fff';
          ctx.beginPath();
          ctx.arc(x, yScale(val), 4, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = modelMeta?.color ?? '#fff';
          ctx.font = '11px monospace';
          ctx.textAlign = 'left';
          ctx.fillText(
            `${modelMeta?.label ?? f.model}: ${val.toFixed(1)}`,
            x + 12,
            boxY,
          );
          boxY += 16;
        }
      }
    }

    // Legend
    ctx.font = 'bold 11px monospace';
    let lx = pad.left;
    const ly = pad.top - 16;
    for (const f of forecasts) {
      const modelMeta = MODELS.find((m) => m.id === f.model);
      ctx.strokeStyle = modelMeta?.color ?? '#fff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(lx + 18, ly);
      ctx.stroke();
      ctx.fillStyle = modelMeta?.color ?? '#fff';
      ctx.textAlign = 'left';
      ctx.fillText(modelMeta?.label ?? f.model, lx + 22, ly + 4);
      lx += 80;
    }

    // Title
    const varMeta = SURFACE_VARIABLES.find((v) => v.id === selectedVar);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      `Model Comparison: ${varMeta?.label ?? selectedVar} — ${location.lat.toFixed(2)}N, ${location.lng.toFixed(2)}E`,
      W / 2,
      18,
    );
  }, [forecasts, hoverIndex, selectedVar, location]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!forecasts.length || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pad = { left: 60, right: 30 };
    const plotW = rect.width - pad.left - pad.right;
    const maxN = Math.max(...forecasts.map((f) => (f.hourly.time as string[]).length));
    const index = Math.round(((x - pad.left) / plotW) * (maxN - 1));
    setHoverIndex(Math.max(0, Math.min(maxN - 1, index)));
  };

  const varOptions: SurfaceVariable[] = [
    'temperature_2m',
    'dewpoint_2m',
    'wind_speed_10m',
    'pressure_msl',
    'cloud_cover',
    'relative_humidity_2m',
  ];

  return (
    <div
      className={`comparison-panel ${expanded ? 'comparison-panel--expanded' : ''}`}
      ref={panelRef}
      style={{ height: panelHeight }}
    >
      {/* Drag handle */}
      <div
        className="comparison-panel__drag-handle"
        onMouseDown={handleDragStart}
        onTouchStart={handleDragStart}
      />
      <div className="comparison-panel__header">
        <h3>Model Comparison</h3>
        <div className="comparison-panel__var-select">
          {varOptions.map((v) => {
            const meta = SURFACE_VARIABLES.find((sv) => sv.id === v);
            return (
              <button
                key={v}
                className={`model-chip ${selectedVar === v ? 'model-chip--active' : ''}`}
                onClick={() => setSelectedVar(v)}
              >
                {meta?.label ?? v}
              </button>
            );
          })}
        </div>
        <button
          className="comparison-panel__expand"
          onClick={toggleExpand}
          title={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? '\u2193' : '\u2191'}
        </button>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>
      {loading && <div className="comparison-panel__loading">Loading multi-model data...</div>}
      {error && <div className="comparison-panel__error">{error}</div>}
      <canvas
        ref={canvasRef}
        className="comparison-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIndex(null)}
      />
    </div>
  );
}
