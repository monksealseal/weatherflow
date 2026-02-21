import { useEffect, useRef, useState } from 'react';
import type { LatLng, PointForecast, WeatherModel } from '../types/weather';
import { MODELS } from '../types/weather';
import { fetchPointForecast } from '../api/openMeteo';

interface MeteogramProps {
  location: LatLng;
  model: WeatherModel;
  onClose: () => void;
}

export default function Meteogram({ location, model, onClose }: MeteogramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [forecast, setForecast] = useState<PointForecast | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchPointForecast(location.lat, location.lng, model, undefined, 7)
      .then((data) => { if (!cancelled) setForecast(data); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [location.lat, location.lng, model]);

  useEffect(() => {
    if (!forecast || !canvasRef.current) return;
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

    const pad = { top: 40, right: 60, bottom: 60, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const times = forecast.hourly.time as string[];
    const temp = forecast.hourly.temperature_2m as number[];
    const dewpoint = forecast.hourly.dewpoint_2m as number[];
    const precip = forecast.hourly.precipitation as number[];
    const wind = forecast.hourly.wind_speed_10m as number[];
    const cloud = forecast.hourly.cloud_cover as number[];
    const pressure = forecast.hourly.pressure_msl as number[];

    if (!temp || temp.length === 0) return;
    const n = temp.length;

    // Clear
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // Compute ranges
    const allTemps = [...(temp || []), ...(dewpoint || [])].filter((v) => v != null);
    const tempMin = Math.floor(Math.min(...allTemps) - 2);
    const tempMax = Math.ceil(Math.max(...allTemps) + 2);

    const precipMax = Math.max(2, ...((precip || []).filter((v) => v != null) as number[]));

    // Scale functions
    const xScale = (i: number) => pad.left + (i / (n - 1)) * plotW;
    const yTempScale = (v: number) => pad.top + plotH - ((v - tempMin) / (tempMax - tempMin)) * plotH;
    const yPrecipScale = (v: number) => pad.top + plotH - (v / precipMax) * plotH * 0.3;

    // Draw grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const tempSteps = 5;
    for (let i = 0; i <= tempSteps; i++) {
      const val = tempMin + (i / tempSteps) * (tempMax - tempMin);
      const y = yTempScale(val);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${val.toFixed(0)}°`, pad.left - 8, y + 4);
    }

    // Draw time labels
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    const labelInterval = Math.max(1, Math.floor(n / 12));
    for (let i = 0; i < n; i += labelInterval) {
      const x = xScale(i);
      const d = new Date(times[i]);
      const label = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      const timeLabel = d.toLocaleTimeString(undefined, { hour: '2-digit', hour12: false });
      ctx.fillText(label, x, H - pad.bottom + 20);
      ctx.fillText(timeLabel, x, H - pad.bottom + 34);

      // Vertical grid line
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, H - pad.bottom);
      ctx.stroke();

      // Day separator (midnight)
      if (d.getHours() === 0) {
        ctx.strokeStyle = 'rgba(255,255,255,0.12)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, H - pad.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw cloud cover as background gradient bars
    if (cloud) {
      for (let i = 0; i < n - 1; i++) {
        const c = cloud[i];
        if (c == null || c < 5) continue;
        const x = xScale(i);
        const w = xScale(i + 1) - x;
        ctx.fillStyle = `rgba(148, 163, 184, ${(c / 100) * 0.15})`;
        ctx.fillRect(x, pad.top, w, plotH);
      }
    }

    // Draw precipitation bars
    if (precip) {
      for (let i = 0; i < n; i++) {
        const p = precip[i];
        if (p == null || p <= 0) continue;
        const x = xScale(i);
        const barW = Math.max(2, plotW / n - 1);
        const barH = (p / precipMax) * plotH * 0.3;
        const gradient = ctx.createLinearGradient(0, H - pad.bottom - barH, 0, H - pad.bottom);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.2)');
        ctx.fillStyle = gradient;
        ctx.fillRect(x - barW / 2, H - pad.bottom - barH, barW, barH);
      }
    }

    // Draw temperature line
    if (temp) {
      // Gradient fill under temperature
      ctx.beginPath();
      ctx.moveTo(xScale(0), yTempScale(temp[0]));
      for (let i = 1; i < n; i++) {
        if (temp[i] != null) {
          ctx.lineTo(xScale(i), yTempScale(temp[i]));
        }
      }
      ctx.lineTo(xScale(n - 1), H - pad.bottom);
      ctx.lineTo(xScale(0), H - pad.bottom);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.15)');
      grad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();

      // Temperature line
      ctx.beginPath();
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2.5;
      ctx.lineJoin = 'round';
      let started = false;
      for (let i = 0; i < n; i++) {
        if (temp[i] != null) {
          if (!started) {
            ctx.moveTo(xScale(i), yTempScale(temp[i]));
            started = true;
          } else {
            ctx.lineTo(xScale(i), yTempScale(temp[i]));
          }
        }
      }
      ctx.stroke();
    }

    // Draw dewpoint line
    if (dewpoint) {
      ctx.beginPath();
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      let started = false;
      for (let i = 0; i < n; i++) {
        if (dewpoint[i] != null) {
          if (!started) {
            ctx.moveTo(xScale(i), yTempScale(dewpoint[i]));
            started = true;
          } else {
            ctx.lineTo(xScale(i), yTempScale(dewpoint[i]));
          }
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw pressure line on secondary axis
    if (pressure) {
      const pMin = Math.min(...pressure.filter((v) => v != null) as number[]) - 2;
      const pMax = Math.max(...pressure.filter((v) => v != null) as number[]) + 2;
      const yPressure = (v: number) => pad.top + plotH - ((v - pMin) / (pMax - pMin)) * plotH;

      ctx.beginPath();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      let started = false;
      for (let i = 0; i < n; i++) {
        if (pressure[i] != null) {
          if (!started) {
            ctx.moveTo(xScale(i), yPressure(pressure[i]));
            started = true;
          } else {
            ctx.lineTo(xScale(i), yPressure(pressure[i]));
          }
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Secondary axis labels
      ctx.fillStyle = '#f59e0b';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      for (let i = 0; i <= 3; i++) {
        const val = pMin + (i / 3) * (pMax - pMin);
        const y = yPressure(val);
        ctx.fillText(`${val.toFixed(0)}`, W - pad.right + 8, y + 4);
      }
    }

    // Draw wind barbs
    if (wind) {
      const barbInterval = Math.max(1, Math.floor(n / 24));
      for (let i = 0; i < n; i += barbInterval) {
        if (wind[i] == null) continue;
        const x = xScale(i);
        const speed = wind[i];
        const windDirArr = forecast.hourly.wind_direction_10m as number[] | undefined;
        const dir = windDirArr?.[i] ?? 0;

        drawWindBarb(ctx, x, pad.top - 6, speed, dir);
      }
    }

    // Draw hover crosshair
    if (hoverIndex != null && hoverIndex >= 0 && hoverIndex < n) {
      const x = xScale(hoverIndex);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, H - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Hover values
      const t = temp[hoverIndex];
      const dp = dewpoint?.[hoverIndex];
      const p = precip?.[hoverIndex];
      const w = wind?.[hoverIndex];
      const time = new Date(times[hoverIndex]);

      ctx.fillStyle = 'rgba(13, 17, 23, 0.9)';
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      const boxX = Math.min(x + 10, W - 180);
      const boxY = pad.top + 10;
      roundRect(ctx, boxX, boxY, 165, 100, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#e2e8f0';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      const timeStr = time.toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      });
      ctx.fillText(timeStr, boxX + 8, boxY + 18);
      ctx.fillStyle = '#ef4444';
      ctx.fillText(`Temp: ${t != null ? t.toFixed(1) + '°C' : '--'}`, boxX + 8, boxY + 36);
      ctx.fillStyle = '#22c55e';
      ctx.fillText(`Dewpt: ${dp != null ? dp.toFixed(1) + '°C' : '--'}`, boxX + 8, boxY + 52);
      ctx.fillStyle = '#3b82f6';
      ctx.fillText(`Precip: ${p != null ? p.toFixed(1) + ' mm' : '--'}`, boxX + 8, boxY + 68);
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`Wind: ${w != null ? w.toFixed(0) + ' km/h' : '--'}`, boxX + 8, boxY + 84);
    }

    // Legend
    const legendY = pad.top - 24;
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    drawLegendItem(ctx, pad.left, legendY, '#ef4444', 'Temp', false);
    drawLegendItem(ctx, pad.left + 70, legendY, '#22c55e', 'Dewpoint', true);
    drawLegendItem(ctx, pad.left + 160, legendY, '#3b82f6', 'Precip', false);
    drawLegendItem(ctx, pad.left + 230, legendY, '#f59e0b', 'MSLP', true);

    // Title
    const modelMeta = MODELS.find((m) => m.id === model);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      `${modelMeta?.label ?? model} Meteogram — ${location.lat.toFixed(2)}N, ${location.lng.toFixed(2)}E`,
      W / 2,
      16,
    );

  }, [forecast, hoverIndex, model, location]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!forecast || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pad = { left: 60, right: 60 };
    const plotW = rect.width - pad.left - pad.right;
    const n = (forecast.hourly.time as string[]).length;
    const index = Math.round(((x - pad.left) / plotW) * (n - 1));
    setHoverIndex(Math.max(0, Math.min(n - 1, index)));
  };

  return (
    <div className="meteogram-panel">
      <div className="meteogram-panel__header">
        <h3>Meteogram</h3>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>
      {loading && <div className="meteogram-panel__loading">Loading meteogram data...</div>}
      {error && <div className="meteogram-panel__error">{error}</div>}
      <canvas
        ref={canvasRef}
        className="meteogram-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIndex(null)}
      />
    </div>
  );
}

// ── Drawing helpers ─────────────────────────────────────────────────

function drawWindBarb(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  speedKmh: number,
  dirDeg: number,
) {
  const speedKt = speedKmh * 0.539957;
  const len = 14;
  const rad = ((dirDeg + 180) * Math.PI) / 180;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rad);
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -len);
  ctx.stroke();

  let remaining = speedKt;
  let pos = -len;
  const step = 3;

  // Flags (50 kt)
  while (remaining >= 50) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(6, pos + step / 2);
    ctx.lineTo(0, pos + step);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fill();
    pos += step;
    remaining -= 50;
  }
  // Full barbs (10 kt)
  while (remaining >= 10) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(6, pos + step / 2);
    ctx.stroke();
    pos += step;
    remaining -= 10;
  }
  // Half barb (5 kt)
  if (remaining >= 5) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(3, pos + step / 3);
    ctx.stroke();
  }

  ctx.restore();
}

function drawLegendItem(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  label: string,
  dashed: boolean,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + 16, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.fillText(label, x + 20, y + 4);
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
