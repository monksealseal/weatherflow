import { useEffect, useRef, useState } from 'react';
import type { LatLng, WeatherModel, SoundingPoint } from '../types/weather';
import { MODELS } from '../types/weather';
import { fetchSounding } from '../api/openMeteo';

interface SkewTProps {
  location: LatLng;
  model: WeatherModel;
  forecastHour: number;
  onClose: () => void;
}

// ── Constants ───────────────────────────────────────────────────────
const P_TOP = 100;   // hPa
const P_BOT = 1050;  // hPa
const T_MIN = -40;   // °C
const T_MAX = 50;    // °C
const SKEW = 1.0;    // skew factor

export default function SkewT({ location, model, forecastHour, onClose }: SkewTProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [sounding, setSounding] = useState<SoundingPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchSounding(location.lat, location.lng, model, forecastHour)
      .then((data) => { if (!cancelled) setSounding(data); })
      .catch((err) => { if (!cancelled) setError(err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [location.lat, location.lng, model, forecastHour]);

  useEffect(() => {
    if (!sounding || !canvasRef.current) return;
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

    const pad = { top: 40, right: 80, bottom: 40, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    // Clear
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, W, H);

    // Scale functions
    const yScale = (p: number) => {
      const logP = Math.log(p);
      const logTop = Math.log(P_TOP);
      const logBot = Math.log(P_BOT);
      return pad.top + ((logP - logTop) / (logBot - logTop)) * plotH;
    };

    const xScale = (t: number, p: number) => {
      const y = yScale(p);
      const yNorm = (y - pad.top) / plotH;
      const tFrac = (t - T_MIN) / (T_MAX - T_MIN);
      return pad.left + (tFrac + SKEW * (1 - yNorm)) * plotW / (1 + SKEW);
    };

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, plotW, plotH);
    ctx.clip();

    // ── Background: dry adiabats ──
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.08)';
    ctx.lineWidth = 0.8;
    for (let theta = -30; theta <= 80; theta += 10) {
      ctx.beginPath();
      let first = true;
      for (let p = P_BOT; p >= P_TOP; p -= 10) {
        const t = dryAdiabat(theta + 273.15, p) - 273.15;
        const x = xScale(t, p);
        const y = yScale(p);
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // ── Background: moist adiabats ──
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.08)';
    ctx.lineWidth = 0.8;
    for (let tw = -20; tw <= 35; tw += 5) {
      ctx.beginPath();
      let first = true;
      let t = tw;
      for (let p = P_BOT; p >= P_TOP; p -= 10) {
        t -= moistAdiabatRate(t, p) * 10;
        const x = xScale(t, p);
        const y = yScale(p);
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // ── Background: mixing ratio lines ──
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.06)';
    ctx.lineWidth = 0.6;
    ctx.setLineDash([3, 5]);
    for (const w of [0.5, 1, 2, 4, 7, 10, 15, 20, 30]) {
      ctx.beginPath();
      let first = true;
      for (let p = P_BOT; p >= 200; p -= 20) {
        const t = dewpointFromMixingRatio(w, p);
        const x = xScale(t, p);
        const y = yScale(p);
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // ── Isotherms ──
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 0.8;
    for (let t = T_MIN; t <= T_MAX; t += 10) {
      ctx.beginPath();
      ctx.moveTo(xScale(t, P_BOT), yScale(P_BOT));
      ctx.lineTo(xScale(t, P_TOP), yScale(P_TOP));
      ctx.stroke();
    }

    // 0°C isotherm
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xScale(0, P_BOT), yScale(P_BOT));
    ctx.lineTo(xScale(0, P_TOP), yScale(P_TOP));
    ctx.stroke();

    // ── CAPE shading ──
    const filteredSounding = sounding.filter((pt) => pt.pressure >= P_TOP && pt.pressure <= P_BOT);
    if (filteredSounding.length > 2) {
      // Simple parcel path from surface
      const surfT = filteredSounding[0]?.temperature ?? 20;
      const surfDp = filteredSounding[0]?.dewpoint ?? 10;
      const surfP = filteredSounding[0]?.pressure ?? 1000;
      const parcelPath = computeParcelPath(surfT, surfDp, surfP);

      // Draw CAPE (parcel warmer than environment)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
      for (let i = 0; i < filteredSounding.length - 1; i++) {
        const pt = filteredSounding[i];
        const ptNext = filteredSounding[i + 1];
        const parcelT = parcelTempAtPressure(parcelPath, pt.pressure);
        const parcelTNext = parcelTempAtPressure(parcelPath, ptNext.pressure);

        if (parcelT != null && parcelTNext != null && parcelT > pt.temperature) {
          ctx.beginPath();
          ctx.moveTo(xScale(pt.temperature, pt.pressure), yScale(pt.pressure));
          ctx.lineTo(xScale(parcelT, pt.pressure), yScale(pt.pressure));
          ctx.lineTo(xScale(parcelTNext, ptNext.pressure), yScale(ptNext.pressure));
          ctx.lineTo(xScale(ptNext.temperature, ptNext.pressure), yScale(ptNext.pressure));
          ctx.closePath();
          ctx.fill();
        }
      }
    }

    // ── Temperature profile ──
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (const pt of filteredSounding) {
      const x = xScale(pt.temperature, pt.pressure);
      const y = yScale(pt.pressure);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // ── Dewpoint profile ──
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    started = false;
    for (const pt of filteredSounding) {
      const x = xScale(pt.dewpoint, pt.pressure);
      const y = yScale(pt.pressure);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.restore(); // Remove clip

    // ── Pressure axis labels ──
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    for (const p of [1000, 925, 850, 700, 500, 400, 300, 200, 150, 100]) {
      const y = yScale(p);
      if (y >= pad.top && y <= pad.top + plotH) {
        ctx.fillText(`${p}`, pad.left - 8, y + 4);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
      }
    }

    // ── Wind barbs on right side ──
    for (const pt of filteredSounding) {
      if (pt.windSpeed == null) continue;
      const y = yScale(pt.pressure);
      const x = pad.left + plotW + 25;
      drawWindBarb(ctx, x, y, pt.windSpeed, pt.windDir);
    }

    // ── Axis labels ──
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('hPa', pad.left - 30, pad.top - 8);

    // Temperature axis at bottom
    for (let t = T_MIN; t <= T_MAX; t += 10) {
      const x = xScale(t, P_BOT);
      if (x >= pad.left && x <= pad.left + plotW) {
        ctx.fillText(`${t}°`, x, H - pad.bottom + 20);
      }
    }

    // ── Title ──
    const modelMeta = MODELS.find((m) => m.id === model);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      `${modelMeta?.label ?? model} Sounding — ${location.lat.toFixed(2)}N, ${location.lng.toFixed(2)}E — F${String(forecastHour).padStart(3, '0')}`,
      W / 2,
      18,
    );

    // Legend
    ctx.font = '11px monospace';
    const lx = pad.left + plotW - 150;
    const ly = pad.top + 16;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 18, ly); ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'left';
    ctx.fillText('Temperature', lx + 22, ly + 4);

    ctx.strokeStyle = '#22c55e';
    ctx.beginPath(); ctx.moveTo(lx, ly + 18); ctx.lineTo(lx + 18, ly + 18); ctx.stroke();
    ctx.fillStyle = '#22c55e';
    ctx.fillText('Dewpoint', lx + 22, ly + 22);

  }, [sounding, model, location, forecastHour]);

  return (
    <div className="skewt-panel">
      <div className="skewt-panel__header">
        <h3>Skew-T Log-P Diagram</h3>
        <button className="forecast-panel__close" onClick={onClose}>x</button>
      </div>
      {loading && <div className="skewt-panel__loading">Loading sounding data...</div>}
      {error && <div className="skewt-panel__error">{error}</div>}
      <canvas ref={canvasRef} className="skewt-canvas" />
    </div>
  );
}

// ── Thermodynamic helpers ───────────────────────────────────────────

function dryAdiabat(thetaK: number, pressure: number): number {
  // Poisson's equation: T = theta * (p / 1000)^(R/cp)
  return thetaK * Math.pow(pressure / 1000, 0.286);
}

function moistAdiabatRate(tC: number, pHpa: number): number {
  // Approximate lapse rate for saturated parcel (°C per hPa)
  const tK = tC + 273.15;
  const es = 6.112 * Math.exp((17.67 * tC) / (tC + 243.5));
  const rs = 0.622 * es / (pHpa - es);
  const Lv = 2.501e6;
  const cp = 1004;
  const Rd = 287;

  const num = (Rd * tK + Lv * rs) / (pHpa * 100);
  const den = 1 + (Lv * Lv * rs) / (cp * Rd * tK * tK);
  return num / den * 100; // per hPa
}

function dewpointFromMixingRatio(w: number, pHpa: number): number {
  // w in g/kg
  const wKg = w / 1000;
  const e = (wKg * pHpa) / (0.622 + wKg);
  return (243.5 * Math.log(e / 6.112)) / (17.67 - Math.log(e / 6.112));
}

interface ParcelPoint {
  pressure: number;
  temperature: number;
}

function computeParcelPath(surfT: number, surfDp: number, surfP: number): ParcelPoint[] {
  const path: ParcelPoint[] = [];
  let t = surfT;
  let p = surfP;

  // Find LCL (lifting condensation level)
  const lclT = surfDp; // simplified
  const lclP = surfP * Math.pow((lclT + 273.15) / (surfT + 273.15), 1 / 0.286);

  // Dry adiabatic ascent to LCL
  for (; p >= Math.max(lclP, P_TOP); p -= 10) {
    t = (surfT + 273.15) * Math.pow(p / surfP, 0.286) - 273.15;
    path.push({ pressure: p, temperature: t });
  }

  // Moist adiabatic ascent above LCL
  t = path.length > 0 ? path[path.length - 1].temperature : lclT;
  for (; p >= P_TOP; p -= 10) {
    t -= moistAdiabatRate(t, p) * 10;
    path.push({ pressure: p, temperature: t });
  }

  return path;
}

function parcelTempAtPressure(path: ParcelPoint[], p: number): number | null {
  for (let i = 0; i < path.length - 1; i++) {
    if (path[i].pressure >= p && path[i + 1].pressure <= p) {
      const frac = (p - path[i + 1].pressure) / (path[i].pressure - path[i + 1].pressure);
      return path[i + 1].temperature + frac * (path[i].temperature - path[i + 1].temperature);
    }
  }
  return null;
}

function drawWindBarb(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  speedKmh: number,
  dirDeg: number,
) {
  const speedKt = speedKmh * 0.539957;
  const len = 20;
  const rad = ((dirDeg + 180) * Math.PI) / 180;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rad);
  ctx.strokeStyle = 'rgba(255,255,255,0.6)';
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -len);
  ctx.stroke();

  let remaining = speedKt;
  let pos = -len;
  const step = 4;

  while (remaining >= 50) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(8, pos + step / 2);
    ctx.lineTo(0, pos + step);
    ctx.fill();
    pos += step;
    remaining -= 50;
  }
  while (remaining >= 10) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(8, pos + step / 2);
    ctx.stroke();
    pos += step;
    remaining -= 10;
  }
  if (remaining >= 5) {
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(4, pos + step / 3);
    ctx.stroke();
  }

  ctx.restore();
}
