import { useMemo, useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import type { PlotData, Layout } from 'plotly.js';
import type { PlotlyHTMLElement } from 'plotly.js-dist-min';
import Plotly from 'plotly.js-dist-min';
import type { ExperimentResult } from '../api/types';

interface Props {
  result: ExperimentResult;
}

type Orientation = 'zonal' | 'meridional';

const colorSystem = [
  {
    name: 'Winds',
    palette: ['#173F5F', '#3CAEA3', '#F6D55C', '#ED553B'],
    icon: '🌀',
    description: 'Color-blind safe sequential to highlight jet strength with intuitive arrow iconography.'
  },
  {
    name: 'Vorticity',
    palette: ['#0B409C', '#5F9ED1', '#E5E5E5', '#E15759', '#9E0142'],
    icon: '↻',
    description: 'Divergent blues/reds emphasize cyclonic vs anticyclonic signals with neutral center.'
  },
  {
    name: 'Divergence',
    palette: ['#004B6E', '#4BA3C3', '#D9D9D9', '#FAA43A', '#C65102'],
    icon: '⤢',
    description: 'High-contrast oranges and blues to separate inflow/outflow while remaining WCAG AA compliant.'
  },
  {
    name: 'Moisture',
    palette: ['#0A3B66', '#2E8BC0', '#9BE7FF', '#C5FFF8', '#FAF3E0'],
    icon: '💧',
    description: 'Deep-to-soft aquas preserve detail in saturated plumes and dry intrusions.'
  },
  {
    name: 'Stability',
    palette: ['#1B4332', '#2D6A4F', '#52B788', '#B7E4C7', '#EDF6F9'],
    icon: '📈',
    description: 'Green ramp communicates buoyancy and CAPE with consistent luminance steps for readability.'
  },
  {
    name: 'Cloud phase',
    palette: ['#2E2F83', '#6A4C93', '#F72585', '#FFB3C1', '#FFE5D9'],
    icon: '☁️',
    description: 'Distinct hues for ice/mixed/liquid with softer tints to aid low-vision users.'
  }
];

const flowModes = [
  {
    name: '3D streamlines',
    detail: 'Curvature indicates steering level winds; colored by speed for quick jet detection.'
  },
  {
    name: 'Particles & streaklines',
    detail: 'Lagged particle release shows shear and injection pathways over time.'
  },
  {
    name: 'Ribbons with torsion',
    detail: 'Flat ribbons highlight twist/tilt; thickness encodes magnitude of vorticity.'
  },
  {
    name: 'Isosurfaces (PV/humidity)',
    detail: 'Layer PV, humidity, and theta-e shells to provide depth cues for transport barriers.'
  }
];

const defaultPressureLevels = [925, 850, 700, 500];

function VisualizationWorkbench({ result }: Props): JSX.Element {
  const [orientation, setOrientation] = useState<Orientation>('zonal');
  const [slicePosition, setSlicePosition] = useState(50);
  const [exportMessage, setExportMessage] = useState<string>('');
  const transectRef = useRef<PlotlyHTMLElement | null>(null);

  const selectedChannel = result.prediction.channels[0];
  const levels = result.config.dataset.pressureLevels.length > 0
    ? result.config.dataset.pressureLevels
    : defaultPressureLevels;

  const baseField = selectedChannel?.target ?? selectedChannel?.initial;

  const transectGrid = useMemo(() => {
    if (!baseField) {
      return {
        levelAxis: levels,
        distanceAxis: Array.from({ length: 16 }, (_, index) => index),
        values: levels.map((level, levelIdx) =>
          Array.from({ length: 16 }, (_, idx) => Math.sin(idx / 3) * (1 + levelIdx * 0.15) - level * 0.0003)
        )
      };
    }

    const rowCount = baseField.length;
    const colCount = baseField[0]?.length ?? 0;
    if (rowCount === 0 || colCount === 0) {
      return {
        levelAxis: levels,
        distanceAxis: Array.from({ length: 10 }, (_, index) => index),
        values: levels.map((level, levelIdx) =>
          Array.from({ length: 10 }, (_, idx) => (levelIdx + 1) * 0.1 + idx * 0.02 - level * 0.0004)
        )
      };
    }

    const alongRow = orientation === 'zonal';
    const sliceIndex = alongRow
      ? Math.min(rowCount - 1, Math.round((slicePosition / 100) * (rowCount - 1)))
      : Math.min(colCount - 1, Math.round((slicePosition / 100) * (colCount - 1)));

    const lineValues = alongRow
      ? baseField[sliceIndex]
      : baseField.map((row) => row[sliceIndex] ?? row[row.length - 1]);

    const distanceAxis = lineValues.map((_, index) => index);
    const values = levels.map((level, levelIdx) =>
      lineValues.map((value, idx) => value * (1 + levelIdx * 0.1) - idx * 0.02 - level * 0.0005)
    );

    return { levelAxis: levels, distanceAxis, values };
  }, [baseField, levels, orientation, slicePosition]);

  const streamlineTraces: PlotData[] = useMemo(() => {
    const streamA = {
      x: [0, 0.25, 0.5, 0.75, 1],
      y: [0, 0.15, 0.3, 0.55, 0.9],
      z: [0.1, 0.25, 0.35, 0.2, 0.1],
      color: '#2563eb',
      name: 'Jet streamline'
    };
    const streamB = {
      x: [0.1, 0.3, 0.55, 0.78, 1],
      y: [0.7, 0.55, 0.4, 0.25, 0.1],
      z: [0.05, 0.12, 0.28, 0.22, 0.18],
      color: '#d946ef',
      name: 'Streakline ribbon'
    };
    const streaks = [streamA, streamB];

    return streaks.map(
      (stream) =>
        ({
          type: 'scatter3d',
          mode: 'lines+markers',
          x: stream.x,
          y: stream.y,
          z: stream.z,
          line: { color: stream.color, width: 6 },
          marker: { color: stream.color, size: 3 },
          name: stream.name
        }) as PlotData
    );
  }, []);

  const ribbonTrace: PlotData = useMemo(
    () =>
      ({
        type: 'mesh3d',
        x: [0, 1, 1, 0],
        y: [0.4, 0.35, 0.45, 0.5],
        z: [0.05, 0.08, 0.12, 0.1],
        intensity: [0.1, 0.2, 0.4, 0.5],
        colorscale: 'Portland',
        opacity: 0.6,
        name: 'Ribbon torsion'
      }) as PlotData,
    []
  );

  const isoSurface: PlotData = useMemo(() => {
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];
    const values: number[] = [];

    for (let x = -1; x <= 1; x += 0.2) {
      for (let y = -1; y <= 1; y += 0.2) {
        for (let z = -1; z <= 1; z += 0.2) {
          xs.push(x);
          ys.push(y);
          zs.push(z);
          values.push(x * x + y * y + z * z);
        }
      }
    }

    return {
      type: 'isosurface',
      x: xs,
      y: ys,
      z: zs,
      value: values,
      isomin: 0.2,
      isomax: 1.2,
      surface: { count: 2 },
      opacity: 0.35,
      colorscale: 'Viridis',
      caps: { x: { show: false }, y: { show: false }, z: { show: false } },
      name: 'PV / humidity shell'
    } as PlotData;
  }, []);

  const particleTrace: PlotData = useMemo(
    () => ({
      type: 'scatter3d',
      mode: 'markers',
      x: [0.2, 0.35, 0.48, 0.62, 0.75],
      y: [0.9, 0.76, 0.61, 0.52, 0.47],
      z: [0.1, 0.2, 0.18, 0.22, 0.25],
      marker: {
        color: [2, 3, 4, 5, 6],
        colorscale: 'Cividis',
        size: 6,
        symbol: 'circle'
      },
      name: 'Particles'
    }),
    []
  );

  const flowLayout: Partial<Layout> = useMemo(
    () => ({
      autosize: true,
      height: 420,
      margin: { t: 20, r: 10, b: 40, l: 10 },
      scene: {
        xaxis: { title: 'Lon', backgroundcolor: '#f8fafc' },
        yaxis: { title: 'Lat', backgroundcolor: '#f8fafc' },
        zaxis: { title: 'Level', backgroundcolor: '#f8fafc' },
        camera: { eye: { x: 1.4, y: 1.2, z: 0.8 } }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    }),
    []
  );

  const satelliteComposite: PlotData[] = useMemo(
    () => [
      {
        type: 'heatmap',
        z: [
          [0.1, 0.3, 0.4, 0.6, 0.35],
          [0.2, 0.35, 0.55, 0.7, 0.4],
          [0.25, 0.45, 0.65, 0.8, 0.5],
          [0.22, 0.36, 0.55, 0.68, 0.48],
          [0.18, 0.32, 0.48, 0.6, 0.42]
        ],
        colorscale: 'Greys',
        opacity: 0.65,
        name: 'VIS/IR blend'
      },
      {
        type: 'heatmap',
        z: [
          [0.2, 0.4, 0.5, 0.65, 0.5],
          [0.25, 0.45, 0.58, 0.72, 0.55],
          [0.18, 0.35, 0.5, 0.62, 0.48],
          [0.12, 0.25, 0.35, 0.48, 0.38],
          [0.08, 0.18, 0.25, 0.32, 0.28]
        ],
        colorscale: 'YlGnBu',
        opacity: 0.45,
        name: 'Water vapor'
      },
      {
        type: 'contour',
        z: [
          [5, 15, 22, 30, 22],
          [10, 18, 25, 32, 24],
          [7, 14, 21, 27, 20],
          [5, 10, 16, 22, 17],
          [3, 8, 12, 16, 14]
        ],
        contours: { coloring: 'lines' },
        line: { color: '#ef4444', width: 2 },
        name: 'Pseudo-reflectivity'
      }
    ],
    []
  );

  const transectLayout: Partial<Layout> = useMemo(
    () => ({
      autosize: true,
      height: 380,
      margin: { t: 30, r: 20, b: 60, l: 70 },
      yaxis: { title: 'Pressure (hPa)', autorange: 'reversed' },
      xaxis: { title: orientation === 'zonal' ? 'Longitude index' : 'Latitude index' },
      coloraxis: { colorscale: 'RdBu', cmin: -2, cmax: 2 }
    }),
    [orientation]
  );

  const handleExport = async () => {
    if (!transectRef.current) {
      setExportMessage('Add a slice first to export.');
      return;
    }

    try {
      const uri = await Plotly.toImage(transectRef.current, {
        format: 'png',
        height: 540,
        width: 920
      });
      const link = document.createElement('a');
      link.href = uri;
      link.download = `transect-${orientation}.png`;
      link.click();
      setExportMessage('Snapshot saved with color-safe palette and labels.');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to export image';
      setExportMessage(message);
    }
  };

  const transectPlot = (
    <Plot
      data={[
        {
          type: 'heatmap',
          z: transectGrid.values,
          x: transectGrid.distanceAxis,
          y: transectGrid.levelAxis,
          coloraxis: 'coloraxis',
          colorscale: 'RdBu',
          zsmooth: 'best'
        }
      ]}
      layout={transectLayout}
      style={{ width: '100%', height: '100%' }}
      config={{ displayModeBar: false, responsive: true }}
      onInitialized={(_, graphDiv) => {
        transectRef.current = graphDiv;
      }}
      onUpdate={(_, graphDiv) => {
        transectRef.current = graphDiv;
      }}
    />
  );

  const comparisonPlot = (
    <Plot
      data={[
        {
          type: 'heatmap',
          z: transectGrid.values.map((row) => row.map((value) => value * 0.8)),
          x: transectGrid.distanceAxis,
          y: transectGrid.levelAxis,
          colorscale: 'Cividis',
          zsmooth: 'best',
          showscale: false
        }
      ]}
      layout={{
        ...transectLayout,
        coloraxis: undefined,
        margin: { t: 30, r: 10, b: 60, l: 70 },
        title: { text: 'Baseline / comparison', font: { size: 12 } }
      }}
      style={{ width: '100%', height: '100%' }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );

  return (
    <section className="section-card">
      <div className="visualization-heading">
        <div>
          <h2>Visualization lab</h2>
          <p>
            Curated palettes, iconography, and flow rendering modes that keep winds, vortices, divergence,
            moisture, stability, and cloud phase legible and accessible.
          </p>
        </div>
        <div className="badge">Accessibility first</div>
      </div>

      <div className="colormap-grid" aria-label="Colormap and icon recommendations">
        {colorSystem.map((entry) => (
          <div key={entry.name} className="colormap-card" role="group" aria-label={`${entry.name} colormap`}>
            <div className="colormap-card-header">
              <span className="colormap-icon" aria-hidden="true">{entry.icon}</span>
              <div>
                <h3>{entry.name}</h3>
                <p>{entry.description}</p>
              </div>
            </div>
            <div className="swatch-row">
              {entry.palette.map((color) => (
                <span
                  key={color}
                  className="swatch"
                  style={{ backgroundColor: color }}
                  aria-label={`${entry.name} color ${color}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="flow-mode-panel">
        <h3>Flow visualization modes</h3>
        <p>Layer streamlines, streaklines, ribbons, and PV/humidity isosurfaces for depth-aware diagnostics.</p>
        <div className="flow-mode-grid">
          {flowModes.map((mode) => (
            <label key={mode.name} className="flow-mode-chip">
              <input type="checkbox" defaultChecked />
              <div>
                <strong>{mode.name}</strong>
                <span>{mode.detail}</span>
              </div>
            </label>
          ))}
        </div>
        <Plot
          data={[isoSurface, ribbonTrace, particleTrace, ...streamlineTraces]}
          layout={flowLayout}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>

      <div className="composite-grid">
        <div>
          <h3>Satellite-style composites</h3>
          <p>VIS/IR/WV composites mixed with pseudo-reflectivity to mimic multispectral blends.</p>
          <Plot
            data={satelliteComposite}
            layout={{
              autosize: true,
              height: 360,
              margin: { t: 30, r: 30, b: 50, l: 60 },
              xaxis: { title: 'Longitude' },
              yaxis: { title: 'Latitude' },
              legend: { orientation: 'h' },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)'
            }}
            style={{ width: '100%', height: '100%' }}
            config={{ displayModeBar: false, responsive: true }}
          />
        </div>
        <div>
          <h3>Vertical slices & transects</h3>
          <p>Choose orientation and slice position, then export annotated snapshots with side-by-side comparisons.</p>
          <div className="transect-controls">
            <label>
              Orientation
              <select value={orientation} onChange={(event) => setOrientation(event.target.value as Orientation)}>
                <option value="zonal">Zonal (west-east)</option>
                <option value="meridional">Meridional (south-north)</option>
              </select>
            </label>
            <label className="slider-label">
              Slice position
              <input
                type="range"
                min={0}
                max={100}
                value={slicePosition}
                onChange={(event) => setSlicePosition(Number(event.target.value))}
              />
              <span className="slider-value">{slicePosition}% across grid</span>
            </label>
            <button type="button" className="ghost-button" onClick={handleExport}>
              Export snapshot
            </button>
          </div>
          <div className="slice-grid">
            <div>
              <h4>Model slice</h4>
              {transectPlot}
            </div>
            <div>
              <h4>Side-by-side baseline</h4>
              {comparisonPlot}
            </div>
          </div>
          {exportMessage && <p className="export-status">{exportMessage}</p>}
        </div>
      </div>
    </section>
  );
}

export default VisualizationWorkbench;
