import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { PredictionResult } from '../api/types';

interface Props {
  prediction: PredictionResult;
}

function PredictionViewer({ prediction }: Props): JSX.Element {
  const [channelIndex, setChannelIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  const channels = prediction.channels;
  const selectedChannel = useMemo(
    () => channels[Math.min(channelIndex, Math.max(channels.length - 1, 0))] ?? null,
    [channelIndex, channels]
  );

  useEffect(() => {
    if (!selectedChannel) {
      setStepIndex(0);
      return;
    }
    setStepIndex((index) => Math.min(index, selectedChannel.trajectory.length - 1));
  }, [channelIndex, selectedChannel]);

  const sliderMax = selectedChannel ? Math.max(selectedChannel.trajectory.length - 1, 0) : 0;
  const safeStepIndex = Math.min(stepIndex, sliderMax);

  const activeStep = useMemo(
    () => (selectedChannel ? selectedChannel.trajectory[safeStepIndex] : null),
    [selectedChannel, safeStepIndex]
  );

  const timeValue =
    prediction.times.length > 0
      ? prediction.times[Math.min(safeStepIndex, prediction.times.length - 1)]
      : undefined;

  const heatmapData = useMemo(
    () => [
      {
        z: activeStep?.data ?? [],
        type: 'heatmap',
        colorscale: 'RdBu',
        reversescale: true,
        zsmooth: 'best'
      }
    ],
    [activeStep]
  );

  if (!selectedChannel || !activeStep) {
    return (
      <section className="section-card">
        <h2>Prediction explorer</h2>
        <p>No prediction data available.</p>
      </section>
    );
  }

  return (
    <section className="section-card">
      <div className="prediction-header">
        <div>
          <h2>Prediction explorer</h2>
          <p>Inspect the generated trajectory for any channel and time step.</p>
        </div>
        <div className="prediction-stats">
          <div>
            <span>RMSE</span>
            <strong>{selectedChannel.rmse.toFixed(4)}</strong>
          </div>
          <div>
            <span>MAE</span>
            <strong>{selectedChannel.mae.toFixed(4)}</strong>
          </div>
          <div>
            <span>Baseline RMSE</span>
            <strong>{selectedChannel.baselineRmse.toFixed(4)}</strong>
          </div>
        </div>
      </div>

      <div className="prediction-controls">
        <label>
          Channel
          <select value={channelIndex} onChange={(event) => setChannelIndex(Number(event.target.value))}>
            {channels.map((channel, index) => (
              <option key={channel.name} value={index}>
                {channel.name.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
        <label className="slider-label">
          Time step
          <input
            type="range"
            min={0}
            max={sliderMax}
            value={stepIndex}
            onChange={(event) => setStepIndex(Number(event.target.value))}
          />
          <span className="slider-value">t={timeValue !== undefined ? timeValue.toFixed(2) : 'n/a'}</span>
        </label>
      </div>

      <div className="heatmap-grid">
        <Plot
          data={heatmapData}
          layout={{
            autosize: true,
            margin: { t: 30, r: 30, b: 40, l: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            height: 360,
            xaxis: { title: 'Longitude' },
            yaxis: { title: 'Latitude' }
          }}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
        <div className="comparison-panels">
          <div>
            <h3>Initial</h3>
            <Plot
              data={[
                {
                  z: selectedChannel.initial,
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  reversescale: true
                }
              ]}
              layout={{
                autosize: true,
                margin: { t: 30, r: 30, b: 40, l: 60 },
                height: 260,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'Lon' },
                yaxis: { title: 'Lat' }
              }}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
          <div>
            <h3>Target</h3>
            <Plot
              data={[
                {
                  z: selectedChannel.target,
                  type: 'heatmap',
                  colorscale: 'RdBu',
                  reversescale: true
                }
              ]}
              layout={{
                autosize: true,
                margin: { t: 30, r: 30, b: 40, l: 60 },
                height: 260,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'Lon' },
                yaxis: { title: 'Lat' }
              }}
              style={{ width: '100%', height: '100%' }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

export default PredictionViewer;
