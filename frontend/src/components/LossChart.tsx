import Plot from 'react-plotly.js';
import { MetricEntry, ValidationMetricEntry } from '../api/types';

interface Props {
  train: MetricEntry[];
  validation: ValidationMetricEntry[];
}

function LossChart({ train, validation }: Props): JSX.Element {
  const trainEpochs = train.map((metric) => metric.epoch);
  const valEpochs = validation.map((metric) => metric.epoch);

  return (
    <section className="section-card">
      <h2>Training history</h2>
      <Plot
        data={[
          {
            x: trainEpochs,
            y: train.map((metric) => metric.loss),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Train total'
          },
          {
            x: trainEpochs,
            y: train.map((metric) => metric.flowLoss),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Train flow',
            line: { dash: 'dot' }
          },
          {
            x: valEpochs,
            y: validation.map((metric) => metric.valLoss),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Validation total'
          },
          {
            x: valEpochs,
            y: validation.map((metric) => metric.valFlowLoss),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Validation flow',
            line: { dash: 'dot' }
          }
        ]}
        layout={{
          autosize: true,
          margin: { t: 30, r: 20, b: 40, l: 50 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          height: 320,
          legend: { orientation: 'h', y: -0.2 },
          xaxis: { title: 'Epoch' },
          yaxis: { title: 'Loss', rangemode: 'tozero' }
        }}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: false, responsive: true }}
      />
    </section>
  );
}

export default LossChart;
