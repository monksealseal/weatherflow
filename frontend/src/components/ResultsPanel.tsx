import { ExperimentResult } from '../api/types';
import DatasetStats from './DatasetStats';
import LossChart from './LossChart';
import PredictionViewer from './PredictionViewer';
import SummaryPanel from './SummaryPanel';

interface Props {
  result: ExperimentResult | null;
  loading: boolean;
  hasConfig: boolean;
}

function ResultsPanel({ result, loading, hasConfig }: Props): JSX.Element {
  if (loading && !result) {
    return (
      <section className="section-card">
        <h2>Running experiment…</h2>
        <p>Training the WeatherFlow model with the selected configuration.</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="section-card">
        <h2>Experiment results</h2>
        <p>
          {hasConfig
            ? 'Adjust the configuration on the left and click “Run experiment” to generate results.'
            : 'Select at least one variable and pressure level to begin.'}
        </p>
      </section>
    );
  }

  return (
    <div className="results-stack">
      <SummaryPanel result={result} />
      <LossChart train={result.metrics.train} validation={result.validation.metrics} />
      <PredictionViewer prediction={result.prediction} />
      <DatasetStats stats={result.datasetSummary.channelStats} />
    </div>
  );
}

export default ResultsPanel;
