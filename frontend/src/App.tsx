import { useEffect, useMemo, useState } from 'react';
import './App.css';
import {
  DatasetConfig,
  ExperimentConfig,
  ExperimentResult,
  ModelConfig,
  ServerOptions,
  TrainingConfig
} from './api/types';
import { fetchOptions, runExperiment } from './api/client';
import DatasetConfigurator from './components/DatasetConfigurator';
import ModelConfigurator from './components/ModelConfigurator';
import TrainingConfigurator from './components/TrainingConfigurator';
import ResultsPanel from './components/ResultsPanel';
import LoadingOverlay from './components/LoadingOverlay';
import ErrorNotice from './components/ErrorNotice';
import ExplorationPlayground from './components/ExplorationPlayground';

const defaultModelConfig: ModelConfig = {
  hiddenDim: 96,
  nLayers: 3,
  useAttention: true,
  physicsInformed: true
};

const createDefaultDatasetConfig = (options: ServerOptions): DatasetConfig => ({
  variables: options.variables.slice(0, 2),
  pressureLevels: [options.pressureLevels[0]],
  gridSize: options.gridSizes[0] ?? { lat: 16, lon: 32 },
  trainSamples: 48,
  valSamples: 16
});

const createDefaultTrainingConfig = (options: ServerOptions): TrainingConfig => ({
  epochs: Math.min(2, options.maxEpochs),
  batchSize: 8,
  learningRate: 5e-4,
  solverMethod: options.solverMethods[0] ?? 'dopri5',
  timeSteps: 5,
  lossType: options.lossTypes[0] ?? 'mse',
  seed: 42,
  dynamicsScale: 0.15
});

function App(): JSX.Element {
  const [options, setOptions] = useState<ServerOptions | null>(null);
  const [datasetConfig, setDatasetConfig] = useState<DatasetConfig | null>(null);
  const [modelConfig, setModelConfig] = useState<ModelConfig>(defaultModelConfig);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig | null>(null);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchOptions()
      .then((data) => {
        setOptions(data);
        setDatasetConfig((current) => current ?? createDefaultDatasetConfig(data));
        setTrainingConfig((current) => current ?? createDefaultTrainingConfig(data));
      })
      .catch((err: Error) => {
        setError(`Failed to load server options: ${err.message}`);
      });
  }, []);

  const canRunExperiment = useMemo(
    () => Boolean(options && datasetConfig && trainingConfig && datasetConfig.variables.length > 0),
    [options, datasetConfig, trainingConfig]
  );

  const handleRunExperiment = async () => {
    if (!options || !datasetConfig || !trainingConfig) {
      return;
    }

    const config: ExperimentConfig = {
      dataset: datasetConfig,
      model: modelConfig,
      training: trainingConfig
    };

    setLoading(true);
    setError(null);

    try {
      const response = await runExperiment(config);
      setResult(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`Experiment failed: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (!options) {
      return;
    }
    setDatasetConfig(createDefaultDatasetConfig(options));
    setModelConfig(defaultModelConfig);
    setTrainingConfig(createDefaultTrainingConfig(options));
    setResult(null);
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>WeatherFlow Studio</h1>
          <p className="subtitle">Configure, train, and evaluate WeatherFlow models with an interactive dashboard.</p>
        </div>
        <div className="header-actions">
          <button type="button" onClick={handleReset} disabled={!options} className="ghost-button">
            Reset configuration
          </button>
        </div>
      </header>

      {error && <ErrorNotice message={error} />}

      <main className="app-main">
        <section className="config-column">
          <DatasetConfigurator
            options={options}
            value={datasetConfig}
            onChange={setDatasetConfig}
          />
          <ModelConfigurator value={modelConfig} onChange={setModelConfig} />
          <TrainingConfigurator
            options={options}
            value={trainingConfig}
            onChange={setTrainingConfig}
          />
          <div className="actions">
            <button
              type="button"
              className="primary-button"
              onClick={handleRunExperiment}
              disabled={!canRunExperiment || loading}
            >
              {loading ? 'Running...' : 'Run experiment'}
            </button>
          </div>
        </section>
        <section className="results-column">
          <ResultsPanel
            result={result}
            loading={loading}
            hasConfig={Boolean(canRunExperiment)}
          />
          <ExplorationPlayground result={result} />
        </section>
      </main>

      {loading && <LoadingOverlay message="Running WeatherFlow experiment..." />}
    </div>
  );
}

export default App;
