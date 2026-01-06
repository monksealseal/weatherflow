import './EvaluationView.css';

type EvaluationMode = 'dashboard' | 'skill' | 'spatial' | 'spectra';

const MODE_COPY: Record<EvaluationMode, { icon: string; title: string; subtitle: string }> = {
  dashboard: {
    icon: '📈',
    title: 'Evaluation Dashboard',
    subtitle: 'Understand what each chart on this page needs from the backend'
  },
  skill: {
    icon: '🎯',
    title: 'Skill Scores',
    subtitle: 'ACC, RMSE, MAE, and baselines straight from training/metrics.py'
  },
  spatial: {
    icon: '🗺️',
    title: 'Spatial Analysis',
    subtitle: 'Regional breakdowns for detectors and flow-matching forecasts'
  },
  spectra: {
    icon: '📉',
    title: 'Energy Spectra',
    subtitle: 'Use existing utilities to chart spectral energy without new code'
  }
};

const METRICS = [
  { name: 'rmse', description: 'Root-mean-square error baseline', file: 'weatherflow/training/metrics.py' },
  { name: 'mae', description: 'Mean absolute error for more robust magnitudes', file: 'weatherflow/training/metrics.py' },
  { name: 'energy_ratio', description: 'Conservation-aware ratio of kinetic energy', file: 'weatherflow/training/metrics.py' },
  { name: 'persistence_rmse', description: 'Persistence baseline for quick sanity checks', file: 'weatherflow/training/metrics.py' }
];

const PIPELINES = [
  {
    title: 'FlowTrainer validation',
    description:
      'FlowTrainer already computes flow loss, physics loss, RMSE, MAE, and energy metrics. Wire those outputs directly into the dashboard.',
    files: ['weatherflow/training/flow_trainer.py']
  },
  {
    title: 'WeatherTrainer monitoring',
    description:
      'For physics-heavy runs, WeatherTrainer exposes train/val losses, physics losses, and learning-rate histories that map cleanly onto the Evaluation pages.',
    files: ['weatherflow/training/trainers.py']
  },
  {
    title: 'Experiment API payloads',
    description:
      'run_experiment.py posts to the FastAPI backend and returns validation metrics that can be rendered without additional glue code.',
    files: ['run_experiment.py', 'weatherflow/server/app.py']
  }
];

const ZERO_TRAINING_ANALYTICS = [
  'Renewable Energy calculators expose capacity factor and hourly power without ML inference.',
  'Extreme Event detectors report counts, footprints, and thresholds from detectors.py even when you skip model training.',
  'ERA5 loaders can drive climatology/persistence baselines for charts before checkpoints exist.'
];

const UI_WIRING = [
  {
    title: 'Experiments → History + Evaluation',
    points: [
      'Run `uvicorn weatherflow.server.app:app --port 8000` then `python run_experiment.py` to seed charts in seconds.',
      'Metrics come straight from weatherflow/training/metrics.py (rmse, mae, energy_ratio, persistence).',
      'Switch VITE_API_URL to a remote FastAPI host or Hugging Face Space when local training is too heavy.'
    ]
  },
  {
    title: 'Model-free analytics',
    points: [
      'Renewable Energy calculators push capacity-factor time series into Evaluation without checkpoints.',
      'applications/extreme_event_analysis/detectors.py produces counts/footprints the Spatial tab can plot.',
      'weatherflow/data/era5.py enables climatology + persistence baselines for the Skill Scores view.'
    ]
  },
  {
    title: 'Checkpoints & Flow Matching',
    points: [
      'Download weights with model_zoo/download_model.py, then chart RMSE/MAE via weatherflow/training/metrics.py.',
      'FlowTrainer and WeatherTrainer validation JSON feeds the dashboard cards without extra glue code.',
      'If training remotely, publish metrics JSON next to the checkpoint so the UI can fetch both together.'
    ]
  }
];

export default function EvaluationView({ mode }: { mode: EvaluationMode }) {
  const copy = MODE_COPY[mode];

  return (
    <div className="view-container evaluation-view">
      <div className="view-header">
        <h1>{copy.icon} {copy.title}</h1>
        <p className="view-subtitle">{copy.subtitle}</p>
      </div>

      <div className="evaluation-banner">
        <div className="banner-icon">🧪</div>
        <div>
          <h3>Use the metrics that already exist</h3>
          <p>
            Every evaluation widget on this site can be powered by the small metrics module in
            <code> weatherflow/training/metrics.py</code>. Plug your predictions and targets in—no new
            Python is needed.
          </p>
        </div>
      </div>

      <section className="evaluation-section">
        <h2>🎯 Core Metrics</h2>
        <div className="metric-grid">
          {METRICS.map((metric) => (
            <div key={metric.name} className="metric-card">
              <div className="metric-header">
                <h3>{metric.name}</h3>
                <code>{metric.file}</code>
              </div>
              <p>{metric.description}</p>
            </div>
          ))}
        </div>
        <div className="code-block">
          <pre><code>{`from weatherflow.training import metrics as wf_metrics

rmse = wf_metrics.rmse(pred, target)
mae = wf_metrics.mae(pred, target)
energy = wf_metrics.energy_ratio(pred, target)
persistence = wf_metrics.persistence_rmse(baseline, target)`}</code></pre>
        </div>
      </section>

      <section className="evaluation-section">
        <h2>🧵 End-to-End Pipelines</h2>
        <div className="pipeline-grid">
          {PIPELINES.map((pipeline) => (
            <div key={pipeline.title} className="pipeline-card">
              <h3>{pipeline.title}</h3>
              <p>{pipeline.description}</p>
              <div className="file-list">
                {pipeline.files.map((file) => (
                  <code key={file}>{file}</code>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="code-block">
          <pre><code>{`from weatherflow.training.flow_trainer import FlowTrainer
val_metrics = trainer.validate(val_loader)
print({
    "val_loss": val_metrics["val_loss"],
    "rmse": val_metrics["val_rmse"],
    "mae": val_metrics["val_mae"],
    "energy_ratio": val_metrics["val_energy_ratio"],
})`}</code></pre>
        </div>
      </section>

      <section className="evaluation-section">
        <h2>🧭 Wire UI tabs to Python</h2>
        <div className="pipeline-grid">
          {UI_WIRING.map((item) => (
            <div key={item.title} className="pipeline-card">
              <h3>{item.title}</h3>
              <ul className="wiring-list">
                {item.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="evaluation-section">
        <h2>⚡ Works Without Training</h2>
        <ul className="no-train-list">
          {ZERO_TRAINING_ANALYTICS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="evaluation-section">
        <h2>🔌 How the UI connects</h2>
        <div className="integration-notes">
          <div>
            <h4>Dashboard & Skill Scores</h4>
            <p>Bind FlowTrainer/WeatherTrainer outputs directly to the dashboard cards and skill score charts.</p>
          </div>
          <div>
            <h4>Spatial & Spectra</h4>
            <p>Use ERA5 baselines plus detector outputs to populate maps and spectra even while models are training elsewhere.</p>
          </div>
          <div>
            <h4>Remote training?</h4>
            <p>Push model_zoo/train_model.py jobs to Hugging Face or cluster runners and stream the resulting metrics JSON back here.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
