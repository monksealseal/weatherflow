import './TrainingWorkflowsView.css';

type TrainingMode = 'basic' | 'advanced' | 'tuning' | 'distributed';

interface WorkflowCard {
  title: string;
  description: string;
  files: string[];
  badge?: string;
}

const MODE_COPY: Record<TrainingMode, { icon: string; title: string; subtitle: string }> = {
  basic: {
    icon: '🏃',
    title: 'Training Playbooks',
    subtitle: 'Decide when to reuse checkpoints versus running a short training job'
  },
  advanced: {
    icon: '🚀',
    title: 'Advanced Training',
    subtitle: 'Physics terms, mixed precision, and custom dataloaders without touching new Python code'
  },
  tuning: {
    icon: '🎛️',
    title: 'Hyperparameter Tuning',
    subtitle: 'Lightweight sweeps on top of existing scripts and metrics utilities'
  },
  distributed: {
    icon: '🌐',
    title: 'Remote / Distributed Plan',
    subtitle: 'Use external compute (Hugging Face, clusters) while keeping WeatherFlow scripts unchanged'
  }
};

const ZERO_TRAINING_FEATURES: WorkflowCard[] = [
  {
    title: 'Renewable energy converters',
    description:
      'Wind and solar power outputs come directly from deterministic converters—no ML weights required. Great for demonstrating Applications → Renewable Energy even without GPUs.',
    files: [
      'applications/renewable_energy/wind_power.py',
      'applications/renewable_energy/solar_power.py'
    ],
    badge: 'Runs instantly'
  },
  {
    title: 'Extreme event detectors',
    description:
      'Atmospheric river, heatwave, and tropical cyclone logic is rule-based. The detectors can wrap model outputs later, but they already work on reanalysis or sample tensors.',
    files: ['applications/extreme_event_analysis/detectors.py'],
    badge: 'Model optional'
  },
  {
    title: 'Experiment API smoke tests',
    description:
      'The FastAPI service falls back to short, two-epoch training runs for demos. Trigger it from the website or via run_experiment.py without provisioning long jobs.',
    files: ['weatherflow/server/app.py', 'run_experiment.py']
  },
  {
    title: 'ERA5 data exploration',
    description:
      'Dataset utilities can download, normalize, and visualize ERA5 slices. Use them to populate visualizations even before you have a checkpoint.',
    files: ['weatherflow/data/era5.py']
  }
];

const TRAINING_BUILDING_BLOCKS = [
  {
    title: 'Model Zoo CLI',
    description: 'Turn a predefined recipe into a checkpoint and model card.',
    code: `python model_zoo/train_model.py z500_3day \\
  --output-dir model_zoo/global_forecasting/z500_3day`,
    files: ['model_zoo/train_model.py', 'model_zoo/README.md']
  },
  {
    title: 'Load or resume checkpoints',
    description: 'Prefer reusing checkpoints when possible—training is optional for many pages.',
    code: `python model_zoo/download_model.py wf_global_multivariable_v2 --output-dir ./model_zoo/global_forecasting/multivariable

import torch
from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE

checkpoint = torch.load("model_zoo/global_forecasting/z500_3day/wf_z500_3day_v1.pt", map_location="cpu")
model = WeatherFlowMatch(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
forecaster = WeatherFlowODE(model)
forecast = forecaster(x0, times=torch.linspace(0, 1, 5))`,
    files: ['model_zoo/download_model.py', 'weatherflow/models/flow_matching.py']
  },
  {
    title: 'Custom training loops',
    description:
      'If you need custom dataloaders or physics losses, start from the existing trainer utilities rather than new code.',
    code: `from weatherflow.training.flow_trainer import FlowTrainer
from weatherflow.training.metrics import rmse, mae
from weatherflow.data.era5 import create_data_loaders
from weatherflow.models import WeatherFlowMatch

train_loader, val_loader = create_data_loaders(
    root_dir="/path/to/era5",
    train_years=[2016, 2017],
    val_years=[2018],
    variables=["z", "t"],
    levels=[500, 850],
    batch_size=8,
)

model = WeatherFlowMatch(input_channels=4, hidden_dim=128, n_layers=4, physics_informed=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = FlowTrainer(model=model, optimizer=optimizer, checkpoint_dir="checkpoints", use_amp=True)

train_metrics = trainer.train_epoch(train_loader)
val_metrics = trainer.validate(val_loader)
print({"rmse": val_metrics["val_rmse"], "mae": val_metrics["val_mae"]})`,
    files: [
      'weatherflow/training/flow_trainer.py',
      'weatherflow/training/metrics.py',
      'examples/flow_matching/era5_strict_training_loop.py'
    ]
  }
];

const WEBSITE_INTEGRATIONS = [
  {
    title: 'Model-aware sections',
    items: [
      'Model Zoo and Flow Matching pages expect checkpoints but can display configs immediately.',
      'Renewable Energy and Extreme Events stay fully interactive without ML weights.'
    ]
  },
  {
    title: 'Experiment runs',
    items: [
      'Use run_experiment.py to hit the FastAPI backend with the same payloads the site will send.',
      'If training is too heavy locally, point the API to a remote host or pre-download checkpoints.'
    ]
  },
  {
    title: 'Remote / Hugging Face',
    items: [
      'Train via model_zoo/train_model.py on a cloud runner and store the resulting *.pt and model_card.json.',
      'Serve the checkpoint back to the site through model_zoo/download_model.py or a Hugging Face artifact URL.'
    ]
  }
];

export default function TrainingWorkflowsView({ mode }: { mode: TrainingMode }) {
  const copy = MODE_COPY[mode];

  return (
    <div className="view-container training-workflows">
      <div className="view-header">
        <h1>{copy.icon} {copy.title}</h1>
        <p className="view-subtitle">{copy.subtitle}</p>
      </div>

      <div className="training-banner">
        <div className="banner-icon">🧭</div>
        <div className="banner-text">
          <h3>Choose the right path</h3>
          <p>
            WeatherFlow already ships working Python flows. Use them directly from the website,
            reuse existing checkpoints when possible, and only launch training when you need new skill
            or variables. The guidance below maps every website section to the scripts that power it.
          </p>
        </div>
      </div>

      <section className="workflow-section">
        <h2>🚫 Training Not Required</h2>
        <div className="workflow-grid">
          {ZERO_TRAINING_FEATURES.map((card) => (
            <div key={card.title} className="workflow-card">
              <div className="card-header">
                <h3>{card.title}</h3>
                {card.badge && <span className="pill">{card.badge}</span>}
              </div>
              <p>{card.description}</p>
              <div className="file-list">
                {card.files.map((file) => (
                  <code key={file}>{file}</code>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="workflow-section">
        <h2>✅ Prefer Reuse First</h2>
        <div className="workflow-grid">
          {TRAINING_BUILDING_BLOCKS.map((block) => (
            <div key={block.title} className="workflow-card">
              <h3>{block.title}</h3>
              <p>{block.description}</p>
              <pre><code>{block.code}</code></pre>
              <div className="file-list">
                {block.files.map((file) => (
                  <code key={file}>{file}</code>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="workflow-section">
        <h2>📌 Where This Shows Up on the Site</h2>
        <div className="integration-grid">
          {WEBSITE_INTEGRATIONS.map((item) => (
            <div key={item.title} className="integration-card">
              <h3>{item.title}</h3>
              <ul>
                {item.items.map((bullet) => (
                  <li key={bullet}>{bullet}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="workflow-section">
        <h2>🚀 If You Do Need Training</h2>
        <div className="next-steps">
          <div className="step">
            <h4>1) Start from an example</h4>
            <p>examples/flow_matching/era5_strict_training_loop.py mirrors our production defaults.</p>
          </div>
          <div className="step">
            <h4>2) Keep runs short for demos</h4>
            <p>FlowTrainer supports AMP, gradient clipping, and EMA—use a handful of epochs to validate the UI.</p>
          </div>
          <div className="step">
            <h4>3) Offload heavy jobs</h4>
            <p>Run model_zoo/train_model.py on Hugging Face or another runner and publish the *.pt + model_card.json back to the repo.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
