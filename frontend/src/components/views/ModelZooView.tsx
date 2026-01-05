import { useState } from 'react';
import './ModelZooView.css';

interface ModelCard {
  id: string;
  name: string;
  description: string;
  variables: string[];
  architecture: string;
  params: string;
  status: 'infrastructure-ready' | 'training-required';
  category: string;
}

const MODEL_CATEGORIES = [
  { id: 'global', name: 'Global Forecasting', icon: '🌍' },
  { id: 'regional', name: 'Regional Models', icon: '🗺️' },
  { id: 'extreme', name: 'Extreme Events', icon: '⚠️' },
  { id: 'climate', name: 'Climate Analysis', icon: '🌡️' },
];

const MODEL_CARDS: ModelCard[] = [
  {
    id: 'z500_3day',
    name: 'Z500 3-Day Forecast',
    description: '500 hPa geopotential height prediction for 3-day lead time',
    variables: ['z'],
    architecture: 'WeatherFlowMatch',
    params: '~2.5M parameters',
    status: 'infrastructure-ready',
    category: 'global'
  },
  {
    id: 't850_weekly',
    name: 'T850 Weekly Forecast',
    description: '850 hPa temperature prediction for weekly forecasts',
    variables: ['t'],
    architecture: 'WeatherFlowMatch with attention',
    params: '~3.2M parameters',
    status: 'infrastructure-ready',
    category: 'global'
  },
  {
    id: 'multi_variable',
    name: 'Multi-Variable Global',
    description: 'Comprehensive prediction of multiple atmospheric variables',
    variables: ['z', 't', 'u', 'v', 'q'],
    architecture: 'Physics-guided FlowMatch',
    params: '~8M parameters',
    status: 'infrastructure-ready',
    category: 'global'
  },
  {
    id: 'tropical_cyclones',
    name: 'Tropical Cyclone Tracker',
    description: 'Track and intensity prediction for tropical cyclones',
    variables: ['z', 'u', 'v', 'msl'],
    architecture: 'Icosahedral grid model',
    params: '~5M parameters',
    status: 'infrastructure-ready',
    category: 'extreme'
  },
  {
    id: 'atmospheric_rivers',
    name: 'Atmospheric River Detection',
    description: 'Detection and tracking of atmospheric rivers',
    variables: ['q', 'u', 'v'],
    architecture: 'FlowMatch + detector',
    params: '~3.5M parameters',
    status: 'infrastructure-ready',
    category: 'extreme'
  },
  {
    id: 'seasonal',
    name: 'Seasonal Forecasting',
    description: 'Long-range seasonal climate predictions',
    variables: ['t', 'pr'],
    architecture: 'Stochastic FlowMatch',
    params: '~6M parameters',
    status: 'infrastructure-ready',
    category: 'climate'
  }
];

export default function ModelZooView() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredModels = MODEL_CARDS.filter(model => {
    const matchesCategory = selectedCategory === 'all' || model.category === selectedCategory;
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         model.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="view-container model-zoo-view">
      <div className="view-header">
        <h1>🏛️ Model Zoo</h1>
        <p className="view-subtitle">
          Pre-trained model infrastructure for weather prediction and climate analysis
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">ℹ️</div>
        <div className="banner-content">
          <h3>Infrastructure Ready</h3>
          <p>
            The Model Zoo infrastructure is complete and ready to host pre-trained models.
            Models can be trained using the provided scripts in <code>model_zoo/train_model.py</code>.
            Each model includes comprehensive metadata, performance metrics, and usage examples.
          </p>
        </div>
      </div>

      <div className="info-section">
        <h2>🚀 Quick Playbooks</h2>
        <div className="info-grid">
          <div className="info-card">
            <h3>Use an existing checkpoint (no training)</h3>
            <p>Great for demos where you do not want to launch a long job.</p>
            <pre><code>{`python model_zoo/download_model.py wf_global_multivariable_v2 --output-dir ./model_zoo/global_forecasting/multivariable

import torch
from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
checkpoint = torch.load("model_zoo/global_forecasting/z500_3day/wf_z500_3day_v1.pt", map_location="cpu")
model = WeatherFlowMatch(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
forecast = WeatherFlowODE(model)(x0, times=torch.linspace(0, 1, 5))`}</code></pre>
            <p className="model-description">
              Files: <code>model_zoo/download_model.py</code>, <code>weatherflow/models/flow_matching.py</code>
            </p>
          </div>
          <div className="info-card">
            <h3>Train & publish a Model Zoo entry</h3>
            <p>When you do need training, stick to the existing CLI and model cards.</p>
            <pre><code>{`python model_zoo/train_model.py z500_3day --output-dir model_zoo/global_forecasting/z500_3day
# Outputs: checkpoint (*.pt) + model_card.json ready for the website`}</code></pre>
            <ul>
              <li>Metrics and cards are generated automatically.</li>
              <li>Can run locally or on a remote runner (e.g., Hugging Face) with the same command.</li>
              <li>Drop the artifacts back into <code>model_zoo/</code> to light up the UI.</li>
            </ul>
          </div>
          <div className="info-card">
            <h3>Wire to the Experiment API</h3>
            <p>Feed Model Zoo checkpoints into the FastAPI backend for notebook-style runs.</p>
            <pre><code>{`python run_experiment.py  # posts payload to weatherflow/server/app.py
# API returns metrics and prediction snapshots you can display here`}</code></pre>
            <p className="model-description">
              Keeps website features useful even when training happens elsewhere.
            </p>
          </div>
        </div>
      </div>

      <div className="model-zoo-controls">
        <input
          type="text"
          placeholder="🔍 Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
        />
        
        <div className="category-filters">
          <button
            className={`category-btn ${selectedCategory === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedCategory('all')}
          >
            All Models
          </button>
          {MODEL_CATEGORIES.map(cat => (
            <button
              key={cat.id}
              className={`category-btn ${selectedCategory === cat.id ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat.id)}
            >
              {cat.icon} {cat.name}
            </button>
          ))}
        </div>
      </div>

      <div className="models-grid">
        {filteredModels.map(model => (
          <div key={model.id} className="model-card">
            <div className="model-card-header">
              <h3>{model.name}</h3>
              <span className={`status-badge ${model.status}`}>
                {model.status === 'infrastructure-ready' ? '🏗️ Ready for Training' : '✅ Available'}
              </span>
            </div>
            
            <p className="model-description">{model.description}</p>
            
            <div className="model-details">
              <div className="detail-item">
                <strong>Variables:</strong>
                <div className="variable-tags">
                  {model.variables.map(v => (
                    <span key={v} className="tag">{v}</span>
                  ))}
                </div>
              </div>
              
              <div className="detail-item">
                <strong>Architecture:</strong> {model.architecture}
              </div>
              
              <div className="detail-item">
                <strong>Parameters:</strong> {model.params}
              </div>
            </div>

            <div className="model-actions">
              <button className="action-btn primary" disabled>
                📥 Download (Coming Soon)
              </button>
              <button className="action-btn secondary">
                📖 View Documentation
              </button>
              <button className="action-btn secondary">
                🏋️ Train Model
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="info-section">
        <h2>📚 Model Zoo Infrastructure</h2>
        <div className="info-grid">
          <div className="info-card">
            <h3>🎯 Model Standards</h3>
            <ul>
              <li>Validated on held-out data (2018-2019)</li>
              <li>Benchmarked against climatology and persistence</li>
              <li>Complete training scripts and configurations</li>
              <li>Comprehensive metadata and usage examples</li>
              <li>Physics consistency evaluations</li>
            </ul>
          </div>

          <div className="info-card">
            <h3>📦 Directory Structure</h3>
            <pre><code>{`model_zoo/
├── global_forecasting/
│   ├── z500_3day/
│   ├── t850_weekly/
│   └── multi_variable/
├── regional_forecasting/
│   ├── north_america/
│   ├── europe/
│   └── tropics/
├── extreme_events/
│   ├── tropical_cyclones/
│   ├── atmospheric_rivers/
│   └── heatwaves/
└── climate_analysis/
    ├── seasonal/
    └── subseasonal/`}</code></pre>
          </div>

          <div className="info-card">
            <h3>🔧 Training a Model</h3>
            <pre><code>{`# Train a Z500 3-day model
python model_zoo/train_model.py \\
  --model-type z500_3day \\
  --variables z \\
  --pressure-levels 500 \\
  --train-years 2015 2016 2017 \\
  --val-years 2018 \\
  --epochs 100 \\
  --save-path model_zoo/global_forecasting/z500_3day/`}</code></pre>
          </div>

          <div className="info-card">
            <h3>📊 Loading a Model</h3>
            <pre><code>{`from weatherflow.model_zoo import load_model

# Load a pre-trained model
model, metadata = load_model('wf_z500_3day_v1')

# View model information
print(metadata.summary())

# Run inference
from weatherflow.models import WeatherFlowODE
ode_solver = WeatherFlowODE(model)
prediction = ode_solver(x0, times)`}</code></pre>
          </div>
        </div>
      </div>
    </div>
  );
}
