import { useState } from 'react';
import NavigationSidebar from './components/NavigationSidebar';
import ExperimentHistory from './components/ExperimentHistory';
import { ExperimentRecord } from './utils/experimentTracker';
import './AppNew.css';

// Import real view components
import ModelZooView from './components/views/ModelZooView';
import ERA5BrowserView from './components/views/ERA5BrowserView';
import RenewableEnergyView from './components/views/RenewableEnergyView';
import TutorialsView from './components/views/TutorialsView';
import AtmosphericDynamicsView from './components/views/AtmosphericDynamicsView';
import ExtremeEventsView from './components/views/ExtremeEventsView';
import PhysicsPrimerView from './components/views/PhysicsPrimerView';
import InteractiveNotebooksView from './components/views/InteractiveNotebooksView';
import FlowMatchingView from './components/views/FlowMatchingView';
import TrainingWorkflowsView from './components/views/TrainingWorkflowsView';
import EvaluationView from './components/views/EvaluationView';

// Placeholder components for different views
function DashboardView() {
  return (
    <div className="view-container">
      <div className="view-header">
        <h1>🏠 Dashboard</h1>
        <p className="view-subtitle">Welcome to WeatherFlow - Your comprehensive weather prediction platform</p>
      </div>
      <div className="dashboard-grid">
        <div className="dashboard-card">
          <h3>🧪 Quick Start</h3>
          <p>Run your first experiment with pre-configured settings</p>
          <button className="card-action">Start Experiment</button>
        </div>
        <div className="dashboard-card">
          <h3>🏛️ Model Zoo</h3>
          <p>Browse and download pre-trained models</p>
          <button className="card-action">Browse Models</button>
        </div>
        <div className="dashboard-card">
          <h3>📊 Recent Experiments</h3>
          <p>View your latest experiment results</p>
          <button className="card-action">View History</button>
        </div>
        <div className="dashboard-card">
          <h3>🎓 Learn</h3>
          <p>Interactive tutorials and educational resources</p>
          <button className="card-action">Start Learning</button>
        </div>
      </div>

      <section className="readiness-section">
        <div className="readiness-header">
          <h2>🧭 Feature readiness map</h2>
          <p>See what runs instantly versus where training or checkpoints unlock extra depth.</p>
        </div>
        <div className="readiness-grid">
          <div className="readiness-card">
            <h3>Works without training</h3>
            <ul>
              <li>Renewable energy converters (<code>applications/renewable_energy/*</code>)</li>
              <li>Extreme event detectors (<code>applications/extreme_event_analysis/detectors.py</code>)</li>
              <li>ERA5 data exploration and persistence baselines (<code>weatherflow/data/era5.py</code>)</li>
              <li>Flow visualisation with toy tensors (<code>weatherflow/models/flow_matching.py</code>)</li>
              <li>FastAPI smoke tests (<code>run_experiment.py</code> → <code>weatherflow/server/app.py</code>)</li>
            </ul>
          </div>
          <div className="readiness-card">
            <h3>Benefits from checkpoints</h3>
            <ul>
              <li>Model Zoo cards and Flow Matching demos (<code>model_zoo/train_model.py</code>)</li>
              <li>Experiment visualizations powered by <code>WeatherFlowODE</code></li>
              <li>Evaluation dashboards using <code>weatherflow/training/metrics.py</code></li>
              <li>Remote/Hugging Face runs for heavier training jobs</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="integration-section">
        <div className="readiness-header">
          <h2>🔌 Wire navigation items to existing Python</h2>
          <p>Keep the UI useful even when you do not want to start a long training job.</p>
        </div>
        <div className="integration-grid">
          <div className="integration-card">
            <div className="integration-card-header">
              <h3>Model Zoo & Flow Matching</h3>
              <span className="pill">Checkpoints preferred</span>
            </div>
            <ul>
              <li><strong>Backend files:</strong> <code>model_zoo/train_model.py</code>, <code>model_zoo/download_model.py</code>, <code>weatherflow/models/flow_matching.py</code></li>
              <li><strong>Instant path:</strong> Load an archived checkpoint with <code>download_model.py</code> and render configs/metadata even before inference runs.</li>
              <li><strong>Training path:</strong> Push <code>train_model.py</code> to a remote runner (Hugging Face, cluster) and drop the *.pt + model_card.json back into <code>model_zoo/</code>.</li>
            </ul>
          </div>
          <div className="integration-card">
            <div className="integration-card-header">
              <h3>Data + Detectors</h3>
              <span className="pill">No training needed</span>
            </div>
            <ul>
              <li><strong>Backend files:</strong> <code>weatherflow/data/era5.py</code>, <code>applications/extreme_event_analysis/detectors.py</code>, <code>applications/renewable_energy/*</code></li>
              <li><strong>UI targets:</strong> ERA5 Browser, Extreme Events, Renewable Energy, Evaluation baselines.</li>
              <li><strong>Workflow:</strong> Use ERA5 loaders to feed detectors or calculators, then render maps and tables without waiting for ML checkpoints.</li>
            </ul>
          </div>
          <div className="integration-card">
            <div className="integration-card-header">
              <h3>Experiment API & Metrics</h3>
              <span className="pill">Short runs OK</span>
            </div>
            <ul>
              <li><strong>Backend files:</strong> <code>run_experiment.py</code>, <code>weatherflow/server/app.py</code>, <code>weatherflow/training/metrics.py</code></li>
              <li><strong>Instant path:</strong> Trigger the FastAPI smoke run to populate Experiment History and Evaluation cards with sample metrics.</li>
              <li><strong>Training path:</strong> Swap the endpoint to a remote host that is training longer jobs; the UI ingests the same payloads.</li>
            </ul>
          </div>
          <div className="integration-card">
            <div className="integration-card-header">
              <h3>Education & Notebooks</h3>
              <span className="pill">Reuse scripts</span>
            </div>
            <ul>
              <li><strong>Backend files:</strong> <code>examples/flow_matching/simple_example.py</code>, <code>examples/flow_matching/era5_strict_training_loop.py</code>, <code>weatherflow/education/graduate_tool.py</code></li>
              <li><strong>Instant path:</strong> Surface the prewritten examples and notebook runners; users can run locally without editing Python.</li>
              <li><strong>Training path:</strong> Point the notebook/gallery to a pre-trained checkpoint to unlock richer visualizations.</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

function PlaceholderView({ title, description }: { title: string; description?: string }) {
  return (
    <div className="view-container">
      <div className="view-header">
        <h1>{title}</h1>
        {description && <p className="view-subtitle">{description}</p>}
      </div>
      <div className="placeholder-content">
        <div className="placeholder-box">
          <p>🚧 This feature is under development</p>
          <p className="placeholder-subtitle">
            Check back soon for {title.toLowerCase()} functionality
          </p>
        </div>
      </div>
    </div>
  );
}

export default function AppNew(): JSX.Element {
  const [currentPath, setCurrentPath] = useState('/');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentRecord | null>(null);

  const handleNavigate = (path: string) => {
    setCurrentPath(path);
    setSelectedExperiment(null);
  };

  const renderView = () => {
    // Dashboard
    if (currentPath === '/' || currentPath === '/dashboard') {
      return <DashboardView />;
    }

    // Experiment views
    if (currentPath === '/experiments/new') {
      return <PlaceholderView title="🧪 New Experiment" description="Configure and launch a new experiment" />;
    }
    if (currentPath === '/experiments/history') {
      return <ExperimentHistory onSelectExperiment={setSelectedExperiment} />;
    }
    if (currentPath === '/experiments/compare') {
      return <PlaceholderView title="⚖️ Compare Experiments" description="Side-by-side experiment comparison" />;
    }
    if (currentPath === '/experiments/ablation') {
      return <PlaceholderView title="🔬 Ablation Study" description="Systematic component analysis" />;
    }

    // Model views
    if (currentPath === '/models/zoo') {
      return <ModelZooView />;
    }
    if (currentPath === '/models/flow-matching') {
      return <FlowMatchingView />;
    }
    if (currentPath === '/models/icosahedral') {
      return <PlaceholderView title="⚽ Icosahedral Grid" description="Spherical mesh for global predictions" />;
    }
    if (currentPath === '/models/physics-guided') {
      return <PlaceholderView title="⚗️ Physics-Guided Models" description="Neural networks with physical constraints" />;
    }
    if (currentPath === '/models/stochastic') {
      return <PlaceholderView title="🎲 Stochastic Models" description="Ensemble forecasting and uncertainty" />;
    }

    // Data views
    if (currentPath === '/data/era5') {
      return <ERA5BrowserView />;
    }
    if (currentPath === '/data/weatherbench2') {
      return <PlaceholderView title="📈 WeatherBench2" description="Benchmark datasets for model evaluation" />;
    }
    if (currentPath === '/data/preprocessing') {
      return <PlaceholderView title="⚙️ Data Preprocessing" description="Configure data pipelines" />;
    }
    if (currentPath === '/data/synthetic') {
      return <PlaceholderView title="🎨 Synthetic Data" description="Generate training data" />;
    }

    // Training views
    if (currentPath === '/training/basic') {
      return <TrainingWorkflowsView mode="basic" />;
    }
    if (currentPath === '/training/advanced') {
      return <TrainingWorkflowsView mode="advanced" />;
    }
    if (currentPath === '/training/distributed') {
      return <TrainingWorkflowsView mode="distributed" />;
    }
    if (currentPath === '/training/tuning') {
      return <TrainingWorkflowsView mode="tuning" />;
    }

    // Visualization views
    if (currentPath === '/visualization/fields') {
      return <PlaceholderView title="🗺️ Field Viewer" description="Visualize weather fields" />;
    }
    if (currentPath === '/visualization/flows') {
      return <PlaceholderView title="🌊 Flow Visualization" description="Vector fields and trajectories" />;
    }
    if (currentPath === '/visualization/skewt') {
      return <PlaceholderView title="📉 SkewT Diagrams" description="Atmospheric soundings" />;
    }
    if (currentPath === '/visualization/3d') {
      return <PlaceholderView title="🎬 3D Rendering" description="Interactive 3D atmosphere" />;
    }
    if (currentPath === '/visualization/clouds') {
      return <PlaceholderView title="☁️ Cloud Rendering" description="Volumetric cloud visualization" />;
    }

    // Application views
    if (currentPath === '/applications/renewable-energy') {
      return <RenewableEnergyView />;
    }
    if (currentPath === '/applications/extreme-events') {
      return <ExtremeEventsView />;
    }
    if (currentPath === '/applications/climate') {
      return <PlaceholderView title="🌡️ Climate Analysis" description="Long-term trends and patterns" />;
    }
    if (currentPath === '/applications/aviation') {
      return <PlaceholderView title="✈️ Aviation Weather" description="Flight planning and turbulence (Coming Soon)" />;
    }

    // Education views
    if (currentPath === '/education/dynamics') {
      return <AtmosphericDynamicsView />;
    }
    if (currentPath === '/education/tutorials') {
      return <TutorialsView />;
    }
    if (currentPath === '/education/notebooks') {
      return <InteractiveNotebooksView />;
    }
    if (currentPath === '/education/physics') {
      return <PhysicsPrimerView />;
    }

    // Evaluation views
    if (currentPath === '/evaluation/dashboard') {
      return <EvaluationView mode="dashboard" />;
    }
    if (currentPath === '/evaluation/skill-scores') {
      return <EvaluationView mode="skill" />;
    }
    if (currentPath === '/evaluation/spatial') {
      return <EvaluationView mode="spatial" />;
    }
    if (currentPath === '/evaluation/spectra') {
      return <EvaluationView mode="spectra" />;
    }

    // Settings views
    if (currentPath === '/settings/api') {
      return <PlaceholderView title="🔌 API Configuration" description="Configure API endpoint" />;
    }
    if (currentPath === '/settings/preferences') {
      return <PlaceholderView title="🎨 Preferences" description="Customize your experience" />;
    }
    if (currentPath === '/settings/data') {
      return <PlaceholderView title="💾 Data Management" description="Manage cached data" />;
    }
    if (currentPath === '/settings/export-import') {
      return <PlaceholderView title="📦 Export/Import" description="Backup and restore" />;
    }

    return <DashboardView />;
  };

  return (
    <div className="app-new">
      <NavigationSidebar
        currentPath={currentPath}
        onNavigate={handleNavigate}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <main className={`app-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        {renderView()}
      </main>
      {selectedExperiment && (
        <div className="modal-overlay" onClick={() => setSelectedExperiment(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedExperiment.name}</h2>
              <button onClick={() => setSelectedExperiment(null)}>✕</button>
            </div>
            <div className="modal-body">
              <p><strong>Status:</strong> {selectedExperiment.status}</p>
              <p><strong>Created:</strong> {new Date(selectedExperiment.timestamp).toLocaleString()}</p>
              {selectedExperiment.description && (
                <p><strong>Description:</strong> {selectedExperiment.description}</p>
              )}
              {selectedExperiment.duration && (
                <p><strong>Duration:</strong> {(selectedExperiment.duration / 1000).toFixed(2)}s</p>
              )}
              <pre>{JSON.stringify(selectedExperiment.config, null, 2)}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
