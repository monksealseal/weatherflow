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
      return <PlaceholderView title="🏃 Basic Training" description="Simple training configuration" />;
    }
    if (currentPath === '/training/advanced') {
      return <PlaceholderView title="🚀 Advanced Training" description="Physics losses and advanced options" />;
    }
    if (currentPath === '/training/distributed') {
      return <PlaceholderView title="🌐 Distributed Training" description="Multi-GPU training (Coming Soon)" />;
    }
    if (currentPath === '/training/tuning') {
      return <PlaceholderView title="🎛️ Hyperparameter Tuning" description="Automated hyperparameter search" />;
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
      return <PlaceholderView title="📊 Metrics Dashboard" description="Comprehensive evaluation metrics" />;
    }
    if (currentPath === '/evaluation/skill-scores') {
      return <PlaceholderView title="🎯 Skill Scores" description="ACC, RMSE, and verification metrics" />;
    }
    if (currentPath === '/evaluation/spatial') {
      return <PlaceholderView title="🗺️ Spatial Analysis" description="Regional error patterns" />;
    }
    if (currentPath === '/evaluation/spectra') {
      return <PlaceholderView title="📉 Energy Spectra" description="Spectral energy analysis" />;
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
