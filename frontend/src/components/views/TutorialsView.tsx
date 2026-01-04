import './TutorialsView.css';

interface Notebook {
  id: string;
  title: string;
  description: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  topics: string[];
  path: string;
}

const NOTEBOOKS: Notebook[] = [
  {
    id: 'complete-guide',
    title: 'Complete WeatherFlow Guide',
    description: 'Comprehensive introduction to WeatherFlow covering data loading, model training, and prediction visualization',
    difficulty: 'Beginner',
    topics: ['Getting Started', 'Data Loading', 'Training', 'Visualization'],
    path: 'notebooks/complete_guide.ipynb'
  },
  {
    id: 'flow-matching-basics',
    title: 'Flow Matching Basics',
    description: 'Introduction to continuous normalizing flows and flow matching for weather prediction',
    difficulty: 'Beginner',
    topics: ['Flow Matching', 'Theory', 'Simple Examples'],
    path: 'notebooks/flow-matching-basics.ipynb'
  },
  {
    id: 'era5-pipeline',
    title: 'ERA5 Flow Matching Pipeline',
    description: 'End-to-end pipeline for training flow matching models on ERA5 reanalysis data',
    difficulty: 'Intermediate',
    topics: ['ERA5', 'Training Pipeline', 'Best Practices'],
    path: 'notebooks/era5_flow_matching_pipeline.ipynb'
  },
  {
    id: 'model-training',
    title: 'Model Training Deep Dive',
    description: 'Advanced training techniques including physics losses, distributed training, and hyperparameter tuning',
    difficulty: 'Advanced',
    topics: ['Training', 'Physics Constraints', 'Optimization'],
    path: 'notebooks/model-training-notebook.ipynb'
  },
  {
    id: 'prediction-viz',
    title: 'Prediction Visualization',
    description: 'Create publication-quality visualizations of weather predictions including maps, animations, and comparisons',
    difficulty: 'Intermediate',
    topics: ['Visualization', 'Plotting', 'Animation'],
    path: 'notebooks/prediction-visualization-notebook.ipynb'
  },
  {
    id: 'weatherbench-eval',
    title: 'WeatherBench Evaluation',
    description: 'Evaluate model performance using WeatherBench2 metrics and benchmarks',
    difficulty: 'Advanced',
    topics: ['Evaluation', 'Metrics', 'Benchmarking'],
    path: 'notebooks/weatherbench-evaluation-notebook.ipynb'
  },
  {
    id: 'data-exploration',
    title: 'Complete Data Exploration',
    description: 'Explore ERA5 dataset structure, variables, and preprocessing techniques',
    difficulty: 'Beginner',
    topics: ['Data', 'Preprocessing', 'Exploration'],
    path: 'notebooks/complete-data-exploration.ipynb'
  },
  {
    id: 'colab-demo',
    title: 'Google Colab Demo',
    description: 'Run WeatherFlow on Google Colab with free GPUs - no local setup required',
    difficulty: 'Beginner',
    topics: ['Cloud', 'Quick Start', 'Demo'],
    path: 'notebooks/weatherflow_colab_demo.ipynb'
  }
];

const EXAMPLES = [
  {
    id: 'weather-prediction',
    title: 'Weather Prediction Example',
    description: 'Complete example showing ERA5 data loading, model training, and prediction generation',
    file: 'examples/weather_prediction.py'
  },
  {
    id: 'physics-loss',
    title: 'Physics Loss Demo',
    description: 'Demonstrate physics-informed training with conservation laws and geostrophic balance',
    file: 'examples/physics_loss_demo.py'
  },
  {
    id: 'skewt-3d',
    title: 'SkewT 3D Visualizer',
    description: 'Interactive 3D visualization of atmospheric soundings and SkewT diagrams',
    file: 'examples/skewt_3d_visualizer.py'
  },
  {
    id: 'incredible-viz',
    title: 'Incredible Visualizations',
    description: 'Advanced visualization examples including flow fields, cloud rendering, and animations',
    file: 'examples/incredible_visualizations.py'
  },
  {
    id: 'simple-flow',
    title: 'Simple Flow Matching',
    description: 'Minimal example demonstrating flow matching on synthetic data',
    file: 'examples/flow_matching/simple_example.py'
  },
  {
    id: 'strict-training',
    title: 'ERA5 Strict Training Loop',
    description: 'Production-ready training loop with strict data handling and validation',
    file: 'examples/flow_matching/era5_strict_training_loop.py'
  }
];

export default function TutorialsView() {
  return (
    <div className="view-container tutorials-view">
      <div className="view-header">
        <h1>📚 Tutorials & Examples</h1>
        <p className="view-subtitle">
          Learn WeatherFlow through interactive notebooks and example scripts
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🎓</div>
        <div className="banner-content">
          <h3>Learning Resources</h3>
          <p>
            All notebooks and examples are available in the repository. 
            Start with the Complete Guide for a comprehensive introduction, 
            or jump to specific topics based on your needs.
          </p>
        </div>
      </div>

      <section className="notebooks-section">
        <h2>📓 Interactive Notebooks</h2>
        <p className="section-description">
          Jupyter notebooks with step-by-step walkthroughs, explanations, and executable code
        </p>
        
        <div className="difficulty-filters">
          <span className="filter-label">Filter by difficulty:</span>
          <button className="filter-btn active">All</button>
          <button className="filter-btn">Beginner</button>
          <button className="filter-btn">Intermediate</button>
          <button className="filter-btn">Advanced</button>
        </div>

        <div className="notebooks-grid">
          {NOTEBOOKS.map(notebook => (
            <div key={notebook.id} className="notebook-card">
              <div className="notebook-header">
                <h3>{notebook.title}</h3>
                <span className={`difficulty-badge ${notebook.difficulty.toLowerCase()}`}>
                  {notebook.difficulty}
                </span>
              </div>
              <p className="notebook-description">{notebook.description}</p>
              <div className="notebook-topics">
                {notebook.topics.map(topic => (
                  <span key={topic} className="topic-tag">{topic}</span>
                ))}
              </div>
              <div className="notebook-actions">
                <a 
                  href={`https://github.com/monksealseal/weatherflow/blob/main/${notebook.path}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-btn primary"
                >
                  📖 View on GitHub
                </a>
                <a 
                  href={`https://colab.research.google.com/github/monksealseal/weatherflow/blob/main/${notebook.path}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-btn secondary"
                >
                  🚀 Open in Colab
                </a>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="examples-section">
        <h2>💻 Python Examples</h2>
        <p className="section-description">
          Complete, runnable Python scripts demonstrating key features
        </p>

        <div className="examples-grid">
          {EXAMPLES.map(example => (
            <div key={example.id} className="example-card">
              <h3>{example.title}</h3>
              <p>{example.description}</p>
              <code className="file-path">{example.file}</code>
              <a 
                href={`https://github.com/monksealseal/weatherflow/blob/main/${example.file}`}
                target="_blank"
                rel="noopener noreferrer"
                className="view-code-link"
              >
                View Source Code →
              </a>
            </div>
          ))}
        </div>
      </section>

      <section className="quick-start-section">
        <h2>🚀 Quick Start</h2>
        <div className="quick-start-steps">
          <div className="step-card">
            <div className="step-number">1</div>
            <h3>Install WeatherFlow</h3>
            <pre><code>pip install weatherflow</code></pre>
          </div>
          <div className="step-card">
            <div className="step-number">2</div>
            <h3>Clone Repository</h3>
            <pre><code>git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow</code></pre>
          </div>
          <div className="step-card">
            <div className="step-number">3</div>
            <h3>Run Jupyter</h3>
            <pre><code>jupyter notebook notebooks/complete_guide.ipynb</code></pre>
          </div>
        </div>
      </section>

      <section className="topics-section">
        <h2>📖 Topics Covered</h2>
        <div className="topics-grid">
          <div className="topic-card">
            <h3>🌍 Data Loading</h3>
            <ul>
              <li>ERA5 dataset access</li>
              <li>WeatherBench2 integration</li>
              <li>Custom data loaders</li>
              <li>Preprocessing pipelines</li>
            </ul>
          </div>
          <div className="topic-card">
            <h3>🧠 Model Training</h3>
            <ul>
              <li>Flow matching basics</li>
              <li>Physics-informed losses</li>
              <li>Distributed training</li>
              <li>Hyperparameter tuning</li>
            </ul>
          </div>
          <div className="topic-card">
            <h3>📊 Visualization</h3>
            <ul>
              <li>Weather field plotting</li>
              <li>Flow vector visualization</li>
              <li>SkewT diagrams</li>
              <li>3D atmosphere rendering</li>
            </ul>
          </div>
          <div className="topic-card">
            <h3>📈 Evaluation</h3>
            <ul>
              <li>WeatherBench metrics</li>
              <li>Skill scores (ACC, RMSE)</li>
              <li>Spatial error analysis</li>
              <li>Energy spectra</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="resources-section">
        <h2>🔗 Additional Resources</h2>
        <div className="resources-grid">
          <a href="https://github.com/monksealseal/weatherflow" className="resource-link" target="_blank" rel="noopener noreferrer">
            📦 GitHub Repository
          </a>
          <a href="https://github.com/monksealseal/weatherflow/tree/main/docs" className="resource-link" target="_blank" rel="noopener noreferrer">
            📘 Documentation
          </a>
          <a href="https://github.com/monksealseal/weatherflow/issues" className="resource-link" target="_blank" rel="noopener noreferrer">
            💬 Community Forum
          </a>
          <a href="https://github.com/monksealseal/weatherflow/blob/main/CONTRIBUTING.md" className="resource-link" target="_blank" rel="noopener noreferrer">
            🤝 Contributing Guide
          </a>
        </div>
      </section>
    </div>
  );
}
