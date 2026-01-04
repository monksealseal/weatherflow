import './InteractiveNotebooksView.css';

interface NotebookInfo {
  title: string;
  path: string;
  description: string;
  topics: string[];
}

const NOTEBOOKS: NotebookInfo[] = [
  {
    title: 'Complete WeatherFlow Guide',
    path: 'notebooks/complete_guide.ipynb',
    description: 'Comprehensive introduction covering data loading, model training, prediction, and visualization',
    topics: ['Getting Started', 'Full Pipeline', 'Best Practices']
  },
  {
    title: 'Flow Matching Basics',
    path: 'notebooks/flow-matching-basics.ipynb',
    description: 'Introduction to continuous normalizing flows and flow matching for weather prediction',
    topics: ['Theory', 'Flow Matching', 'Mathematics']
  },
  {
    title: 'ERA5 Flow Matching Pipeline',
    path: 'notebooks/era5_flow_matching_pipeline.ipynb',
    description: 'End-to-end pipeline for training on ERA5 data with best practices and evaluation',
    topics: ['ERA5', 'Training', 'Production']
  },
  {
    title: 'Model Training Notebook',
    path: 'notebooks/model-training-notebook.ipynb',
    description: 'Advanced training techniques, physics losses, and hyperparameter optimization',
    topics: ['Training', 'Physics Losses', 'Optimization']
  },
  {
    title: 'Prediction Visualization',
    path: 'notebooks/prediction-visualization-notebook.ipynb',
    description: 'Create publication-quality visualizations of predictions, comparisons, and animations',
    topics: ['Visualization', 'Plotting', 'Publishing']
  },
  {
    title: 'WeatherBench2 Evaluation',
    path: 'notebooks/weatherbench-evaluation-notebook.ipynb',
    description: 'Evaluate models using WeatherBench2 metrics and compare against baselines',
    topics: ['Evaluation', 'Benchmarking', 'Metrics']
  },
  {
    title: 'Complete Data Exploration',
    path: 'notebooks/complete-data-exploration.ipynb',
    description: 'Explore ERA5 dataset structure, variables, statistics, and preprocessing',
    topics: ['Data', 'Exploration', 'Preprocessing']
  },
  {
    title: 'Google Colab Demo',
    path: 'notebooks/weatherflow_colab_demo.ipynb',
    description: 'Run WeatherFlow on Google Colab with free GPUs - no local setup required',
    topics: ['Cloud', 'Colab', 'Quick Start']
  }
];

export default function InteractiveNotebooksView() {
  return (
    <div className="view-container interactive-notebooks-view">
      <div className="view-header">
        <h1>📓 Interactive Notebooks</h1>
        <p className="view-subtitle">
          Hands-on Jupyter notebooks for learning and experimentation
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🚀</div>
        <div className="banner-content">
          <h3>Learn by Doing</h3>
          <p>
            All notebooks are fully executable and include detailed explanations, code examples,
            and visualizations. Run them locally or open directly in Google Colab.
          </p>
        </div>
      </div>

      <section className="quick-launch-section">
        <h2>⚡ Quick Launch</h2>
        <div className="launch-grid">
          <a 
            href="https://colab.research.google.com/github/monksealseal/weatherflow/blob/main/notebooks/weatherflow_colab_demo.ipynb"
            className="launch-card"
            target="_blank"
            rel="noopener noreferrer"
          >
            <div className="launch-icon">🚀</div>
            <h3>Start in Colab</h3>
            <p>No installation needed - run in your browser with free GPU</p>
          </a>
          
          <div className="launch-card">
            <div className="launch-icon">💻</div>
            <h3>Run Locally</h3>
            <pre><code>git clone https://github.com/monksealseal/weatherflow
cd weatherflow
jupyter notebook</code></pre>
          </div>
        </div>
      </section>

      <section className="notebooks-section">
        <h2>📚 Available Notebooks</h2>
        <div className="notebooks-grid">
          {NOTEBOOKS.map((notebook, idx) => (
            <div key={idx} className="notebook-card">
              <h3>{notebook.title}</h3>
              <p className="notebook-description">{notebook.description}</p>
              
              <div className="topics">
                {notebook.topics.map(topic => (
                  <span key={topic} className="topic-tag">{topic}</span>
                ))}
              </div>
              
              <code className="notebook-path">{notebook.path}</code>
              
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

      <section className="features-section">
        <h2>✨ Notebook Features</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>📝 Detailed Explanations</h3>
            <p>Every code cell includes comprehensive markdown explanations</p>
          </div>
          <div className="feature-card">
            <h3>🎯 Working Examples</h3>
            <p>Real, executable code that demonstrates actual functionality</p>
          </div>
          <div className="feature-card">
            <h3>📊 Visualizations</h3>
            <p>Interactive plots and animations to understand results</p>
          </div>
          <div className="feature-card">
            <h3>🔬 Experiments</h3>
            <p>Modify parameters and see immediate results</p>
          </div>
          <div className="feature-card">
            <h3>💡 Best Practices</h3>
            <p>Learn recommended approaches and common pitfalls</p>
          </div>
          <div className="feature-card">
            <h3>🏃 Ready to Run</h3>
            <p>All dependencies handled, just execute cells in order</p>
          </div>
        </div>
      </section>

      <section className="tips-section">
        <h2>💡 Tips for Using Notebooks</h2>
        <div className="tips-grid">
          <div className="tip-card">
            <h3>Start with Complete Guide</h3>
            <p>
              Begin with <code>complete_guide.ipynb</code> for a comprehensive overview
              of all WeatherFlow capabilities.
            </p>
          </div>
          <div className="tip-card">
            <h3>Use Colab for GPU Access</h3>
            <p>
              Google Colab provides free GPU access. Enable it via Runtime → Change runtime type → GPU.
            </p>
          </div>
          <div className="tip-card">
            <h3>Modify and Experiment</h3>
            <p>
              Don't just run cells - modify parameters, try different settings,
              and see what happens!
            </p>
          </div>
          <div className="tip-card">
            <h3>Save Your Work</h3>
            <p>
              In Colab: File → Save a copy in Drive to keep your changes.
              Locally: notebooks are version controlled.
            </p>
          </div>
        </div>
      </section>

      <section className="troubleshooting-section">
        <h2>🔧 Troubleshooting</h2>
        <div className="troubleshooting-content">
          <details>
            <summary>Module Import Errors</summary>
            <div className="solution">
              <p>If you see import errors, install WeatherFlow:</p>
              <pre><code>!pip install git+https://github.com/monksealseal/weatherflow.git</code></pre>
            </div>
          </details>
          
          <details>
            <summary>Out of Memory Errors</summary>
            <div className="solution">
              <p>Reduce batch size or use smaller data:</p>
              <pre><code>batch_size = 8  # Try smaller value
# Or use subset of data
dataset = dataset[:100]</code></pre>
            </div>
          </details>
          
          <details>
            <summary>Data Download Issues</summary>
            <div className="solution">
              <p>For ERA5 data, notebooks use small synthetic data by default. For real data:</p>
              <pre><code># Use WeatherBench2 remote data (no download needed)
dataset = ERA5Dataset(use_remote=True)</code></pre>
            </div>
          </details>
        </div>
      </section>
    </div>
  );
}
