import GenericInfoView from './GenericInfoView';
import './FlowMatchingView.css';

export default function FlowMatchingView() {
  const sections = [
    {
      title: '🌊 What is Flow Matching?',
      content: (
        <>
          <p>
            Flow matching is a powerful technique for generative modeling that learns continuous
            transformations between probability distributions. Unlike traditional approaches,
            flow matching directly learns the vector field that transports samples from an initial
            distribution to a target distribution.
          </p>
          <p>
            In weather prediction, we use flow matching to learn the evolution of atmospheric
            states over time. Given an initial weather state x₀, the model learns to predict
            the future state x₁ by following a continuous path through state space.
          </p>
        </>
      )
    },
    {
      title: '🧮 Mathematical Framework',
      content: (
        <>
          <p>
            Flow matching models learn a vector field v(x, t) that describes how atmospheric
            states evolve. The evolution is governed by the ODE:
          </p>
          <div className="equation-display">
            <code>dx/dt = v(x, t)</code>
          </div>
          <p>
            The model is trained to match the conditional vector field that interpolates
            between initial and target states:
          </p>
          <div className="equation-display">
            <code>v_t(x|x₀, x₁) = (x₁ - x₀) / (1 - t)</code>
          </div>
        </>
      )
    },
    {
      title: '🏗️ Architecture',
      content: (
        <>
          <ul>
            <li><strong>Input:</strong> Current atmospheric state + time embedding</li>
            <li><strong>Encoder:</strong> Multi-scale convolutional layers to extract features</li>
            <li><strong>Transformer blocks:</strong> Self-attention for capturing global patterns</li>
            <li><strong>Decoder:</strong> Upsampling to predict velocity field</li>
            <li><strong>Output:</strong> Vector field describing state evolution</li>
          </ul>
        </>
      )
    },
    {
      title: '✨ Advantages for Weather Prediction',
      content: (
        <>
          <ul>
            <li>Continuous-time prediction (any forecast lead time)</li>
            <li>Inherently probabilistic (sample multiple trajectories)</li>
            <li>Physically interpretable (velocity field has clear meaning)</li>
            <li>Flexible conditioning (initial state, forcing, parameters)</li>
            <li>Stable training (regression-like objective)</li>
          </ul>
        </>
      )
    },
    {
      title: '🔌 Repository wiring for this page',
      content: (
        <>
          <p>
            Everything shown on the Flow Matching and Model Zoo navigation items already maps to concrete
            Python entry points—no new backend code is required.
          </p>
          <ul>
            <li><strong>Core models:</strong> <code>weatherflow/models/flow_matching.py</code> and <code>weatherflow/models/weather_flow.py</code> back the architecture cards.</li>
            <li><strong>Training metrics:</strong> <code>weatherflow/training/flow_trainer.py</code> exports the RMSE/MAE/energy fields rendered in Evaluation.</li>
            <li><strong>API payloads:</strong> <code>run_experiment.py</code> posts to <code>weatherflow/server/app.py</code>, letting the website ingest runs without bespoke adapters.</li>
            <li><strong>Artifacts:</strong> Drop <code>.pt</code> checkpoints and <code>model_card.json</code> files into <code>model_zoo/</code> so Model Zoo and Flow Matching cards light up immediately.</li>
          </ul>
        </>
      )
    },
    {
      title: '🚦 Instant vs. training paths',
      content: (
        <>
          <ul>
            <li><strong>Skip training:</strong> Run <code>model_zoo/download_model.py</code> or <code>examples/flow_matching/simple_example.py</code> to populate cards and visualizations without GPUs.</li>
            <li><strong>Short local run:</strong> Use <code>examples/flow_matching/era5_strict_training_loop.py</code> with a handful of epochs—FlowTrainer already emits the metrics the UI expects.</li>
            <li><strong>Remote job:</strong> Train via <code>model_zoo/train_model.py</code> on Hugging Face or a cluster, then point <code>VITE_API_URL</code> at the remote FastAPI host while the UI reuses the same payloads.</li>
          </ul>
        </>
      )
    }
  ];

  const codeExamples = [
    {
      title: 'Creating a Flow Matching Model',
      code: `from weatherflow.models import WeatherFlowMatch

# Create flow matching model
model = WeatherFlowMatch(
    input_channels=4,        # z, t, u, v
    hidden_dim=256,
    n_layers=6,
    use_attention=True,
    physics_informed=True,
    grid_size=(32, 64)
)

# Input: atmospheric state [batch, channels, lat, lon]
x0 = torch.randn(32, 4, 32, 64)  # Initial state
x1 = torch.randn(32, 4, 32, 64)  # Target state (e.g., 6h later)
t = torch.rand(32)                # Random time in [0, 1]

# Training: learn to predict velocity field
velocity = model(x0, t)
loss = model.compute_flow_loss(x0, x1, t)['total_loss']`
    },
    {
      title: 'Generating Predictions',
      code: `from weatherflow.models import WeatherFlowODE

# Wrap model with ODE solver for prediction
ode_model = WeatherFlowODE(
    flow_model=model,
    solver_method='dopri5',  # Adaptive Runge-Kutta
    rtol=1e-4,
    atol=1e-4
)

# Generate forecast
times = torch.linspace(0, 1, 13)  # 0-72 hours in 6h steps
prediction = ode_model(x0, times)  # [time_steps, batch, channels, lat, lon]

# Access specific forecast times
t6h = prediction[1]   # +6 hour forecast
t24h = prediction[4]  # +24 hour forecast
t72h = prediction[12] # +72 hour forecast`
    },
    {
      title: 'Training Loop',
      code: `import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_loader:
        x0, x1 = batch['input'], batch['target']
        t = torch.rand(x0.size(0))
        
        # Compute loss
        losses = model.compute_flow_loss(x0, x1, t)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")`
    },
    {
      title: 'Serve flow runs to the website',
      code: `# Start the FastAPI backend
uvicorn weatherflow.server.app:app --port 8000 --reload

# Post the bundled payload (used by the Experiments page)
python run_experiment.py

# Point the UI to a remote or local API for GitHub Pages builds
export VITE_API_URL="https://<your-api-host>"
npm run build`
    }
  ];

  const features = [
    'Continuous normalizing flows for flexible time prediction',
    'Physics-informed architecture with conservation constraints',
    'Adaptive ODE solvers for accurate trajectory integration',
    'Multi-scale attention for capturing global patterns',
    'Support for ensemble forecasting via stochastic sampling',
    'Efficient training with simple regression objective',
    'Compatible with ERA5 and WeatherBench2 datasets'
  ];

  return (
    <>
      <GenericInfoView
        icon="🌊"
        title="Flow Matching Models"
        subtitle="Continuous normalizing flows for weather prediction"
        bannerTitle="Advanced Generative Modeling"
        bannerContent="Flow matching combines the flexibility of normalizing flows with the efficiency of regression-based training. Learn continuous transformations of atmospheric states over time."
        sections={sections}
        codeExamples={codeExamples}
        features={features}
        relatedPages={[
          'Model Zoo - Pre-trained Models',
          'Physics-Guided Models',
          'Basic Training',
          'WeatherFlow Paper (coming soon)'
        ]}
      />
      <section className="source-section">
        <h2>📦 Source Code</h2>
        <div className="source-links">
          <code>weatherflow/models/flow_matching.py</code> - Core flow matching implementation
          <code>weatherflow/models/weather_flow.py</code> - WeatherFlowMatch model class
          <code>examples/flow_matching/simple_example.py</code> - Basic usage example
          <code>examples/flow_matching/era5_strict_training_loop.py</code> - Production training
        </div>
      </section>
    </>
  );
}
