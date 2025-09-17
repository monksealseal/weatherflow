# Getting Started with WeatherFlow

This guide walks through the quickest route from a fresh clone of the
repository to training and evaluating a small flow-matching model. It assumes
you have already followed the [installation steps](installation.md) and have an
active virtual environment.

## 1. Import the package and check the version

```python
import weatherflow
print(weatherflow.__version__)
```

The package exposes high-level entry points from `weatherflow.__init__` so that
you can import the most common classes from the top level:

```python
from weatherflow.data import create_data_loaders
from weatherflow.models import WeatherFlowMatch
from weatherflow.models.flow_matching import WeatherFlowODE
from weatherflow.training.flow_trainer import FlowTrainer
from weatherflow.utils import WeatherVisualizer
```

## 2. Load ERA5 samples

The `ERA5Dataset` streams slices of reanalysis data either from the public
WeatherBench2 bucket or from a local archive. The helper
`create_data_loaders(...)` instantiates train/validation datasets and wraps them
in PyTorch data loaders with sensible defaults.

```python
from weatherflow.data import create_data_loaders

train_loader, val_loader = create_data_loaders(
    variables=["z", "t"],          # geopotential height and temperature
    pressure_levels=[500],         # hPa levels to include
    train_slice=("2015", "2015"),  # inclusive date range
    val_slice=("2016", "2016"),
    batch_size=4,                  # keep tiny while exploring
    num_workers=0,                 # use >0 once the dataset is cached
    normalize=True,                # standardises using built-in statistics
)

sample = next(iter(train_loader))
print(sample["input"].shape)      # (batch, variables, levels, lat, lon)
print(sample["metadata"])
```

Key dataset features:

- Automatic fallback between anonymous GCS access, HTTP mirrors, local Zarr, and
  NetCDF files.
- Optional derived diagnostics (wind speed, vorticity) via
  `ERA5Dataset(..., add_physics_features=True)`.
- In-memory caching with `cache_data=True` for small experiments.

## 3. Configure a flow-matching model

`WeatherFlowMatch` implements the velocity field used for flow matching. The
architecture combines convolutional blocks, sinusoidal time embeddings, and an
optional multi-head attention layer. You can enable a lightweight physics
projection that nudges the vector field toward divergence-free behaviour.

```python
from weatherflow.models import WeatherFlowMatch

model = WeatherFlowMatch(
    input_channels=sample["input"].shape[1],
    hidden_dim=128,
    n_layers=4,
    use_attention=True,
    physics_informed=True,
)
```

If you want an even smaller baseline, experiment with
`weatherflow.models.PhysicsGuidedAttention` or
`weatherflow.models.StochasticFlowModel`.

## 4. Train with `FlowTrainer`

`FlowTrainer` wraps a model, optimizer, and optional scheduler/regularisation
settings. It handles mixed-precision, physics-aware penalties, and checkpointing.

```python
import torch
from weatherflow.training.flow_trainer import FlowTrainer, compute_flow_loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
trainer = FlowTrainer(
    model=model,
    optimizer=optimizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    physics_regularization=True,
    physics_lambda=0.1,
)

for epoch in range(2):
    metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.validate(val_loader)
    print(f"Epoch {epoch}: {metrics['loss']:.4f} | val {val_metrics['val_loss']:.4f}")
```

Behind the scenes, each iteration draws random interpolation times `t`, calls
`model(x0, t)` to obtain the predicted velocity, and compares it against the
straight-line target `(x1 - x0)/(1 - t)` computed by `compute_flow_loss`.

> **Tip:** You can enable Weights & Biases logging by instantiating the trainer
> with `use_wandb=True` after configuring your `wandb` credentials.

## 5. Generate forecasts with `WeatherFlowODE`

Once the flow model converges, wrap it with `WeatherFlowODE` to integrate the
state forward in time using `torchdiffeq`.

```python
from weatherflow.models.flow_matching import WeatherFlowODE

ode_model = WeatherFlowODE(model, solver_method="dopri5", rtol=1e-4, atol=1e-4)

with torch.no_grad():
    batch = next(iter(val_loader))["input"].to(trainer.device)
    times = torch.linspace(0.0, 1.0, steps=5, device=trainer.device)
    trajectory = ode_model(batch, times)
```

`trajectory` has shape `(num_times, batch, channels, lat, lon)` and can be fed
into the visualisation utilities or custom diagnostics.

## 6. Visualise predictions

The `WeatherVisualizer` class renders global maps, comparisons, and animations
without leaving Matplotlib and Cartopy.

```python
from weatherflow.utils import WeatherVisualizer

visualizer = WeatherVisualizer()
visualizer.plot_comparison(
    true_data={"z": batch[0, 0].cpu()},
    pred_data={"z": trajectory[-1, 0, 0].cpu()},
    var_name="z",
    title="Geopotential Height – Forecast vs Truth",
)
```

For flow-specific insight, `weatherflow.utils.FlowVisualizer` animates how the
state evolves along the learned velocity field. If you have sounding imagery,
`weatherflow.utils.SkewTImageParser` and `SkewT3DVisualizer` can turn it into an
interactive 3D curtain plot.

## 7. Next steps

- Explore the [Quick Start tutorial](tutorials/quickstart.md) for a fully worked
  example with command-line arguments and disk outputs.
- Read the [ERA5 tutorial](tutorials/era5.md) to customise data access, derived
  fields, and caching strategies.
- Dive into [Advanced Usage](advanced_usage.md) for notebooks, the FastAPI
  dashboard, and deployment considerations.
- Check the [API reference](api/data.md) when you need the full signature of any
  class or function.
