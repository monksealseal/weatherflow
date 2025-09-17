# Quick Start Tutorial

This tutorial replicates the workflow demonstrated in `examples/weather_prediction.py`
and breaks it down into digestible steps. By the end you will have trained a small
flow-matching model on ERA5 data, generated a short forecast trajectory, and
rendered diagnostic plots.

## Prerequisites

- Follow the [installation guide](../installation.md) to create a virtual
  environment and install WeatherFlow with the required extras.
- Ensure you can import the package: `python -c "import weatherflow"`.
- Optional: install the development extras (`pip install -e .[dev]`) so you can
  run the pytest suite.

## 1. Configure data access

Create a Python file `quickstart.py` and start by constructing the data loaders.
The helper `create_data_loaders` handles all ERA5 access mechanics.

```python
from weatherflow.data import create_data_loaders

train_loader, val_loader = create_data_loaders(
    variables=["z", "t"],
    pressure_levels=[500],
    train_slice=("2015", "2015"),
    val_slice=("2016", "2016"),
    batch_size=8,
    num_workers=2,
    normalize=True,
)
```

> **Note:** When working offline, download the ERA5 subset manually and pass
> `data_path="/path/to/local.zarr"` to load from disk instead of Google Cloud
> Storage.

## 2. Instantiate the model and optimiser

```python
import torch
from weatherflow.models import WeatherFlowMatch

model = WeatherFlowMatch(
    input_channels=train_loader.dataset.shape[0],
    hidden_dim=128,
    n_layers=4,
    use_attention=True,
    physics_informed=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

If you prefer a lighter network, replace `WeatherFlowMatch` with
`PhysicsGuidedAttention` or `StochasticFlowModel`.

## 3. Train with FlowTrainer

```python
import torch
from weatherflow.training.flow_trainer import FlowTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = FlowTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
    physics_regularization=True,
    physics_lambda=0.1,
)

for epoch in range(5):
    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.validate(val_loader)
    print(
        f"Epoch {epoch:02d} | "
        f"train={train_metrics['loss']:.4f} "
        f"val={val_metrics['val_loss']:.4f}"
    )
```

Enable checkpointing by setting `checkpoint_dir="runs/quickstart"` when
instantiating the trainer.

## 4. Integrate a short forecast

After training, generate a small trajectory using `WeatherFlowODE`:

```python
from weatherflow.models.flow_matching import WeatherFlowODE

ode_model = WeatherFlowODE(model, solver_method="dopri5")

with torch.no_grad():
    batch = next(iter(val_loader))["input"].to(device)
    times = torch.linspace(0.0, 1.0, steps=6, device=device)
    trajectory = ode_model(batch, times)
```

Each entry in `trajectory` represents the predicted state at a given integration
step.

## 5. Visualise results

Use `WeatherVisualizer` to inspect the final forecast versus the ground truth.

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

Call `visualizer.plot_field(...)` or `visualizer.create_prediction_animation(...)`
to explore additional diagnostics. For flow evolution, try
`weatherflow.utils.FlowVisualizer().visualize_flow(...)`.

## 6. Optional: export metrics and checkpoints

Add a small helper to persist results:

```python
import json
from pathlib import Path

output_dir = Path("runs/quickstart")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_dir / "model.pt")
with open(output_dir / "metrics.json", "w") as fh:
    json.dump({"train": train_metrics, "val": val_metrics}, fh, indent=2)
```

## 7. Validate your environment

Run the unit tests to ensure everything functions as expected:

```bash
pytest
```

If you installed the documentation extras, you can also verify the docs build
with `mkdocs build`.

## 8. Next steps

- Swap in more variables or multiple pressure levels by editing `variables` and
  `pressure_levels`.
- Enable the FastAPI dashboard described in [Advanced Usage](../advanced_usage.md)
  for a guided, no-code exploration.
- Continue with the [ERA5 tutorial](era5.md) to learn about derived variables and
  dataset caching.
