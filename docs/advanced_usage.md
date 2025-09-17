# Advanced Usage Guide

This page collects battle-tested patterns for running larger experiments,
integrating WeatherFlow with other services, and taking advantage of the more
specialised utilities bundled with the library.

## Train at scale with `FlowTrainer`

`FlowTrainer` already includes features you typically need in a research loop:

- **Mixed precision** via PyTorch AMP (`use_amp=True` by default on CUDA).
- **Physics regularisation** by calling `model.compute_physics_loss` when the
  model implements it (`physics_regularization=True`).
- **Configurable loss surfaces** through the `loss_type` parameter
  (`"mse"`, `"huber"`, or `"smooth_l1"`).
- **Checkpointing** with `save_checkpoint`/`load_checkpoint` and an optional
  scheduler slot.

Example configuration with logging and checkpoints:

```python
import torch
from weatherflow.training.flow_trainer import FlowTrainer

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
trainer = FlowTrainer(
    model=model,
    optimizer=optimizer,
    device="cuda:0",
    use_amp=True,
    use_wandb=True,             # log batch metrics to Weights & Biases
    checkpoint_dir="runs/exp1", # checkpoints stored here
    physics_regularization=True,
    physics_lambda=0.05,
    loss_type="huber",
)

for epoch in range(20):
    metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.validate(val_loader)
    trainer.current_epoch += 1
    if val_metrics["val_loss"] < trainer.best_val_loss:
        trainer.best_val_loss = val_metrics["val_loss"]
        trainer.save_checkpoint("best.pt")
```

Reloading is as simple as `trainer.load_checkpoint("best.pt")`. If you need to
alter the physics weighting on the fly, adjust `trainer.physics_lambda` between
epochs.

## Custom physics projections with `WeatherODESolver`

When you want fine-grained control over integration and constraint enforcement,
use `weatherflow.solvers.WeatherODESolver` directly instead of the convenience
wrapper `WeatherFlowODE`.

```python
import numpy as np
import torch
from weatherflow.solvers import WeatherODESolver

solver = WeatherODESolver(
    method="rk4",
    rtol=5e-5,
    atol=5e-5,
    physics_constraints=True,
    constraint_types=["mass", "energy", "vorticity"],
    constraint_weights={"mass": 1.0, "energy": 0.3, "vorticity": 0.2},
    grid_spacing=(np.deg2rad(2.5), np.deg2rad(2.5)),
)

with torch.no_grad():
    trajectory, stats = solver.solve(
        velocity_fn=lambda x, t: model(x, t.expand(x.shape[0])),
        x0=initial_state,
        t_span=torch.linspace(0.0, 1.0, steps=25, device=initial_state.device),
    )

print(stats)
```

The solver tracks constraint violations and relative energy drift in `stats`.
Use these diagnostics to tune `constraint_weights` or swap to a different
integration scheme.

## Serving interactive experiments

WeatherFlow ships with a lightweight FastAPI application that generates
synthetic datasets, trains a small `WeatherFlowMatch` model, and exposes the
results over a REST API consumed by the React dashboard in `frontend/`.

1. Start the API:

   ```bash
   uvicorn weatherflow.server.app:app --reload --port 8000
   ```

   The server exposes `/api/options` (configuration metadata) and
   `/api/experiments` (launch a training + inference run). Internally it uses the
   same `FlowTrainer` and solver utilities documented above.

2. In another terminal, install and launch the dashboard:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Open the printed URL (usually `http://localhost:5173`) to explore the guided
   workflow. The dashboard walks through dataset synthesis, model
   configuration, training progress, and channel-wise trajectory visualisations.

For production usage you can freeze the API responses, mount a GPU-backed model,
or extend the schemas defined in `weatherflow/server/app.py`.

## Working with notebooks and the dashboard

The repository includes a helper script that provisions a notebook environment,
installs the package in editable mode, and fixes relative imports in the bundled
notebooks:

```bash
python setup_notebook_env.py
```

Alternatively, install the notebook requirements manually:

```bash
pip install -r notebooks/notebook_requirements.txt
python notebooks/fix_notebook_imports.py
```

Use `check_notebooks.py` to execute and validate the notebooks inside CI. The
`NOTEBOOK_GUIDE.md` file explains the structure and recommended workflow.

## Atmospheric education and SKEW-T tooling

WeatherFlow includes unique tools for teaching and visualising atmospheric
concepts:

- `weatherflow.utils.SkewTImageParser` extracts temperature/dewpoint traces from
  SKEW-T imagery and returns thermodynamic profiles.
- `weatherflow.utils.SkewT3DVisualizer` renders the extracted profile as an
  interactive 3D curtain plot using Plotly.
- `weatherflow.education.GraduateAtmosphericDynamicsTool` builds interactive
  dashboards for geostrophic balance, Rossby-wave dispersion, and practice
  problem generation.

Example workflow:

```python
from weatherflow.utils import SkewTImageParser, SkewT3DVisualizer
from weatherflow.education import GraduateAtmosphericDynamicsTool

parser = SkewTImageParser()
profile = parser.parse_image("data/sample_skewt.png")
figure_path = SkewT3DVisualizer().create_and_save(profile, "skewt.html")

print("3D sounding saved to", figure_path)

tool = GraduateAtmosphericDynamicsTool(reference_latitude=45.0)
rossby_fig = tool.create_rossby_wave_lab(mean_flow=20.0)
rossby_fig.show()
```

These components are optional at runtime; install Plotly if you plan to use the
visualisers.

## Continuous integration and quality checks

- Run `pytest` before pushing changes to exercise the test suite in `tests/`.
- Apply `flake8`, `black --check`, and `isort --check-only` if you installed the
  development extras. The project ships with a `.pre-commit-config.yaml` file so
  you can enable automatic formatting.
- Use `mkdocs build` (with the docs extra installed) to validate the
  documentation renders without warnings.

Combine these practices with the patterns above to move from exploratory scripts
to reproducible experiments quickly.
