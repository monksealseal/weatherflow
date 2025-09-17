# API Reference Overview

The WeatherFlow package exposes a cohesive API organised around the workflow of
loading data, fitting a flow model, integrating trajectories, and presenting the
results. The following pages document each subsystem in detail:

- [Data Loading](api/data.md) – `ERA5Dataset`, `WeatherDataset`,
  `create_data_loaders`, and dataset utilities.
- [Models](api/models.md) – flow-matching networks, physics-guided baselines,
  score-matching helpers, and conversion utilities.
- [Training](api/training.md) – the `FlowTrainer` class, `compute_flow_loss`, and
  checkpoint helpers.
- [Solvers and Paths](api/solvers.md) – probability paths (`ProbPath`,
  `GaussianProbPath`, `CondOTPath`) and the `WeatherODESolver` integration
  toolkit.
- [Utilities and Visualization](api/visualization.md) – map plotting,
  flow-trajectory inspectors, SKEW-T tooling, and the educational dashboard
  helpers.

The package root (`weatherflow/__init__.py`) re-exports the most common symbols
so you can do:

```python
from weatherflow import (
    ERA5Dataset,
    WeatherFlowMatch,
    WeatherFlowODE,
    FlowTrainer,
    WeatherVisualizer,
)
```

Refer back to this page whenever you need to locate a specific class; each entry
links to the module containing the implementation and docstring.
