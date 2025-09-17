# Release Notes

This page highlights notable changes in the WeatherFlow codebase. For the full
commit history see [`RELEASE_NOTES.md`](../RELEASE_NOTES.md).

## v0.4.1 (current)

- **Robust ERA5 loader** – `ERA5Dataset` now cycles through anonymous GCS,
  HTTPS, local Zarr, and NetCDF strategies with detailed logging and optional
  caching/derived diagnostics.
- **FlowTrainer upgrades** – mixed precision by default on CUDA, configurable
  loss surfaces (`mse`, `huber`, `smooth_l1`), physics regularisation hooks, and
  integrated checkpoint management.
- **Physics-aware ODE solver** – `WeatherODESolver` enforces mass, energy, and
  vorticity constraints while reporting violation statistics for each
  integration.
- **Interactive stack** – FastAPI service (`weatherflow.server.app`) and the
  accompanying React dashboard guide users through configuring datasets, models,
  and training runs without code.
- **Atmospheric education toolkit** – `GraduateAtmosphericDynamicsTool` ships
  with geostrophic balance dashboards, Rossby-wave laboratories, and step-by-step
  practice problems.
- **SKEW-T parsing and 3D rendering** – convert sounding imagery to quantitative
  profiles and interactive Plotly visualisations with
  `SkewTImageParser`/`SkewT3DVisualizer`.

## Earlier releases

- **v0.1.x** – initial publication featuring the first versions of
  `PhysicsGuidedAttention`, `StochasticFlowModel`, ERA5 data loaders, and the
  Matplotlib-based `WeatherVisualizer`.

We follow semantic versioning; expect breaking API changes only on major version
bumps. Each tagged release is accompanied by a git tag and matching entry in the
root `RELEASE_NOTES.md` file.
