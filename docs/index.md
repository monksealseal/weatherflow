# WeatherFlow

WeatherFlow is a research-friendly Python library for building flow-matching
weather models that respect atmospheric physics. It wraps PyTorch modules for
vector-field learning, spherical geometry utilities, probabilistic paths, and a
set of visualization and education tools that make it easy to prototype new
ideas or reproduce the examples that ship with the repository.

## What makes WeatherFlow different?

WeatherFlow collects everything you need for a flow-matching weather workflow in
one place:

- **Rich data loaders** that can stream ERA5 reanalysis data directly from the
  WeatherBench2 bucket, open local NetCDF/Zarr archives, and optionally add
  derived diagnostics such as wind speed and vorticity.
- **Model zoo for experimentation**, including the `WeatherFlowMatch` flow
  network, physics-guided attention baselines, stochastic surrogates, and dense
  neural ODE formulations.
- **Training and solver utilities** that understand flow-matching loss
  functions, mixed precision, physics regularisation, and ODE-based inference.
- **Visualisation and education helpers** for publishing maps, animations, and
  interactive graduate-level atmospheric dynamics dashboards.
- **FastAPI + React dashboard** that lets new users explore a synthetic
  workflow end-to-end without writing code.

Whether you are validating a paper idea or want a turnkey demo, the library is
organised so you can start from a minimal script and scale to complex
experiments without rewriting core infrastructure.

## How to use this documentation

The documentation mirrors the stages of a typical project:

1. [Installation](installation.md) explains how to set up a working Python
   environment (with optional extras for docs, notebooks, and linting).
2. [Getting Started](getting_started.md) walks through loading ERA5 data,
   training `WeatherFlowMatch`, producing forecasts with `WeatherFlowODE`, and
   creating quick diagnostics.
3. [Tutorials](tutorials/quickstart.md) dive deeper into common tasks such as
   manipulating ERA5 archives or configuring custom physics constraints.
4. [Advanced Usage](advanced_usage.md) covers topics like the interactive API,
   notebook tooling, SKEW-T parsing, and production check-pointing.
5. [API Reference](api/data.md) documents the public Python surfaces organised
   by domain (data, models, training, solvers, utilities).
6. [Advanced Topics](advanced_topics/continuous_normalizing_flows.md) collects
   background notes on continuous normalising flows and WeatherFlow's specific
   implementation choices.

Each page links back to the relevant modules and examples so you can jump from
concepts to runnable code quickly.

## Repository layout

The repository is split into a few key directories:

| Location | Description |
| --- | --- |
| `weatherflow/` | Python package containing data loaders, models, solvers, training loops, and utilities. |
| `examples/` | End-to-end scripts (ERA5 flow-matching, SKEW-T visualisation, minimal flow examples). |
| `docs/` | MkDocs source for this documentation site. |
| `notebooks/` | Jupyter notebooks illustrating the concepts in narrative form. |
| `frontend/` | Vite/React dashboard that speaks to the FastAPI service defined in `weatherflow/server`. |
| `tests/` | Pytest suite and utilities for regression testing. |

See [Advanced Usage](advanced_usage.md#working-with-notebooks-and-the-dashboard)
for tips on setting up the dashboard and notebook environment.

## Typical workflow

A flow-matching experiment with WeatherFlow usually follows these steps:

1. **Prepare data** – load ERA5 slices with `ERA5Dataset` or synthetic data from
   the FastAPI service, optionally enabling normalisation or derived fields.
2. **Configure a model** – start with `WeatherFlowMatch` for flow matching or
   `PhysicsGuidedAttention`/`StochasticFlowModel` for lighter baselines.
3. **Train** – use `FlowTrainer` or the scripted examples to optimise the model
   with physics-aware loss terms and mixed precision.
4. **Integrate trajectories** – wrap the trained flow with `WeatherFlowODE` or
   call `WeatherODESolver` directly to evolve the state through time.
5. **Evaluate and visualise** – quantify skill, render maps with
   `WeatherVisualizer`, and export animations or 3D soundings.
6. **Deploy or share** – expose the model through the FastAPI interface or the
   notebook gallery for interactive exploration.

Refer to the [Quick Start tutorial](tutorials/quickstart.md) for a runnable
version of this workflow.

## Where to go next

If you are new to the project, install the dependencies and run through the
[Quick Start tutorial](tutorials/quickstart.md). Users interested in the theory
can review the [Flow Matching Primer](flow_matching.md), while advanced users
can jump straight to the [API reference](api/data.md) to integrate WeatherFlow
into their own pipelines.
