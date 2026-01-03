# Comprehensive Repository Guide

This guide distills the WeatherFlow repository into a single reference so you
can quickly understand what is already implemented, how the pieces fit
together, and where to focus next. Use it alongside the API reference and
tutorials when planning new work.

## What the project delivers today

- **Flow-matching weather models** that respect atmospheric physics through
  optional enhanced physics losses.
- **Rich data tooling** for ERA5 and synthetic datasets, including streaming,
  sequence forecasting, and WebDataset ingest.
- **Training and inference utilities** that wrap flow models with ODE-based
  solvers and evaluation metrics.
- **Interactive surfaces**: a FastAPI experimentation service and React
  dashboard for no-code demos, plus graduate-level educational visualisations.
- **Reproducible experiments** covering physics ablations and WeatherBench2
  benchmarks with saved artefacts.

## Architecture at a glance

| Area | Key entry points | What they provide |
| --- | --- | --- |
| Data & preprocessing | `weatherflow/data/era5.py`, `weatherflow/data/streaming.py`, `weatherflow/data/sequence.py`, `weatherflow/data/webdataset_loader.py` | ERA5 loading from WeatherBench2 or local stores, streaming readers, multi-step datasets, and WebDataset ingest helpers. |
| Models & physics | `weatherflow/models/flow_matching.py`, `weatherflow/models/icosahedral.py`, `weatherflow/models/physics_guided.py`, `weatherflow/models/stochastic.py`, `weatherflow/physics/losses.py` | Flow-matching backbones (grid and icosahedral), physics-guided attention, stochastic surrogates, and enhanced physics loss terms (PV, energy spectra, divergence, geostrophic balance). |
| Training & evaluation | `weatherflow/training/flow_trainer.py`, `weatherflow/training/metrics.py`, `weatherflow/test_flow_trainer.py` | High-level training loop, flow loss computation, and reference smoke tests. |
| Paths & solvers | `weatherflow/path`, `weatherflow/solvers/ode_solver.py`, `weatherflow/solvers/riemannian.py`, `weatherflow/solvers/langevin.py`, `weatherflow/models/conversion.py` | Trajectory/path abstractions, deterministic and Riemannian ODE solvers, Langevin dynamics, and score/vector-field conversions. |
| Simulation & manifolds | `weatherflow/simulation`, `weatherflow/manifolds` | Lightweight simulation orchestrator and spherical/icosahedral geometry utilities. |
| Visualization & education | `weatherflow/utils/visualization.py`, `weatherflow/education` | Plotting helpers (maps, animations, vector fields) and graduate-level dynamics dashboards. |
| API & dashboard | `weatherflow/server/app.py`, `frontend/` | FastAPI service that synthesizes data, configures/trains models, and streams results to the React/Vite dashboard (Plotly + Three.js widgets). |
| Docs, examples, notebooks | `docs/`, `examples/`, `notebooks/`, `weatherflow-notebooks/` | MkDocs site, runnable scripts, tutorial notebooks, and fixed-import notebook copies. |

## End-to-end workflows

### Scripted training and forecasting
1. **Load data** with `ERA5Dataset` or `create_data_loaders`.
2. **Configure a model** using `WeatherFlowMatch` (grid) or
   `IcosahedralFlowMatch` (mesh). Enable `physics_informed=True` or set
   `enhanced_physics_losses` when needed.
3. **Train** via `FlowTrainer` or the example scripts:
   - `examples/weather_prediction.py` for a configurable ERA5 pipeline.
   - `examples/flow_matching/simple_example.py` for a minimal flow demo.
4. **Integrate trajectories** with `WeatherFlowODE` and
   `WeatherODESolver` to roll forecasts forward.
5. **Visualise and evaluate** using `WeatherVisualizer` plus metrics in
   `weatherflow/training/metrics.py`.

### Interactive synthetic demo
1. Start the API: `uvicorn weatherflow.server.app:app --reload --port 8000`.
2. In `frontend/`, run `npm install` then `npm run dev` to launch the dashboard.
3. Use the UI to choose variables, grid size, backbone (grid/icosahedral),
   solver, and physics toggles; trigger an experiment to see loss curves and
   generated trajectories.

### Research experiments and benchmarks
- **Physics ablations**: `experiments/ablation_study.py` and
  `experiments/quick_ablation_demo.py` compare baseline vs. physics-enhanced
  models; outputs live in `experiments/ablation_results/`.
- **WeatherBench2 validation**: `experiments/weatherbench2_evaluation.py`
  measures skill against IFS HRES, GraphCast, and Pangu-Weather with plots and
  JSON summaries in `experiments/weatherbench2_results/`.
- **Educational utilities**: `examples/physics_loss_demo.py` and
  `examples/skewt_3d_visualizer.py` showcase the enhanced physics losses and
  SKEW-T visualizations.

## Development and operations

- **Environments**
  - Base: `pip install -e .`
  - Development extras (linting/docs/tests): `pip install -r requirements-dev.txt`
    or `pip install -e .[docs]`
  - Notebooks: run `python setup_notebook_env.py` to create a kernel and fix
    imports, or install `notebooks/notebook_requirements.txt` manually.
- **Testing & quality**
  - Python: `pytest` covers data loaders, models, physics losses, and trainer
    smoke tests.
  - Frontend: `npm test` (Vitest) and `npm run lint` (ESLint) in `frontend/`.
  - Fast checks during iteration: target specific pytest modules or small
    dataset sizes to keep runtime manageable.
- **Docs**
  - Build the MkDocs site locally with `mkdocs serve` after installing docs
    extras; content lives under `docs/` with navigation configured in
    `mkdocs.yml`.
- **Release and packaging**
  - Python packaging metadata is defined in `pyproject.toml`, `setup.py`, and
    `setup.cfg`. Update `VERSION` in `weatherflow/version.py` alongside release
    notes in `CHANGELOG.md` or `RELEASE_NOTES.md` when cutting a build.

## How to move forward

- **Decide on data scale and fidelity**: choose between streaming ERA5,
  WebDataset ingestion, or local NetCDF/Zarr; confirm required derived fields
  before training.
- **Pick the backbone**: start with `WeatherFlowMatch` on lat/lon grids; move to
  `IcosahedralFlowMatch` if spherical uniformity or higher resolution is needed.
- **Select physics rigor**: toggle `physics_informed` and the enhanced physics
  losses to balance speed vs. physical consistency; use the ablation scripts to
  validate the trade-off.
- **Plan the interface**: for collaborators who prefer UIs, keep the FastAPI +
  React stack running; otherwise, standardize on the scripted workflows above.
- **Operationalize testing**: enforce `pytest` and frontend lint/test runs in
  CI; lean on the notebook setup script for reproducible demos.

