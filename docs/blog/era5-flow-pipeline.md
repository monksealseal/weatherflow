# Building an ERA5 Flow Matching Pipeline from Scratch

The new [`notebooks/era5_flow_matching_pipeline.ipynb`](../../notebooks/era5_flow_matching_pipeline.ipynb) notebook stitches together the most impactful WeatherFlow components so you can experience the full "aha!" moment—loading ERA5-style data, training a flow matching vector field, running ODE-based inference, and visualising the result in a single session.

## Why this pipeline matters

Flow matching turns weather prediction into a continuous trajectory problem: instead of forecasting discrete steps, the model learns a velocity field that drives an ODE solver. WeatherFlow provides everything required to explore that idea:

- **Data adapters** such as `ERA5Dataset` that know how to open WeatherBench2 ERA5 archives or local Zarr/NetCDF stores with robust fallbacks and metadata preservation.
- **Model primitives** like `WeatherFlowMatch`, a ConvNeXt-inspired backbone with optional multi-head attention and physics-aware divergence control.
- **Training infrastructure** via `FlowTrainer`, which implements the stochastic time sampling and flow-matching loss described in the research literature.
- **Inference tools** (`WeatherFlowODE`) that wrap the learned vector field in `torchdiffeq` solvers, allowing you to query arbitrary lead times with a single call.
- **Visual diagnostics** using `WeatherVisualizer`, complete with cartographic projections and difference maps.

The pipeline notebook demonstrates how each piece slots into place without assuming you already have the ERA5 data locally. A synthetic ERA5-style dataset mimics the tensor shapes and metadata so that the code paths remain identical when you later switch to the real data sources.

## Walk through the notebook

1. **Experiment configuration** – You choose the variables, pressure levels, grid resolution, and optimisation hyperparameters. The configuration mirrors the arguments taken by `ERA5Dataset` and `WeatherFlowMatch`, so it doubles as a script template.
2. **Data loading** – When `use_synthetic_data=True`, the notebook instantiates a smooth toy dataset that produces realistic-looking geopotential, temperature, and wind patterns while preserving ERA5 metadata. Toggle the flag off and the same loader wraps an `ERA5Dataset` instance with a channel-first view ready for the model.
3. **Model assembly** – `WeatherFlowMatch` is initialised with the discovered channel count and your chosen depth/width settings. You can quickly inspect the parameter count and the physics-informed branches that enforce approximate mass conservation.
4. **Training loop** – `FlowTrainer` handles randomising the flow times, computing the MSE-based flow loss, and optionally penalising divergence. The notebook logs both training and validation flow losses each epoch and visualises the curves so you can judge convergence at a glance.
5. **Inference** – Wrapping the trained model with `WeatherFlowODE` exposes a simple interface for sampling trajectories. One `ode_model(x0, lead_times)` call yields states at arbitrarily chosen fractions between the conditioning time and the forecast horizon.
6. **Visual analysis** – Finally, `WeatherVisualizer` generates side-by-side truth, prediction, and difference maps. The notebook automatically falls back to standard Matplotlib plots if cartopy is unavailable, so you still get informative diagnostics in constrained environments.

## Tips for custom experiments

- **Swap data sources** – Provide `era5_data_path` when you have WeatherBench2 credentials or a local Zarr; the wrapper keeps the tensor format identical to the synthetic fallback.
- **Scale the model** – The `hidden_dim` and `n_layers` configuration keys map directly to the ConvNeXt-style blocks inside `WeatherFlowMatch`. Increase them as you add more variables or higher resolution grids.
- **Monitor physics** – Enable `physics_informed=True` to activate the divergence penalty. The trainer logs the physics term so you can quantify how well the constraint holds during training.
- **Go multi-step** – The notebook demonstrates five evenly spaced lead times, but you can pass any `torch.linspace` (or custom tensor) into `WeatherFlowODE` to inspect seasonal or daily cycles at finer granularity.

Once you are comfortable with the pipeline, explore the other notebooks (`complete-data-exploration.ipynb`, `model-training-notebook.ipynb`, `prediction-visualization-notebook.ipynb`) for deeper dives into data preprocessing, hyperparameter sweeps, and WeatherBench evaluations.
