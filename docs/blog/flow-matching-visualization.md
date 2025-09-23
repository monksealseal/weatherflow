# Visualising Flow Matching Forecasts Like a Pro

After training a `WeatherFlowMatch` model, the real insight comes from seeing how its trajectories compare to the ERA5 ground truth. The new pipeline notebook ends with a rich visual analysis section that you can adapt for your own experiments. This post distils the key ideas so you can turn raw tensors into compelling diagnostics.

## From velocity field to forecasts

1. **Freeze the vector field** – Set the trainer's model to evaluation mode and construct a `WeatherFlowODE` wrapper. The solver accepts any sequence of lead times, so you can explore short-term extrapolations and longer horizons without retraining.
2. **Sample trajectories** – Grab a batch from your validation loader and call `ode_model(x0, lead_times)`. The output tensor has shape `[n_times, batch, channels, lat, lon]`; the final slice corresponds to the target interval (e.g., six hours ahead in the ERA5 6-hourly dataset).
3. **Unpack channels** – Because the loader flattens variables and pressure levels into the channel dimension, reshape the tensor back to `[variables, levels, lat, lon]` with a helper such as `unflatten_channels`. This lets you address specific variables (`'z'`, `'t'`, `'u'`, `'v'`) and pressure levels by index.

## Visual diagnostics with WeatherVisualizer

`WeatherVisualizer` was built to lower the friction of inspecting global weather fields:

- **Projection-aware plotting** – `plot_comparison` handles the geographic projection, coastlines, and color scales. Pass dictionaries like `{var_name: data}` and let the utility generate true, predicted, and difference maps in one figure.
- **Fallback support** – When `cartopy` is unavailable, the notebook automatically falls back to Matplotlib heatmaps, ensuring the workflow works even in stripped-down environments.
- **Consistent palettes** – Variable-specific colour maps (e.g., `'viridis'` for geopotential) and labels make it easy to compare across experiments.

Here's a minimal snippet you can drop into other notebooks or scripts:

```python
from weatherflow.models import WeatherFlowODE
from weatherflow.utils import WeatherVisualizer

model.eval()
ode_model = WeatherFlowODE(model, solver_method="dopri5", rtol=1e-4, atol=1e-4)
lead_times = torch.linspace(0.0, 1.0, steps=5, device=device)
trajectory = ode_model(x0, lead_times)
forecast = trajectory[-1].cpu()
truth = batch["target"]

truth_fields = unflatten_channels(truth[0], variables, pressure_levels)
forecast_fields = unflatten_channels(forecast[0], variables, pressure_levels)

visualizer = WeatherVisualizer(figsize=(16, 6))
fig, axes = visualizer.plot_comparison(
    true_data={"z": truth_fields[0, 0]},
    pred_data={"z": forecast_fields[0, 0]},
    var_name="z",
    level_idx=0,
    title=f"Geopotential @ {pressure_levels[0]} hPa"
)
```

## Beyond static maps

- **Difference statistics** – Combine the plotted fields with metrics such as per-sample MSE or anomaly correlations to quantify the gaps you observe.
- **Lead-time animations** – Because `WeatherFlowODE` returns the whole trajectory, you can turn the `trajectory` tensor into an animation using Matplotlib's `FuncAnimation` for time-lapse forecasts.
- **Dashboard integration** – The WeatherFlow FastAPI + React dashboard (see `frontend/`) already consumes the same vector field and ODE components; feed it your checkpoints to create interactive demos for stakeholders.

Whether you are validating a research prototype or preparing a production rollout, pairing the flow matching model with rich visual diagnostics keeps the focus on the physics that matter.
