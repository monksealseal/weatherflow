# Flow Matching and Diffusion Models

WeatherFlow implements continuous flow matching as described in recent neural ODE
and diffusion literature. This page summarises how the components in the
repository map onto the theory so you know where to look when extending the
library.

## Probability paths

Flow matching starts by defining a probability path \(p_t(x)\) that bridges a
simple reference distribution to the data distribution. WeatherFlow provides a
compact hierarchy of path classes in `weatherflow.path`:

- [`ProbPath`](../weatherflow/path/prob_path.py) – abstract base class defining
  `sample_path(...)` and `get_flow_vector(...)`.
- [`GaussianProbPath`](../weatherflow/path/gaussian_path.py) – implements the
  conditional Gaussian path \(p_t(x \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)\)
  with helper methods to compute the analytic score and vector field.
- `CondOTPath` – a thin wrapper for conditional optimal transport schedules.

Use these classes when you need analytic reference paths or to convert between
vector-field and score-based parameterisations.

## Vector-field models

The default flow-matching network is [`WeatherFlowMatch`](../weatherflow/models/flow_matching.py),
a ConvNeXt-inspired architecture with sinusoidal time embeddings, optional
multi-head attention, and an approximate divergence projection. It exposes
`forward(x, t)` returning the velocity field \(u_t(x)\) and
`compute_flow_loss(x0, x1, t)` for convenience.

Other models in `weatherflow.models` include:

- [`PhysicsGuidedAttention`](../weatherflow/models/physics_guided.py) – a compact
  baseline with channel attention and energy normalisation.
- [`StochasticFlowModel`](../weatherflow/models/stochastic.py) – a residual
  architecture that approximates stochastic drift and provides explicit mass and
  energy constraint terms.
- [`ScoreMatchingModel`](../weatherflow/models/score_matching.py) – learns the
  score \(\nabla_x \log p_t(x)\) for a supplied probability path.
- Conversion utilities [`vector_field_to_score`](../weatherflow/models/conversion.py)
  and [`score_to_vector_field`](../weatherflow/models/conversion.py) implement the
  relationships described in the lecture notes (Eq. 54–55).

## Training objective

The loss follows the rectified flow formulation: evaluate the model on an
interpolated state \(x_t = (1 - t)x_0 + t x_1\) and regress the time-independent
displacement \(x_1 - x_0\). A lightly time-reweighted MSE emphasises the middle
of the path, avoiding singular endpoints:

```python
from weatherflow.training.flow_trainer import compute_flow_loss

t = torch.rand(x0.size(0), device=x0.device)
x_t = torch.lerp(x0, x1, t.view(-1, 1, 1, 1))
v_t = model(x_t, t)
loss = compute_flow_loss(v_t, x0, x1, t, loss_type="mse", weighting="time")
```

`FlowTrainer` wraps this computation in a full training loop with optional
physics regularisation (`model.compute_physics_loss`) and mixed precision. You
can combine it with Weights & Biases logging, schedulers, or your own training
scripts as shown in the [Getting Started guide](getting_started.md).

## Integrating trajectories

To turn the learned velocity field into forecasts, integrate the ODE using one
of the solver utilities:

- [`WeatherFlowODE`](../weatherflow/models/flow_matching.py) – minimal wrapper
  that calls the flow model inside `torchdiffeq.odeint`.
- [`WeatherODESolver`](../weatherflow/solvers/ode_solver.py) – richer solver with
  constraint tracking, energy conservation monitoring, and configurable
  integration schemes.

Example:

```python
from weatherflow.models.flow_matching import WeatherFlowODE

ode_model = WeatherFlowODE(flow_model, solver_method="dopri5")
trajectory = ode_model(initial_state, torch.linspace(0.0, 1.0, 11))
```

## Putting it together

A minimal training loop looks like:

```python
import torch

for batch in train_loader:
    x0, x1 = batch["input"].to(device), batch["target"].to(device)
    t = torch.rand(x0.size(0), device=device)
    x_t = torch.lerp(x0, x1, t.view(-1, 1, 1, 1))
    v_t = model(x_t, t)
    losses = model.compute_flow_loss(x0, x1, t)
    optimizer.zero_grad()
    losses["total_loss"].backward()
    optimizer.step()
```

For more elaborate workflows, study `examples/weather_prediction.py` and the API
reference pages linked above. The
[Continuous Normalizing Flows note](advanced_topics/continuous_normalizing_flows.md)
contains additional theory, while the
[Flow Matching Implementation guide](advanced_topics/flow_matching_implementation.md)
explains how the spherical manifold and probability paths interact with the
solvers.
