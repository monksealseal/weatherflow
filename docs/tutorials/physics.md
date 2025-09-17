# Custom Physics Workflows

WeatherFlow lets you combine machine learning with physically inspired
constraints at multiple points in the pipeline. This tutorial walks through a
few common patterns.

## 1. Enable the built-in divergence projection

`WeatherFlowMatch` ships with an optional physics hook that nudges the velocity
field toward divergence-free behaviour by subtracting the gradient of the
computed divergence from the first two channels. Activate it with
`physics_informed=True`:

```python
from weatherflow.models import WeatherFlowMatch

model = WeatherFlowMatch(
    input_channels=4,
    hidden_dim=192,
    n_layers=6,
    use_attention=True,
    physics_informed=True,
)
```

Inside `forward(...)` the model calls `_apply_physics_constraints` before
returning the velocity field. You can override this method in a subclass to
implement domain-specific corrections.

## 2. Implement custom constraints

To add your own projection or penalty, subclass `WeatherFlowMatch` and override
`_apply_physics_constraints` and optionally `compute_flow_loss`:

```python
import torch

class BalancedFlowMatch(WeatherFlowMatch):
    def _apply_physics_constraints(self, v, x):
        v = super()._apply_physics_constraints(v, x)
        kinetic_energy = torch.sum(v**2, dim=(1, 2, 3), keepdim=True)
        scale = torch.sqrt(torch.sum(x**2, dim=(1, 2, 3), keepdim=True) / (kinetic_energy + 1e-6))
        return v * scale
```

Pair the model with `FlowTrainer(physics_regularization=True)` to include
`model.compute_physics_loss` in the total loss. The base implementation computes
mass and energy penalties; you can override it to add more terms.

## 3. Use physics-aware baselines

[`PhysicsGuidedAttention`](../../weatherflow/models/physics_guided.py) embeds
physics considerations directly in the network:

- Sinusoidal time embeddings allow the model to condition on forecast lead time.
- Channel attention emphasises physically important variables.
- Energy normalisation rescales outputs to match the input energy.

Swap it in when you want a smaller model that still respects fundamental
constraints.

```python
from weatherflow.models import PhysicsGuidedAttention

model = PhysicsGuidedAttention(
    channels=sample["input"].shape[1],
    hidden_dim=64,
    num_layers=4,
    dropout=0.1,
)
```

## 4. Configure FlowTrainer for physics-aware training

`FlowTrainer` exposes two knobs for incorporating physics penalties:

- `physics_regularization=True` enables the additional term.
- `physics_lambda` controls its contribution to the total loss.

```python
import torch
from weatherflow.training.flow_trainer import FlowTrainer

trainer = FlowTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
    physics_regularization=True,
    physics_lambda=0.2,
)
```

Monitor `metrics["physics_loss"]` (or `"val_physics_loss"`) to gauge the
strength of the constraint during training.

## 5. Adjust constraint weights during integration

When integrating trajectories, `WeatherODESolver` can enforce mass, energy, and
vorticity constraints. Tune them per use case:

```python
from weatherflow.solvers import WeatherODESolver

solver = WeatherODESolver(
    physics_constraints=True,
    constraint_types=["mass", "energy"],
    constraint_weights={"mass": 1.0, "energy": 0.1},
)
```

The returned `stats` dictionary contains average divergence and relative energy
errors so you can adjust the weights dynamically.

## 6. Next steps

- Combine the techniques above with `WeatherVisualizer.plot_flow_vectors` to
  inspect how the corrected velocity field behaves spatially.
- Experiment with the [educational toolkit](../../weatherflow/education/graduate_tool.py)
  to build intuition for balance constraints before embedding them in the model.
- Review the [`WeatherODESolver` API](../api/solvers.md) for the full list of
  supported constraint types and solver options.
