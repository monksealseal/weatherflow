# Flow Matching Implementation Guide

This guide connects the high-level theory to the concrete implementation inside
WeatherFlow. It highlights how the probability paths, spherical geometry
routines, and solvers interact when you train and integrate a flow model.

## Probability paths and schedules

The core abstractions live in `weatherflow.path`:

- [`ProbPath`](../../weatherflow/path/prob_path.py) defines the interface for
  sampling points along a path and for retrieving the flow vector field at time
  \(t\).
- [`GaussianProbPath`](../../weatherflow/path/gaussian_path.py) implements the
  conditional Gaussian path \(\mathcal{N}(\alpha_t z, \beta_t^2 I)\) used
  throughout the examples. It exposes helper methods to compute analytic scores
  and vector fields, along with `sample_conditional(...)` for data-dependent
  draws.
- [`CondOTPath`](../../weatherflow/path/condot_path.py) demonstrates how to plug
  in alternative schedules such as conditional optimal transport.

The conversion utilities in `weatherflow.models.conversion` bridge score-based
and vector-field parameterisations by implementing Proposition 1 from the flow
matching lecture notes. Use them to wrap custom score networks or to export a
trained vector field as a score model for diffusion experiments.

## Working on the sphere

Global weather modelling requires respecting Earth's spherical geometry.
`weatherflow.manifolds.Sphere` provides numerically stable exponential and
logarithmic maps, geodesic interpolation, and parallel transport. These routines
operate on three-dimensional Cartesian coordinates and include dtype-aware
stability safeguards that prevent floating-point breakdowns when vectors are
nearly aligned.

The manifold utilities are used by the higher-level `WeatherFlowModel` and can
be incorporated into custom models when you need geodesic interpolation between
states or to project intermediate results back onto the sphere.

## ODE solver with physics constraints

[`WeatherODESolver`](../../weatherflow/solvers/ode_solver.py) is the workhorse
solver that integrates the learned velocity field while optionally enforcing
physical constraints:

- **Mass conservation** – approximated by subtracting the gradient of the
  divergence from the \(u\) and \(v\) wind components.
- **Energy conservation** – normalises the velocity magnitude relative to the
  current state energy and tracks drift statistics.
- **Vorticity control** – adds a rotational correction derived from the curl of
  the flow field.

The solver keeps running statistics on constraint violations and energy drift so
you can assess stability. It also exposes knobs for relative/absolute tolerances
and supports multiple integration methods (Dopri5, RK4, midpoint).

## Training loop integration

`FlowTrainer` stitches everything together:

1. Samples \(t \sim \mathcal{U}(0, 1)\) for each batch element.
2. Calls the model to evaluate the velocity field \(u_t(x)\).
3. Computes the straight-line target velocity and the loss via
   `compute_flow_loss` (with optional Huber/Smooth-L1 variants).
4. Applies physics regularisation by delegating to
   `model.compute_physics_loss(...)` when available.
5. Steps the optimiser with optional AMP scaling and scheduler updates.

Models such as `WeatherFlowMatch` embed light-weight divergence projection and
energy checks in their own loss functions, providing an additional layer of
physical awareness.

## Putting the pieces together

A typical experiment will:

1. Instantiate a probability path (Gaussian by default).
2. Train `WeatherFlowMatch` or another model with `FlowTrainer`.
3. Wrap the trained model with `WeatherFlowODE` or feed it into
   `WeatherODESolver` for trajectory integration.
4. Use `WeatherVisualizer` or the Plotly-based tools to analyse results.

The examples in `examples/weather_prediction.py` and the
[Advanced Usage](../advanced_usage.md) page provide concrete scripts that tie
these components into reproducible workflows.
