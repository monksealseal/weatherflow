# Models API Reference

WeatherFlow provides several neural architectures for flow matching and related
modelling tasks. All models are implemented with PyTorch and expose familiar
`forward(...)` methods.

## Flow matching networks

`WeatherFlowMatch` is the flagship architecture used throughout the examples. It
produces a velocity field conditioned on the current state and a scalar time. A
companion class `WeatherFlowODE` wraps a trained flow model and integrates it
with `torchdiffeq`.

::: weatherflow.models.flow_matching.WeatherFlowMatch
    :members:
    :show-inheritance:

::: weatherflow.models.flow_matching.WeatherFlowODE
    :members:

::: weatherflow.models.flow_matching.ConvNextBlock
    :members:
    :show-inheritance:

## Physics-guided attention baseline

`PhysicsGuidedAttention` offers a compact residual architecture with channel
attention, sinusoidal time embeddings, and energy normalisation. It is useful
for unit tests and quick experiments.

::: weatherflow.models.physics_guided.PhysicsGuidedAttention
    :members:

## Stochastic flow surrogate

`StochasticFlowModel` approximates a stochastic flow while still satisfying the
`BaseWeatherModel` constraints. It includes explicit implementations of mass and
energy conservation penalties.

::: weatherflow.models.stochastic.StochasticFlowModel
    :members:
    :show-inheritance:

## Dense neural ODE model

`WeatherFlowModel` couples a learned velocity network with the general-purpose
`WeatherODESolver` and the spherical manifold utilities. It serves as a template
for building denser models that operate on flattened grids.

::: weatherflow.models.weather_flow.WeatherFlowModel
    :members:

## Base classes and score utilities

The base class defines the physics-aware interface shared by multiple models.
Score-matching helpers make it easy to interoperate with diffusion techniques.

::: weatherflow.models.base.BaseWeatherModel
    :members:
    :show-inheritance:

::: weatherflow.models.score_matching.ScoreMatchingModel
    :members:

::: weatherflow.models.conversion.vector_field_to_score

::: weatherflow.models.conversion.score_to_vector_field
