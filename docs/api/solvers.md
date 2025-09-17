# Solvers and Probability Paths

These utilities bridge trained velocity fields with numerical integrators and
provide reusable probability paths for score/flow conversions.

## Probability path abstractions

::: weatherflow.path.prob_path.ProbPath
    :members:
    :show-inheritance:

::: weatherflow.path.gaussian_path.GaussianProbPath
    :members:

::: weatherflow.path.condot_path.CondOTPath
    :members:

## WeatherODESolver

`WeatherODESolver` integrates a velocity field while enforcing optional mass,
energy, and vorticity constraints. It is a lower-level alternative to
`WeatherFlowODE` with detailed diagnostics and control over the numerical method.

::: weatherflow.solvers.ode_solver.WeatherODESolver
    :members:
