# Flow Matching and Diffusion Models

This module implements flow matching and diffusion models based on MIT lecture notes on Flow Matching and Diffusion Models.

## Components

### Path Module

- `GaussianProbPath`: Implements Gaussian conditional probability paths with customizable schedules
- `CondOTPath`: Implements Conditional Optimal Transport paths (a special case of Gaussian paths)

### Models Module

- `ScoreMatchingModel`: Implements score matching for diffusion models
- Conversion utilities between vector fields and score functions

### Solvers Module

- `langevin_dynamics`: Implements Langevin dynamics sampling
- Enhanced ODE solvers with Heun's method and ODE-to-SDE conversion

### Training Module

- `FlowMatchTrainer`: Implements training for flow matching models

## Usage Examples

See the examples directory for usage examples.
