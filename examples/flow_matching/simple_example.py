import torch
from weatherflow.path import GaussianProbPath, CondOTPath
from weatherflow.models.score_matching import ScoreMatchingModel
from weatherflow.solvers.langevin import langevin_dynamics

# Example of creating a Gaussian probability path
alpha_schedule = lambda t: t  # Linear schedule from 0 to 1
beta_schedule = lambda t: 1 - t  # Linear schedule from 1 to 0
gaussian_path = GaussianProbPath(alpha_schedule, beta_schedule)

# Example of creating a CondOT path (simplified Gaussian path)
condot_path = CondOTPath()

# Create some example data
z = torch.randn(10, 4, 32, 32)  # Batch of 10, 4 channels, 32x32 grid
t = torch.rand(10)  # Time values between 0 and 1

# Sample from the conditional path
x = condot_path.sample_conditional(z, t)
print(f"Sampled data shape: {x.shape}")

# Compute conditional score
score = condot_path.get_conditional_score(x, z, t)
print(f"Score shape: {score.shape}")

# Compute conditional vector field
vector_field = condot_path.get_conditional_vector_field(x, z, t)
print(f"Vector field shape: {vector_field.shape}")

# Example of using Langevin dynamics
def simple_score_fn(x, t):
    return -x  # Simple score function pointing toward origin

x0 = torch.randn(2, 4, 8, 8)  # Small data for demonstration
result = langevin_dynamics(
    simple_score_fn,
    x0,
    n_steps=10,
    step_size=0.01,
    sigma=0.01
)
print(f"Langevin dynamics result shape: {result.shape}")
