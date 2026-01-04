# WeatherFlow Copilot Instructions

## Project Overview

WeatherFlow is a Python library for weather prediction using flow matching techniques. It's built on PyTorch and provides a flexible framework for developing weather models that integrate with ERA5 reanalysis data and incorporate physics-guided neural network architectures.

**Version:** 0.4.2  
**Python:** 3.8+  
**License:** MIT

## Architecture & Key Components

### Core Modules

- **`weatherflow/data/`** - ERA5 data loading and preprocessing
  - `ERA5Dataset`: Main dataset class for loading ERA5 reanalysis data
  - `create_data_loaders()`: Helper for creating train/val data loaders
  - Supports both local netCDF files and WeatherBench2 remote data

- **`weatherflow/models/`** - Neural network models
  - `WeatherFlowMatch`: Primary flow matching model for weather prediction
  - `WeatherFlowODE`: ODE solver wrapper for generating predictions
  - `PhysicsGuidedAttention`: Attention mechanisms with physical constraints
  - `StochasticFlowModel`: Stochastic variations for ensemble forecasting

- **`weatherflow/physics/`** - Physics-informed constraints and losses
  - Conservation laws (potential vorticity, mass, energy)
  - Geostrophic balance constraints
  - Energy spectra calculations

- **`weatherflow/utils/`** - Visualization and evaluation utilities
  - `WeatherVisualizer`: Plotting and animation tools
  - `FlowVisualizer`: Vector field visualizations
  - `WeatherMetrics`: Evaluation metrics for weather predictions

- **`weatherflow/training/`** - Training infrastructure
  - `FlowTrainer`: Main training loop with physics losses
  - `compute_flow_loss()`: Flow matching loss computation

- **`weatherflow/education/`** - Educational tools
  - `GraduateAtmosphericDynamicsTool`: Interactive dashboards for teaching

- **`weatherflow/server/`** - FastAPI web service
- **`frontend/`** - React-based interactive dashboard

### Project Structure

```
weatherflow/
├── weatherflow/          # Main Python package
│   ├── data/            # Data loading modules
│   ├── models/          # Neural network architectures
│   ├── physics/         # Physics constraints
│   ├── utils/           # Utilities and visualization
│   ├── training/        # Training loops
│   └── ...
├── tests/               # Test suite
├── examples/            # Example scripts
├── applications/        # Real-world applications (renewable energy, etc.)
├── model_zoo/          # Pre-trained model infrastructure
├── notebooks/          # Jupyter notebooks
├── frontend/           # React web interface
└── docs/               # MkDocs documentation
```

## Coding Conventions

### Style Guidelines

- **Follow PEP 8** for Python code style
- **Line length:** 88 characters (Black formatter default)
- **Use Black** for code formatting
- **Use isort** with Black profile for import sorting
- **Type hints:** Required for all function signatures
- **Docstrings:** Use Google-style docstrings

### Code Style Tools

Run these before committing:
```bash
black .
isort --profile black .
flake8
mypy
```

Or install pre-commit hooks:
```bash
pre-commit install
```

### Import Organization

Imports should be organized by isort with Black profile:
1. Standard library imports
2. Third-party imports
3. Local application imports

### Naming Conventions

- **Classes:** PascalCase (e.g., `WeatherFlowMatch`, `ERA5Dataset`)
- **Functions/methods:** snake_case (e.g., `create_data_loaders`, `compute_flow_loss`)
- **Constants:** UPPER_SNAKE_CASE (e.g., `R_EARTH`)
- **Private members:** Prefix with underscore (e.g., `_apply_physics_constraints`)

### Type Hints

Always use type hints for function signatures:

```python
def create_data_loaders(
    variables: List[str],
    pressure_levels: List[int],
    train_slice: Tuple[str, str],
    val_slice: Tuple[str, str],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_flow_loss(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute flow matching loss.

    Args:
        x0: Initial state tensor of shape [batch, channels, lat, lon]
        x1: Target state tensor of shape [batch, channels, lat, lon]
        t: Time values of shape [batch]

    Returns:
        Dictionary containing loss components:
            - 'total_loss': Combined loss value
            - 'flow_loss': Flow matching loss
            - 'physics_loss': Physics constraint violation (if enabled)

    Raises:
        ValueError: If input shapes are incompatible
    """
    ...
```

## Testing Practices

### Test Structure

- Tests located in `tests/` directory
- Test files named `test_*.py`
- Use pytest framework
- Test functions named `test_*`

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=weatherflow tests/
```

### Test Patterns

1. **Use fixtures** for common test data:
```python
@pytest.fixture
def sample_wind_field():
    return torch.randn(2, 2, 32, 64)  # [batch, channels, lat, lon]
```

2. **Test shape invariants:**
```python
def test_model_output_shape():
    model = WeatherFlowMatch(input_channels=4, hidden_dim=32)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == x.shape
```

3. **Skip imports if dependencies missing:**
```python
torch = pytest.importorskip("torch")
```

## Development Workflow

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Build and Test Commands

```bash
# Format code
black .
isort --profile black .

# Lint
flake8
mypy

# Run tests
pytest tests/

# Build documentation
pip install -r requirements-docs.txt
mkdocs serve  # View at http://localhost:8000

# Run web service
uvicorn weatherflow.server.app:app --reload --port 8000

# Run React frontend (in frontend/ directory)
cd frontend
npm install
npm run dev
npm test
npm run build
```

## Key Design Patterns

### Lazy Loading

The package uses lazy loading for optional dependencies to avoid import errors in minimal environments:

```python
# In __init__.py
def __getattr__(name: str):
    if name in _lazy_exports:
        module_path, attr_name = _lazy_exports[name]
        attr = _import_attr(module_path, attr_name)
        globals()[name] = attr
        return attr
```

### Physics-Informed Models

Models can include physics constraints:

```python
model = WeatherFlowMatch(
    input_channels=4,
    hidden_dim=256,
    physics_informed=True,  # Enable physics constraints
    grid_size=(32, 64)      # Required for physics calculations
)
```

### Flow Matching Training

Standard training pattern:

```python
# Create model and ODE solver
flow_model = WeatherFlowMatch(...)
ode_model = WeatherFlowODE(flow_model)

# Training loop
for x0, x1 in data_loader:
    t = torch.rand(x0.size(0))
    loss = model.compute_flow_loss(x0, x1, t)['total_loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate predictions
times = torch.linspace(0, 1, steps)
predictions = ode_model(x0, times)
```

## Dependencies

### Core Dependencies

- **PyTorch** ≥2.0.0 - Deep learning framework
- **NumPy** ≥1.24.0, <2.0.0 - Numerical computing
- **xarray** ≥2023.1.0 - Multi-dimensional labeled arrays
- **torchdiffeq** ≥0.2.3 - ODE solvers for PyTorch
- **matplotlib** ≥3.7.0 - Plotting
- **cartopy** ≥0.21 - Geospatial data processing
- **netCDF4** ≥1.5.7 - NetCDF file handling

### Optional Dependencies

- **FastAPI** + **uvicorn** - Web service
- **plotly** ≥5.18.0 - Interactive visualizations
- **wandb** - Experiment tracking

### Development Dependencies

- **pytest** ≥7.3.1 - Testing framework
- **pytest-cov** ≥4.1.0 - Coverage reporting
- **black** ≥23.3.0 - Code formatter
- **isort** ≥5.12.0 - Import sorter
- **flake8** ≥6.0.0 - Linter
- **mypy** ≥1.3.0 - Type checker
- **pre-commit** ≥3.3.3 - Git hooks

## Common Patterns and Best Practices

### Data Loading

```python
from weatherflow.data import create_data_loaders

train_loader, val_loader = create_data_loaders(
    variables=['z', 't', 'u', 'v'],  # Variables to load
    pressure_levels=[500, 850],       # Pressure levels in hPa
    train_slice=('2015', '2016'),     # Training period
    val_slice=('2017', '2017'),       # Validation period
    batch_size=32,
    normalize=True                     # Apply normalization
)
```

### Model Initialization

```python
from weatherflow.models import WeatherFlowMatch

model = WeatherFlowMatch(
    input_channels=4,           # Number of input variables
    hidden_dim=256,             # Hidden layer dimension
    n_layers=6,                 # Number of layers
    use_attention=True,         # Use attention mechanism
    physics_informed=True,      # Apply physics constraints
    grid_size=(32, 64)          # (lat, lon) grid dimensions
)
```

### Visualization

```python
from weatherflow.utils import WeatherVisualizer

visualizer = WeatherVisualizer()

# Plot comparison
visualizer.plot_comparison(
    true_data={'z': true_geopotential},
    pred_data={'z': predicted_geopotential},
    var_name='z',
    title="Geopotential Height Comparison"
)

# Create animation
visualizer.create_prediction_animation(
    predictions=predictions,
    var_name='temperature',
    save_path='forecast.gif'
)
```

## Error Handling

### Common Patterns

1. **Validate input shapes:**
```python
if x0.shape != x1.shape:
    raise ValueError(f"Shape mismatch: x0={x0.shape}, x1={x1.shape}")
```

2. **Check for required dependencies:**
```python
try:
    import cartopy
except ImportError as e:
    raise ImportError(
        "cartopy is required for this feature. "
        "Install with: pip install cartopy"
    ) from e
```

3. **Provide helpful error messages:**
```python
if len(variables) == 0:
    raise ValueError(
        "At least one variable must be specified. "
        "Available variables: ['z', 't', 'u', 'v', 'q']"
    )
```

## Performance Considerations

1. **Use batch processing** for efficient GPU utilization
2. **Enable mixed precision training** when appropriate
3. **Leverage lazy loading** to minimize import overhead
4. **Cache expensive computations** (e.g., grid coordinates)
5. **Use `torch.no_grad()`** during inference

## Contributing Guidelines

1. **Create a feature branch** from main
2. **Write tests** for new features
3. **Update documentation** if adding new APIs
4. **Run pre-commit hooks** before committing
5. **Keep PRs focused** on a single feature/fix
6. **Follow existing code patterns** in the module you're editing

## Additional Resources

- **Documentation:** Build locally with `mkdocs serve`
- **Examples:** See `examples/` directory for usage patterns
- **Applications:** See `applications/` for real-world use cases
- **Model Zoo:** See `model_zoo/` for pre-trained model infrastructure
- **Notebooks:** See `notebooks/` for interactive tutorials

## Questions or Issues?

- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review examples in `examples/` directory
- See CONTRIBUTING.md for contribution guidelines
