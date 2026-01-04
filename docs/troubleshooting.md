# Troubleshooting Guide

This guide helps you resolve common issues when working with WeatherFlow.

## Installation Issues

### Module Not Found: 'torch'

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Install PyTorch first
pip install torch torchvision

# Then install WeatherFlow
pip install -e .
```

For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Module Not Found: 'weatherflow'

**Symptom:**
```
ModuleNotFoundError: No module named 'weatherflow'
```

**Solution:**
Install the package from the repository root:
```bash
cd /path/to/weatherflow
pip install -e .
```

### Import Errors in Examples

**Symptom:**
```python
from weatherflow.path import GaussianProbPath  # ImportError
```

**Solution:**
The example files have been fixed to use correct imports. If you're using an older version or encounter import issues, add path setup at the top of your script:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from weatherflow.path.gaussian_path import GaussianProbPath
from weatherflow.path.condot_path import CondOTPath
```

## Data Access Issues

### ERA5 Data Not Accessible

**Symptom:**
```
Unable to access ERA5 data from WeatherBench2
```

**Solutions:**

1. **Check internet connection:**
   ```bash
   ping -c 4 8.8.8.8
   ```

2. **Install cloud storage dependencies:**
   ```bash
   pip install gcsfs zarr fsspec
   ```

3. **Download data locally:**
   - Download from [WeatherBench2](https://github.com/google-research/weatherbench2)
   - Point to local path:
   ```python
   dataset = ERA5Dataset(
       data_path='/local/path/to/era5_data.nc',
       variables=['z', 't'],
       pressure_levels=[500]
   )
   ```

4. **Set up GCS authentication (if needed):**
   ```bash
   export GCSFS_TOKEN=/path/to/credentials.json
   ```

### Zarr Access Errors

**Symptom:**
```
ValueError: Cannot read from zarr store
```

**Solution:**
```bash
# Install zarr with all dependencies
pip install "zarr[all]"

# For read-only access, anonymous mode should work:
import zarr
store = zarr.open('gs://weatherbench2/...', mode='r')
```

## Training Issues

### CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python examples/weather_prediction.py --batch-size 2
   ```

2. **Reduce model size:**
   ```python
   model = WeatherFlowMatch(
       input_channels=4,
       hidden_dim=128,  # Instead of 256
       n_layers=4,      # Instead of 6
   )
   ```

3. **Enable gradient checkpointing:**
   ```python
   trainer = FlowTrainer(
       model=model,
       use_amp=True,  # Mixed precision
       # ... other args
   )
   ```

4. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Disk Space Issues

**Symptom:**
```
OSError: [Errno 28] No space left on device
```

**Solutions:**

1. **Clean up temporary files:**
   ```bash
   rm -rf /tmp/*
   rm -rf ~/.cache/pip
   ```

2. **Use a different checkpoint directory:**
   ```bash
   python examples/weather_prediction.py --checkpoint-dir /path/with/more/space
   ```

3. **Disable data caching:**
   ```python
   dataset = ERA5Dataset(cache_data=False, ...)
   ```

4. **Check disk space:**
   ```bash
   df -h
   ```

### Loss is NaN or Exploding

**Symptom:**
Training loss becomes NaN or grows exponentially

**Solutions:**

1. **Reduce learning rate:**
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Instead of 1e-4
   ```

2. **Use gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Check for data issues:**
   ```python
   # Verify data is normalized
   batch = next(iter(train_loader))
   print(f"Min: {batch['input'].min()}, Max: {batch['input'].max()}")
   print(f"Mean: {batch['input'].mean()}, Std: {batch['input'].std()}")
   ```

4. **Use a more robust loss function:**
   ```python
   trainer = FlowTrainer(
       model=model,
       loss_type='huber',  # Instead of 'mse'
       # ... other args
   )
   ```

### Training is Very Slow

**Symptom:**
Training takes too long per epoch

**Solutions:**

1. **Use GPU:**
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = model.to(device)
   ```

2. **Increase num_workers:**
   ```python
   train_loader, val_loader = create_data_loaders(
       num_workers=4,  # Instead of 0
       # ... other args
   )
   ```

3. **Enable mixed precision:**
   ```python
   trainer = FlowTrainer(
       model=model,
       use_amp=True,  # Automatic Mixed Precision
       # ... other args
   )
   ```

4. **Use DataLoader pin_memory:**
   ```python
   train_loader = DataLoader(dataset, pin_memory=True)
   ```

## Visualization Issues

### Cartopy Not Available

**Symptom:**
```
ModuleNotFoundError: No module named 'cartopy'
```

**Solution:**
```bash
# Using conda (recommended)
conda install -c conda-forge cartopy

# Using pip (may require system dependencies)
pip install cartopy
```

### Matplotlib Backend Errors

**Symptom:**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

**Solution:**
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

### Animation Not Working

**Symptom:**
Animations don't display in Jupyter notebooks

**Solution:**
```python
# In Jupyter notebooks
%matplotlib inline
from IPython.display import HTML

# Create animation
anim = visualizer.create_prediction_animation(...)
HTML(anim.to_html5_video())
```

## Model Issues

### Checkpoint Loading Errors

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

1. **Check model architecture matches:**
   ```python
   # Model used for training
   model = WeatherFlowMatch(input_channels=4, hidden_dim=256, n_layers=6)
   # Must match when loading
   ```

2. **Use strict=False for partial loading:**
   ```python
   model.load_state_dict(torch.load('checkpoint.pt'), strict=False)
   ```

3. **Inspect checkpoint contents:**
   ```python
   checkpoint = torch.load('checkpoint.pt')
   print(checkpoint.keys())
   ```

### ODE Solver Failures

**Symptom:**
```
RuntimeError: ODE solver failed to converge
```

**Solutions:**

1. **Use more tolerant solver:**
   ```python
   ode_solver = WeatherFlowODE(
       flow_model=model,
       solver_method='euler',  # Instead of 'dopri5'
       rtol=1e-3,  # Less strict
       atol=1e-3,
   )
   ```

2. **Reduce time span:**
   ```python
   times = torch.linspace(0, 1, 5)  # Instead of 100 steps
   ```

3. **Check model outputs:**
   ```python
   with torch.no_grad():
       v = model(x, t)
       print(f"Velocity field stats: min={v.min()}, max={v.max()}, mean={v.mean()}")
   ```

## Development Issues

### Pre-commit Hook Failures

**Symptom:**
```
black....................................................................Failed
```

**Solution:**
```bash
# Format code
black .

# Sort imports
isort .

# Run pre-commit manually
pre-commit run --all-files
```

### Test Failures

**Symptom:**
```
pytest tests/ fails
```

**Solutions:**

1. **Install test dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run specific test:**
   ```bash
   pytest tests/test_models.py::test_weatherflow_match -v
   ```

3. **Check for dependency issues:**
   ```bash
   pip list | grep -E "torch|numpy|xarray"
   ```

## Getting More Help

If you're still experiencing issues:

1. **Check documentation:**
   - [TRAINING_INSTRUCTIONS.txt](../TRAINING_INSTRUCTIONS.txt)
   - [Advanced Usage Guide](advanced_usage.md)
   - [API Reference](api_reference.md)

2. **Search existing issues:**
   - [GitHub Issues](https://github.com/monksealseal/weatherflow/issues)

3. **File a new issue:**
   - Include error messages
   - Provide minimal reproducible example
   - Specify your environment (OS, Python version, GPU, etc.)

4. **Enable verbose logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # For data loading
   dataset = ERA5Dataset(..., verbose=True)
   ```
