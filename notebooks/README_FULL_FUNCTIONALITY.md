# Getting Full Functionality from WeatherFlow Notebooks

This guide explains how to run the WeatherFlow notebooks with full functionality, including access to real ERA5 weather data.

## Complete Guide Notebook

We've created a comprehensive demonstration notebook at `notebooks/complete_guide.ipynb` that covers the entire workflow:

1. Loading real ERA5 reanalysis data
2. Preprocessing and exploring weather data
3. Training flow matching models
4. Making predictions
5. Visualizing results with advanced plotting techniques
6. Creating animations of weather evolution

## Requirements for Full Functionality

To run the notebooks with all features:

### 1. Environment Setup

Create a virtual environment with all necessary dependencies:

```bash
# Using conda
conda create -n weatherflow python=3.8
conda activate weatherflow

# Or using venv
python -m venv weatherflow-env
source weatherflow-env/bin/activate  # On Windows: weatherflow-env\Scripts\activate
```

### 2. Install Required Packages

```bash
# Install torch first (CUDA version if you have a GPU)
pip install torch torchvision

# Install weatherflow in development mode
pip install -e .

# Install other dependencies
pip install matplotlib numpy pandas xarray ipykernel jupyterlab tqdm cartopy
pip install netCDF4 zarr fsspec gcsfs torchdiffeq
```

### 3. Access to ERA5 Data

The notebooks can access ERA5 data in several ways:

#### Option A: Use WeatherBench 2 Cloud Storage (Easiest)
The notebooks are configured to access ERA5 data from the WeatherBench 2 Google Cloud Storage bucket. This requires:
- Internet access
- `gcsfs` package installed

#### Option B: Download Local ERA5 Data
Download data from ECMWF or WeatherBench:
- [WeatherBench 2 Data Guide](https://github.com/google-research/weatherbench2)
- [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)

Update the paths in the notebook to point to your local data files.

### 4. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name weatherflow --display-name "Python (WeatherFlow)"
```

### 5. Launch Jupyter

```bash
jupyter lab
```

## Configuring Fallbacks

The notebooks include fallback options when external data or certain dependencies aren't available:

1. **Mock Data Generation**: If ERA5 data can't be accessed, synthetic data is generated
2. **Simple Interpolation**: If ODE solving fails, linear interpolation is used
3. **Basic Visualization**: If cartopy isn't available, simple matplotlib plots are used

## Troubleshooting

### Data Access Issues
If you encounter errors accessing ERA5 data:
- Check your internet connection
- Verify that `gcsfs` and `zarr` are installed
- Try downloading a small subset of ERA5 data locally

### GPU Acceleration
To use GPU acceleration:
- Install the CUDA version of PyTorch
- The code automatically detects and uses available GPUs
- For CPU-only systems, smaller models and datasets are recommended

### Visualization Problems
If you have issues with visualization:
- Install cartopy for geographic plotting: `conda install -c conda-forge cartopy`
- For animations, make sure you're running in Jupyter Lab/Notebook
- Consider lowering resolution of plots if memory is an issue

## Getting Help

If you encounter issues or have questions:
- Check the WeatherFlow documentation in the `docs/` directory
- Look at example scripts in the `examples/` directory
- File an issue on the GitHub repository

## Best Practices

For the best experience:
- Use a machine with at least 8GB RAM and preferably a GPU
- For large datasets, consider using a subset of the data initially
- Adjust model complexity based on your hardware capabilities
- Save trained models to avoid retraining