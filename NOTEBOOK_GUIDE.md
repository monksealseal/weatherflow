# WeatherFlow Notebook Guide

This guide provides instructions for running the Jupyter notebooks in this repository, even if you don't have all the dependencies installed.

## Running the Notebooks

### Option 1: Run with JupyterLab/Jupyter Notebook (if installed)

If you have JupyterLab or Jupyter Notebook installed:

1. The notebooks have been fixed to include the proper path for importing the weatherflow package.
2. Run Jupyter:
   ```
   jupyter notebook
   ```
   or
   ```
   jupyter lab
   ```
3. Navigate to the notebook you want to run (in the `notebooks/` directory).
4. You might still need to install the required dependencies listed in the next section.

### Option 2: Run in Google Colab

Google Colab is a free cloud service that allows you to run Jupyter notebooks without any local setup:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click on "File" > "Upload Notebook"
3. Upload one of the notebooks from this repository
4. Upload the repository code by running this in a code cell:
   ```python
   # Mount Google Drive (if you want to save outputs)
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Clone the repository
   !git clone https://github.com/monksealseal/weatherflow.git
   
   # Add repo to Python path
   import sys
   sys.path.append('/content/weatherflow')
   
   # Install dependencies
   !pip install torch xarray matplotlib cartopy netCDF4 tqdm wandb zarr fsspec gcsfs torchdiffeq
   ```
5. Run the notebook

### Option 3: Run in Binder (No Installation Required)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/monksealseal/weatherflow/HEAD)

Click the Binder badge above to launch this repository in a ready-to-use environment.

## Required Dependencies

The notebooks require these main dependencies:

```
torch>=2.0.0
numpy>=1.24.0
xarray>=0.19.0
pandas>=1.5.0 
matplotlib>=3.7.0
cartopy>=0.20.0
tqdm>=4.60.0
zarr>=2.14.2
fsspec>=2023.9.0
gcsfs>=2023.9.0
torchdiffeq>=0.2.3
```

## Notebook Overview

Here's a brief overview of the available notebooks:

1. **flow-matching-basics.ipynb**
   - Introduction to flow matching concepts and basic implementation
   - Good starting point for understanding the core principles

2. **model-training-notebook.ipynb**
   - Demonstrates how to train weather prediction models
   - Contains examples of model configuration and training loops

3. **prediction-visualization-notebook.ipynb**
   - Shows how to generate and visualize weather predictions
   - Includes maps, comparisons, and animations

4. **complete-data-exploration.ipynb**
   - Explores ERA5 climate data and demonstrates preprocessing steps
   - Contains data loading, normalization, and visualization

5. **weatherbench-evaluation-notebook.ipynb**
   - Evaluates models on the WeatherBench benchmark
   - Compares model performance with metrics and visualizations

## Troubleshooting

If you encounter issues running the notebooks:

1. **ImportError for weatherflow package**:
   - Ensure you're running the notebooks from the repository root
   - Check that the path import code at the beginning of each notebook is executed
   - Try running `python direct_fix_notebooks.py` again to fix imports

2. **Missing dependencies**:
   - Install the required dependencies using:
     ```
     pip install torch xarray matplotlib cartopy netCDF4 tqdm wandb zarr fsspec gcsfs torchdiffeq
     ```

3. **Error loading ERA5 data**:
   - The notebooks try to load data from Google Cloud Storage
   - If you don't have internet access, data loading will fail
   - Consider downloading sample data locally from WeatherBench

4. **Memory issues**:
   - The models may require significant memory
   - Try reducing batch sizes or model dimensions
   - For large datasets, use data subsetting