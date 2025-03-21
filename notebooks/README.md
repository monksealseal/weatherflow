# WeatherFlow Notebooks

This directory contains Jupyter notebooks for working with the WeatherFlow library.

## Running the Notebooks

There are two ways to run these notebooks:

### Option 1: Install the Package

1. Install the WeatherFlow package from the repository root:
   ```bash
   pip install -e ..
   ```

2. Open and run the notebooks normally.

### Option 2: Use the Path Fix

1. Install the notebook dependencies:
   ```bash
   pip install -r notebook_requirements.txt
   ```

2. Run the fix script to add the proper import paths to all notebooks:
   ```bash
   python fix_notebook_imports.py
   ```

3. Open and run the notebooks normally.

## Notebooks Overview

- `complete-data-exploration.ipynb`: Data exploration and visualization of ERA5 datasets.
- `flow-matching-basics.ipynb`: Introduction to flow matching concepts.
- `model-training-notebook.ipynb`: Training WeatherFlow models for prediction.
- `prediction-visualization-notebook.ipynb`: Visualizing weather predictions.
- `weatherbench-evaluation-notebook.ipynb`: Evaluating models on the WeatherBench benchmark.

## Troubleshooting

If you encounter import errors or other issues running the notebooks:

1. Make sure you're using a Python environment with all dependencies installed:
   ```bash
   pip install -r notebook_requirements.txt
   ```

2. Check that your Jupyter kernel is using the correct environment.

3. Try running the notebooks with the path fix:
   ```bash
   python fix_notebook_imports.py
   ```

4. If all else fails, install the package from the repository root:
   ```bash
   pip install -e ..
   ```