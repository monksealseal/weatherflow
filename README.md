# WeatherFlow: A Deep Learning Library for Weather Prediction

## Introduction

WeatherFlow is a Python library built on PyTorch that aims to provide a flexible and extensible framework for developing and evaluating deep learning models for weather prediction. It leverages recent advancements in flow matching and incorporates design principles for handling spatiotemporal data, particularly geopotential height fields.  This documentation covers the package structure, API, usage examples, and development guidelines.

## Key Features

*   **Modular Design:** The library is organized into modules for data loading, model definition, training, evaluation, and visualization, making it easy to extend and customize.
*   **ERA5 Data Integration:** Includes a `WeatherDataset` class for easy loading and preprocessing of ERA5 reanalysis data, a standard dataset for weather prediction research.  It supports both local netCDF files and direct loading from the WeatherBench 2 Google Cloud Storage (requires authentication).
*   **Flow Matching Models:** Implements a `WeatherFlowMatch` model that utilizes the principles of flow matching for generating weather predictions.
*   **Physics-Guided Architecture (Planned):** Future versions will incorporate physics-informed components.
*   **Configurable Training:** Provides a flexible `train_model` function with options for various optimizers, learning rate schedulers, and early stopping.
*   **Comprehensive Evaluation:** Includes functions for calculating standard weather prediction metrics (RMSE, ACC) and visualizing predictions.
*   **Extensible Design:** The modular structure allows users to easily add custom models, data processing steps, and evaluation metrics.

## Installation


**Dependencies:**

*   Python >= 3.8
*   torch >= 1.9
*   xarray
*   numpy
*   matplotlib
*   cartopy (for visualizations)
*   fsspec (for Google Cloud Storage access)
*   gcsfs (for Google Cloud Storage access)
*   tqdm (for progress bars)
*   wandb (optional, for experiment tracking)
*   scipy
*   netCDF4
*   h5py

## Quick Start

Here's a quick example of how to load data, train a model, and visualize predictions:


## API Reference

### `weatherflow.data`

*   **`WeatherDataset(data_path, variables, years, input_length=1, lead_time=1)`:**
    *   `data_path`: Path to the directory containing ERA5 netCDF files (one per year).
    *   `variables`: List of variable names (e.g., `['z', 't']` for geopotential and temperature).  Use the short names.
    *   `years`: A list of years (integers) to include in the dataset, or the string 'all' to include all available .nc files.
    *   `input_length`: The number of timesteps to include in the input sequence (default: 1).
    *   `lead_time`: The number of timesteps between the last input timestep and the target timestep (default: 1, meaning a 6-hour forecast).

*   **`create_data_loaders(variables, pressure_levels, data_dir, train_years, val_years, batch_size, num_workers)`:**  A convenience function to create PyTorch `DataLoader` instances for training and validation.  It handles splitting the data by year.

### `weatherflow.models`

*   **`WeatherFlowMatch(input_dim, hidden_dim)`:**
    *   `input_dim`: The flattened dimension of a single input timestep (e.g., 64 * 32 = 2048 for a 64x32 grid).
    *   `hidden_dim`: The hidden dimension of the internal layers of the model.
    *   `forward(self, x, t)`: Performs a forward pass.  `x` is the input tensor (shape: `[batch, channels, height, width]`), and `t` is the time tensor (shape: `[batch]`).  Returns the predicted velocity field.
    *   `compute_loss`: Includes a magnitude loss in addition to the direction loss.

### `weatherflow.utils`
* **`plot_prediction_comparison`**: Plots the difference between the true and predicted values.
* **`create_prediction_animation`**: Creates an animation of predictions.
* **`plot_raw_fields`**: Plots the data before any transformation.
* **`calculate_metrics`**: Calculates root mean squared error and anomaly correlation coefficient.
* **`generate_wb2_predictions`**: Prepares model output to be in the correct weatherbench2 format.
* **`evaluate_saved_predictions`**: Performs weatherbench2 evaluation.

## Advanced Usage

*   **Custom Models:** You can create your own models by subclassing `nn.Module` and implementing the `forward` method.  You'll likely want to modify the `train_model` function to work with your custom model.

*   **Custom Loss Functions:**  You can define custom loss functions beyond the standard MSE loss.

*   **Data Augmentation:**  Add data augmentation to the `WeatherDataset` (e.g., random rotations, flips) to improve model robustness.

*   **Distributed Training:**  Adapt the training loop to use PyTorch's distributed training capabilities for larger datasets and models.

*   **More Sophisticated Visualization:** Use the `WeatherVisualizer` class as a starting point to create more advanced visualizations, such as animations of weather patterns over time, or plots that highlight specific regions or features.

This documentation gives you a solid foundation for using and extending the `weatherflow` package.  Remember to refer to the docstrings within the code for more detailed information on specific functions and classes.  Good luck with your weather prediction projects!
