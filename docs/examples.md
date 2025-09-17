# Examples

The `examples/` directory contains runnable scripts and notebooks that
demonstrate the library in action. Use them as templates for your own projects or
as quick regression tests when upgrading dependencies.

## `weather_prediction.py`

End-to-end ERA5 flow-matching pipeline that covers:

1. Loading data with `create_data_loaders`.
2. Configuring `WeatherFlowMatch` (with optional attention/physics flags).
3. Training with Adam and a ReduceLROnPlateau scheduler.
4. Saving checkpoints, metrics, and generated predictions.
5. Plotting geopotential height comparisons and error maps.

Run it with:

```bash
python examples/weather_prediction.py --variables z t --pressure-levels 500 \
    --train-years 2015 2016 --val-years 2017 --epochs 10 --use-attention \
    --physics-informed --save-model --save-results
```

The script writes outputs under `results/<timestamp>/` and logs metrics to the
console.

## `flow_matching/simple_example.py`

Minimal synthetic demonstration that trains a small `WeatherFlowMatch` on random
data. It is ideal for debugging custom layers or verifying new environments can
load PyTorch + TorchDiffEq correctly.

## `skewt_3d_visualizer.py`

Shows how to parse a SKEW-T sounding image with `SkewTImageParser` and render an
interactive 3D curtain plot using `SkewT3DVisualizer`. The resulting HTML file
opens in any browser and is perfect for presentations.

## `visualization_examples.ipynb`

Notebook gallery of map and animation utilities powered by `WeatherVisualizer`
and `FlowVisualizer`. Launch it through Jupyter after running
`python setup_notebook_env.py`.

## Tips

- Pass `--help` to any script to view the supported command-line arguments.
- Copy the scripts into your own project and swap in custom datasets or models.
- Pair the scripts with [Advanced Usage](advanced_usage.md) to integrate them
  into automated pipelines or the FastAPI dashboard.
