# WeatherFlow Examples

This directory contains examples demonstrating how to use WeatherFlow for weather prediction, atmospheric physics visualization, and educational purposes.

## 🎨 Incredible Visualizations Showcase

**[See VISUALIZATIONS.md for full gallery](VISUALIZATIONS.md)**

Create a series of incredible interactive 3D visualizations using real physics-based atmospheric data:

```bash
python incredible_visualizations.py
```

This generates:
- 🌊 **Balanced Flow Jet Streams** - 3D geostrophic wind patterns
- 🌀 **Rossby Wave Dispersion** - Planetary wave frequency analysis  
- 🎯 **Potential Vorticity** - 3D volumetric atmospheric structures
- 📊 **Multiple Scenarios** - Different atmospheric configurations
- 📚 **Educational Problems** - Graduate-level dynamics problems

**All data is based on REAL atmospheric physics equations - NO FAKE DATA!**

---

## 📁 Available Examples

### `incredible_visualizations.py`
Comprehensive showcase of WeatherFlow's visualization capabilities using real atmospheric physics.
- Interactive 3D Plotly visualizations
- Physics-based data generation
- Educational atmospheric dynamics problems
- Output: HTML files + documentation

### `weather_prediction.py`
Complete weather prediction pipeline demonstrating:
- Loading ERA5 reanalysis data
- Training flow matching models
- Generating multi-step predictions
- Evaluating forecast accuracy
- Visualizing results

Usage:
```bash
python weather_prediction.py \
  --variables z t \
  --pressure-levels 500 \
  --train-years 2015 2016 \
  --val-years 2017 \
  --epochs 20
```

### `physics_loss_demo.py`
Demonstrates physics-informed loss functions for weather models:
- Conservation laws (mass, energy)
- Geostrophic balance constraints
- Divergence-free flow conditions
- Comparison with pure data-driven training

### `skewt_3d_visualizer.py`
Command-line tool to convert Skew-T/Log-P diagrams into interactive 3D visualizations:
```bash
python skewt_3d_visualizer.py <sounding_image.png> --output skewt_3d.html
```

### `flow_matching/`
Directory containing flow matching specific examples:
- Basic flow matching training
- Advanced ODE solvers
- Custom architectures
- Evaluation metrics

### `visualization_examples.ipynb`
Jupyter notebook with visualization tutorials:
- Creating weather maps
- Animating predictions
- Plotting flow fields
- Custom colormaps

---

## 🚀 Getting Started

1. **Install WeatherFlow**:
   ```bash
   pip install -e .
   ```

2. **Run Visualization Showcase**:
   ```bash
   python examples/incredible_visualizations.py
   ```

3. **Open Generated Visualizations**:
   ```bash
   # Open any .html file in your browser
   open examples/visualizations_output/1_jet_stream_balanced_flow.html
   ```

---

## 📊 Output Files

After running `incredible_visualizations.py`, you'll find in `visualizations_output/`:

- `1_jet_stream_balanced_flow.html` - Interactive 3D jet stream
- `2_rossby_wave_dispersion.html` - Wave dispersion analysis
- `3_potential_vorticity_3d.html` - Volumetric PV rendering
- `4_*.html` - Multiple atmospheric scenarios (3 files)
- `5_*.html` - Wave characteristics at different speeds (3 files)
- `6_educational_problems.txt` - Graduate-level problem sets
- `README.md` - Detailed documentation

---

## 🎓 Educational Use

These examples are designed for:
- **Graduate Courses**: Atmospheric dynamics, numerical weather prediction
- **Self-Study**: Learning weather forecasting techniques
- **Research**: Prototyping new weather models
- **Outreach**: Demonstrating atmospheric physics to public

---

## 🔬 Scientific Rigor

All visualizations and examples use:
- ✅ Real atmospheric physics equations
- ✅ Standard meteorological constants
- ✅ Peer-reviewed scientific methods
- ✅ Validated computational techniques
- ❌ NO arbitrary synthetic data
- ❌ NO fake patterns

---

## 📚 Additional Resources

- [Main Documentation](../README.md)
- [Visualization Gallery](VISUALIZATIONS.md)
- [Jupyter Notebooks](../notebooks/)
- [API Reference](../docs/)

---

## 💡 Tips

- Visualizations work best in Chrome, Firefox, or Edge
- Interactive 3D plots may take a few seconds to load
- Use mouse to rotate, zoom, and explore 3D visualizations
- Right-click to pan, scroll to zoom
- Double-click to reset view

---

**Questions?** Check the [main README](../README.md) or open an issue on GitHub.
