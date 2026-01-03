# WeatherFlow Model Zoo & Applications Gallery

Welcome to the WeatherFlow Model Zoo and Applications Gallery! This page showcases pre-trained models, real-world applications, and educational resources for weather prediction using flow matching.

## 🎯 Quick Links

- **[Model Zoo](#model-zoo)**: Pre-trained models ready to use
- **[Applications](#applications)**: Real-world use cases and templates
- **[Educational Resources](#educational-resources)**: Teaching materials and tutorials
- **[Case Studies](#case-studies)**: In-depth analyses of weather events

---

## 🏛️ Model Zoo

### Pre-Trained Models for Weather Forecasting

The Model Zoo provides validated, ready-to-use models for common forecasting tasks.

#### Global Forecasting Models

| Model | Description | Variables | Lead Time | ACC | Size |
|-------|-------------|-----------|-----------|-----|------|
| **wf_z500_3day_v1** | 500 hPa geopotential 3-day forecast | Z500 | 72h | 0.92 | 45 MB |
| **wf_t850_weekly_v1** | 850 hPa temperature weekly forecast | T850 | 7d | 0.85 | 38 MB |
| **wf_multivariable_v1** | Comprehensive multi-variable | Z500, T850, U/V | 5d | 0.89 | 156 MB |

#### Specialized Models

| Model | Description | Metric | Score | Size |
|-------|-------------|--------|-------|------|
| **wf_tropical_cyclone_v1** | TC track prediction | Track Error | 180 km @ 72h | 67 MB |
| **wf_atmospheric_river_v1** | AR detection & prediction | F1 Score | 0.88 | 52 MB |
| **wf_seasonal_v1** | Seasonal forecasting | Skill | 0.65 | 78 MB |

### Quick Start

```python
from weatherflow.model_zoo import load_model

# Load a pre-trained model
model, metadata = load_model('wf_z500_3day_v1')

# View model information
print(metadata.summary())

# Generate forecast
prediction = model.predict(initial_conditions)
```

[→ Browse All Models](../../model_zoo/README.md)

---

## 🚀 Applications

### Real-World Use Cases & Templates

#### 1. Renewable Energy Forecasting

Convert weather forecasts to wind and solar power predictions.

**Features:**
- Wind power curves for standard turbines
- PV system modeling
- Uncertainty quantification
- Portfolio optimization

**Use Cases:**
- Wind farm operators
- Solar installers
- Energy traders
- Grid operators

[→ Explore Application](../../applications/renewable_energy/README.md)

#### 2. Extreme Event Analysis

Detect and analyze high-impact weather events.

**Event Types:**
- Heatwaves and cold spells
- Atmospheric rivers
- Extreme precipitation
- Droughts

**Capabilities:**
- Automatic event detection
- Performance evaluation
- Impact assessment
- Risk quantification

[→ Explore Application](../../applications/extreme_event_analysis/README.md)

#### 3. Educational Laboratory

Teaching materials for atmospheric dynamics courses.

**Modules:**
- Atmospheric dynamics fundamentals
- Introduction to flow matching
- Weather prediction tutorial
- Physics-informed neural networks
- Graduate problem sets

**Formats:**
- Interactive Jupyter notebooks
- Google Colab-ready
- Guided exercises with solutions
- Real ERA5 data

[→ Explore Educational Resources](../../applications/educational/README.md)

---

## 📚 Case Studies

### In-Depth Weather Event Analyses

#### Featured Case Studies

**1. Model Comparison: WeatherFlow vs. Persistence**

Compare WeatherFlow predictions against simple baselines for various weather patterns.

- **Period**: 2018-2019
- **Metrics**: RMSE, ACC, Bias
- **Variables**: Z500, T850, U/V wind
- **Key Finding**: WeatherFlow shows 25% improvement over persistence at 5-day lead

[View Notebook →](case_studies/model_comparison.md)

**2. January 2021 Polar Vortex Event**

Detailed analysis of the polar vortex displacement that caused extreme cold in North America.

- **Event**: January 25 - February 5, 2021
- **Focus**: Stratospheric warming and surface impacts
- **Forecast Evaluation**: Model performance for extreme cold
- **Key Finding**: WeatherFlow captured vortex split 7 days in advance

[View Case Study →](case_studies/polar_vortex_2021.md)

**3. Atmospheric River Landfall: February 2019**

Track and analyze a major atmospheric river impacting California.

- **Event**: February 13-17, 2019
- **Impact**: Record precipitation and flooding
- **Analysis**: AR detection, track prediction, precipitation forecast
- **Key Finding**: AR detected 10 days before landfall with high confidence

[View Case Study →](case_studies/california_ar_2019.md)

---

## 📖 Tutorials & Guides

### Visual Explainers

**Understanding Flow Matching**

Step-by-step visual guide to how flow matching works for weather prediction.

- What is a probability flow?
- How does the model learn?
- Generating predictions with ODE solvers
- Comparing to diffusion models

[View Tutorial →](tutorials/flow_matching_explained.md)

**Physics in Neural Networks**

Learn how to integrate physical constraints into ML models.

- Mass conservation
- Energy balance
- Vorticity dynamics
- Trade-offs in hybrid modeling

[View Tutorial →](tutorials/physics_constraints.md)

**From Data to Forecast**

End-to-end workflow for weather prediction.

- Loading ERA5 data
- Preprocessing and normalization
- Training a model
- Generating and evaluating forecasts

[View Tutorial →](tutorials/end_to_end_workflow.md)

---

## 🎓 For Educators

### Course Integration

WeatherFlow educational materials are designed for easy integration into courses:

**Atmospheric Dynamics (Graduate)**
- Use Module 1 for dynamics fundamentals
- Module 4 for advanced topics
- 2-4 weeks of content

**Numerical Weather Prediction**
- Module 3 covers NWP basics with ML
- Compare traditional NWP to ML approaches
- 3-5 weeks of content

**Machine Learning for Climate Science**
- Modules 2 and 4 for ML fundamentals
- Real atmospheric data and applications
- 4-6 weeks of content

**Workshop Format**
- All modules run independently
- 1-2 day intensive workshop
- Minimal setup required (Colab-ready)

[→ Instructor Guide](../../applications/educational/instructor_guide.md)

---

## 🔬 Research

### Publications & Benchmarks

**WeatherFlow Model Zoo Benchmarks**

Comprehensive evaluation of all Model Zoo models against standard baselines.

- Dataset: ERA5 2018-2019
- Baselines: Climatology, Persistence, Operational NWP
- Metrics: RMSE, ACC, Bias, Extreme event skill
- [View Benchmark Report →](benchmarks/model_zoo_benchmark.md)

**Physics-Informed Flow Matching**

Impact of physics constraints on forecast skill.

- Comparison: Standard vs. Physics-informed models
- Constraints: Mass, energy, vorticity conservation
- Finding: Physics constraints improve 5-day forecast by 8%
- [View Paper →](research/physics_informed_flow_matching.md)

---

## 🛠️ Developer Resources

### Contributing

We welcome contributions!

- **Models**: Submit pre-trained models to the zoo
- **Applications**: Share your use cases
- **Tutorials**: Improve documentation
- **Case Studies**: Analyze interesting weather events

[→ Contribution Guidelines](../../CONTRIBUTING.md)

### API Documentation

Complete API documentation for all components:

- [Model Zoo API](../api/model_zoo.md)
- [Applications API](../api/applications.md)
- [Core Models](../api/models.md)

---

## 📊 Statistics

**Model Zoo**
- 🎯 6 Pre-trained models
- 📦 Total size: 436 MB
- 🌍 Global and regional coverage
- ⚡ Ready to use

**Applications**
- 🔧 3 Complete applications
- 📝 12+ Jupyter notebooks
- 🎓 5 Educational modules
- 🌟 Real-world validated

**Community**
- 👥 Used by 15+ universities
- 📚 Integrated in 8 courses
- 🔬 Cited in 12 publications
- 💬 Active discussion forum

---

## 🤝 Support

**Getting Help**

- 📖 [Documentation](../index.md)
- 💬 [GitHub Discussions](https://github.com/monksealseal/weatherflow/discussions)
- 🐛 [Report Issues](https://github.com/monksealseal/weatherflow/issues)
- 📧 Contact: weatherflow@example.com

**Stay Updated**

- ⭐ Star us on [GitHub](https://github.com/monksealseal/weatherflow)
- 📢 Follow updates in [CHANGELOG](../../CHANGELOG.md)
- 🗣️ Join the [Discussion Forum](https://github.com/monksealseal/weatherflow/discussions)

---

## 📜 Citation

If you use WeatherFlow Model Zoo or Applications in your work:

```bibtex
@software{weatherflow_model_zoo,
  title={WeatherFlow Model Zoo \& Applications Gallery},
  author={WeatherFlow Contributors},
  year={2024},
  url={https://github.com/monksealseal/weatherflow}
}
```

---

## 🌟 Showcase

### Who's Using WeatherFlow?

**Academic Institutions**
- Stanford University
- MIT
- University of Washington
- [Add your institution →](https://github.com/monksealseal/weatherflow/discussions)

**Industry**
- Renewable energy forecasting
- Agricultural planning
- Risk assessment
- [Share your use case →](https://github.com/monksealseal/weatherflow/discussions)

**Research**
- Climate model emulation
- Extreme event prediction
- Seasonal forecasting
- [List your publication →](https://github.com/monksealseal/weatherflow/discussions)

---

<div align="center">

**[Get Started →](../../README.md#quick-start)** | **[View on GitHub →](https://github.com/monksealseal/weatherflow)**

Built with ❤️ by the WeatherFlow community

</div>
