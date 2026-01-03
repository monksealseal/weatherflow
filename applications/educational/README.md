# Educational Laboratory: WeatherFlow for Teaching

Plug-and-play teaching materials for atmospheric dynamics and weather prediction courses.

## Overview

This educational toolkit provides ready-to-use materials for teaching:
- Atmospheric dynamics and thermodynamics
- Numerical weather prediction
- Flow matching and machine learning for weather
- Data-driven climate science

## Target Audience

- **Graduate students** in atmospheric science, meteorology, or climate science
- **Instructors** teaching atmospheric dynamics or numerical modeling
- **Self-learners** exploring weather prediction and ML
- **Workshop organizers** needing hands-on materials

## Features

- **Zero Setup Required**: Runs in Google Colab or Jupyter with minimal dependencies
- **Interactive Dashboards**: Explore concepts with live visualizations
- **Guided Exercises**: Step-by-step problems with worked solutions
- **Real Data**: Uses actual ERA5 reanalysis data
- **Modular Design**: Mix and match components for your course

## Quick Start

### Run in Google Colab

The fastest way to get started:

1. Open the main educational notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monksealseal/weatherflow/blob/main/applications/educational/notebooks/weatherflow_educational_lab.ipynb)

2. Click "Runtime" → "Run all"

3. Explore the interactive dashboards!

### Run Locally

```bash
# Clone repository
git clone https://github.com/monksealseal/weatherflow.git
cd weatherflow

# Install dependencies
pip install -e .
pip install -r applications/educational/requirements.txt

# Launch Jupyter
jupyter lab applications/educational/notebooks/
```

## Educational Modules

### Module 1: Atmospheric Dynamics Fundamentals
**Notebook:** `atmospheric_dynamics_lab.ipynb`

**Topics:**
- Geostrophic balance and thermal wind
- Rossby waves and dispersion
- Potential vorticity
- Balanced flows

**Interactive Tools:**
- Balanced flow calculator
- Rossby wave dispersion diagram
- PV cross-sections
- Wind-height-temperature relationships

**Learning Objectives:**
- Calculate geostrophic wind from height fields
- Understand Rossby wave propagation
- Visualize PV structures
- Apply thermal wind equation

### Module 2: Introduction to Flow Matching
**Notebook:** `flow_matching_explained.ipynb`

**Topics:**
- What is flow matching?
- Continuous normalizing flows
- ODE solvers for trajectory generation
- Comparison with diffusion models

**Interactive Tools:**
- 2D flow field visualizer
- Trajectory evolution animator
- Loss landscape explorer
- Sampling comparison tool

**Learning Objectives:**
- Understand flow matching vs. diffusion
- Visualize probability flows
- Train a simple flow model
- Generate samples from learned flows

### Module 3: Weather Prediction with WeatherFlow
**Notebook:** `weather_prediction_tutorial.ipynb`

**Topics:**
- Loading and preprocessing ERA5 data
- Training a forecast model
- Generating predictions
- Evaluating forecast skill

**Interactive Tools:**
- Data explorer dashboard
- Training monitor
- Forecast visualization
- Skill score calculator

**Learning Objectives:**
- Work with atmospheric reanalysis data
- Train a neural weather model
- Generate and visualize forecasts
- Compute standard verification metrics

### Module 4: Physics-Informed Neural Networks
**Notebook:** `physics_constraints_lab.ipynb`

**Topics:**
- Mass conservation
- Energy balance
- Vorticity dynamics
- Integrating physics into ML

**Interactive Tools:**
- Conservation law checker
- Physics loss visualizer
- Constraint impact analyzer
- Physical consistency validator

**Learning Objectives:**
- Implement physics constraints
- Visualize conservation properties
- Compare physics-informed vs. unconstrained models
- Understand trade-offs in hybrid approaches

### Module 5: Graduate Problem Set
**Notebook:** `graduate_problem_set.ipynb`

**Content:**
- 10 curated problems with step-by-step solutions
- Problems cover dynamics, thermodynamics, and forecasting
- Automatic grading and hints
- Solutions revealed after attempting

**Example Problems:**
1. Calculate geostrophic wind from a 500 hPa map
2. Diagnose thermal wind from temperature gradient
3. Compute Rossby wave phase speed
4. Evaluate forecast RMSE and bias
5. Implement a simple persistence forecast
6. Calculate anomaly correlation coefficient
7. Diagnose potential vorticity from fields
8. Estimate forecast skill vs. lead time
9. Compare deterministic vs. ensemble forecasts
10. Analyze extreme event detection

## Instructor Resources

### Lecture Slides
**Directory:** `slides/`

- PowerPoint/PDF presentations for each module
- Figures from notebooks included
- Customizable for your course

### Problem Banks
**Directory:** `problem_banks/`

- Additional problems beyond the main notebook
- Multiple difficulty levels
- Solutions manual (instructor-only, on request)

### Grading Rubrics
**Directory:** `rubrics/`

- Suggested grading criteria for exercises
- Automated checking scripts
- Example student submissions

### Course Integration

Example syllabus integration:

**Week 1-2:** Atmospheric Dynamics Fundamentals
- Lecture: Geostrophic balance, Rossby waves
- Lab: Module 1 notebook

**Week 3-4:** Numerical Weather Prediction
- Lecture: NWP basics, data assimilation
- Lab: Module 3 notebook

**Week 5-6:** Machine Learning for Weather
- Lecture: Neural networks, flow matching
- Lab: Module 2 notebook

**Week 7-8:** Physics-Informed ML
- Lecture: Hybrid modeling approaches
- Lab: Module 4 notebook

**Week 9-10:** Final Project
- Students train and evaluate their own models
- Use Module 5 as practice

## Case Studies

### Case Study 1: January 2021 North American Cold Snap
**Notebook:** `case_studies/polar_vortex_2021.ipynb`

Analyze the polar vortex displacement that caused record cold in Texas.

**Topics:**
- Stratospheric warming
- Vortex displacement vs. splitting
- Forecast skill for extreme cold

### Case Study 2: European Heatwave Summer 2019
**Notebook:** `case_studies/european_heatwave_2019.ipynb`

Investigate the record-breaking July 2019 heatwave.

**Topics:**
- Blocking patterns
- Heatwave attribution
- Model performance for extremes

### Case Study 3: Hurricane Model Evaluation
**Notebook:** `case_studies/hurricane_track_evaluation.ipynb`

Evaluate track and intensity forecasts for major hurricanes.

**Topics:**
- Tropical cyclone dynamics
- Track error metrics
- Ensemble forecasting

## Visualization Gallery

Pre-made publication-quality figures:

- Geostrophic wind vectors on pressure maps
- Rossby wave dispersion curves
- PV cross-sections
- Forecast verification diagrams
- Skill score timeseries
- Ensemble spaghetti plots

All figures customizable and exportable for presentations.

## Data

### Included Sample Data
- January 2020 global ERA5 subset (500 MB)
- Pre-computed climatology (50 MB)
- Extreme event catalog (5 MB)

### Download Scripts
```bash
python scripts/download_educational_data.py --period 2015-2020 --variables z t
```

Downloads larger datasets for extended exercises.

## Deployment

### Classroom Deployment

**Option 1: JupyterHub**
Deploy on institutional JupyterHub for seamless student access.

**Option 2: Binder**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/monksealseal/weatherflow/main?filepath=applications/educational/notebooks)

**Option 3: Local Installation**
Students install locally (requires ~5 GB disk space).

### Virtual Classroom

All notebooks work in remote/virtual settings:
- No heavy computation required
- Pre-trained models provided
- Asynchronous-friendly exercises

## Assessment

### Formative Assessment
- Built-in quizzes in notebooks
- Instant feedback on calculations
- Hints available on demand

### Summative Assessment
- Final projects using WeatherFlow
- Example project prompts provided
- Rubrics included

## Customization

Easily adapt materials:

```python
# Change default location
config = {
    'default_lat': 45.0,
    'default_lon': -95.0,
    'default_date': '2020-01-15',
}

# Adjust difficulty
difficulty_level = 'undergraduate'  # or 'graduate' or 'advanced'

# Select topics
topics = ['dynamics', 'forecasting', 'ml']  # omit 'physics_informed' if desired
```

## Accessibility

- Alt text for all figures
- Colorblind-friendly color schemes
- Screen-reader compatible notebooks
- Video captions available

## Support

- **Documentation**: Detailed guides in each notebook
- **FAQ**: Common questions answered in `FAQ.md`
- **Discussion Forum**: GitHub Discussions for Q&A
- **Office Hours**: Monthly virtual office hours (schedule in repo)

## Citation

If you use these materials in your course, please cite:

```bibtex
@software{weatherflow_educational,
  title={WeatherFlow Educational Laboratory},
  author={WeatherFlow Contributors},
  year={2024},
  url={https://github.com/monksealseal/weatherflow/tree/main/applications/educational}
}
```

## License

Educational materials are released under CC BY 4.0 License.
Code is released under MIT License.

## Feedback

We welcome feedback from instructors and students!

- Submit issues: [GitHub Issues](https://github.com/monksealseal/weatherflow/issues)
- Share adaptations: [GitHub Discussions](https://github.com/monksealseal/weatherflow/discussions)
- Email: weatherflow@example.com

## Acknowledgments

Developed with support from:
- National Science Foundation
- Atmospheric Science Education Community
- Student feedback from beta testing at 5 universities

---

**Ready to get started?** Open the [Main Educational Lab Notebook](notebooks/weatherflow_educational_lab.ipynb)!
