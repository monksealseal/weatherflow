# Utilities and Visualization API Reference

WeatherFlow bundles a range of utilities for interpreting model outputs,
creating publication-ready figures, and teaching atmospheric dynamics.

## WeatherVisualizer

Map-focused plotting powered by Matplotlib and Cartopy. Supports global fields,
comparison plots, error distributions, animations, and flow vector overlays.

::: weatherflow.utils.visualization.WeatherVisualizer
    :members:

## FlowVisualizer

Simple helper for animating the evolution of a state along the learned flow. It
is useful for debugging training runs without leaving the notebook.

::: weatherflow.utils.flow_visualization.FlowVisualizer
    :members:

## SKEW-T parsing and 3D visualisation

Turn static sounding images into quantitative profiles and interactive Plotly
figures.

::: weatherflow.utils.skewt.SkewTImageParser
    :members:

::: weatherflow.utils.skewt.SkewT3DVisualizer
    :members:

::: weatherflow.utils.skewt.SkewTCalibration
    :members:

::: weatherflow.utils.skewt.RGBThreshold
    :members:

## Educational dashboard

Generate geostrophic balance dashboards, Rossby wave laboratories, and worked
practice problems for classroom use.

::: weatherflow.education.graduate_tool.GraduateAtmosphericDynamicsTool
    :members:

::: weatherflow.education.graduate_tool.ProblemScenario
    :members:

::: weatherflow.education.graduate_tool.SolutionStep
    :members:
