"""
WeatherBench2 Visualization Utilities

Reusable visualization components for WeatherBench2-style evaluation displays.
Includes scorecards, skill curves, regional analysis plots, and spectral displays.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# Color Schemes
# =============================================================================

def get_improvement_color(improvement: float, threshold: float = 10.0) -> str:
    """Get color based on skill improvement percentage.

    Args:
        improvement: Percentage improvement (positive = better)
        threshold: Threshold for max color intensity

    Returns:
        Hex color string
    """
    if improvement is None:
        return "#cccccc"

    # Color scale from red (worse) to green (better)
    colors = {
        "dark_green": "#1a9850",
        "light_green": "#91cf60",
        "yellow_green": "#d9ef8b",
        "yellow": "#fee08b",
        "orange": "#fc8d59",
        "red": "#d73027",
    }

    if improvement > threshold:
        return colors["dark_green"]
    elif improvement > threshold / 2:
        return colors["light_green"]
    elif improvement > 0:
        return colors["yellow_green"]
    elif improvement > -threshold / 2:
        return colors["yellow"]
    elif improvement > -threshold:
        return colors["orange"]
    else:
        return colors["red"]


def get_model_colors() -> Dict[str, str]:
    """Get consistent color mapping for models."""
    return {
        "GraphCast": "#1f77b4",
        "Pangu-Weather": "#ff7f0e",
        "FourCastNet": "#2ca02c",
        "GenCast": "#d62728",
        "Aurora": "#9467bd",
        "ClimaX": "#8c564b",
        "NeuralGCM": "#e377c2",
        "IFS HRES": "#7f7f7f",
        "IFS ENS": "#bcbd22",
        "Persistence": "#17becf",
        "Climatology": "#aaaaaa",
    }


# =============================================================================
# Scorecard Visualizations
# =============================================================================

def create_scorecard_heatmap(
    data: Dict[str, Dict[str, Dict[int, float]]],
    baseline: str,
    variables: List[str],
    lead_times: List[int],
    title: str = "Model Skill Scorecard",
    metric: str = "rmse",
) -> go.Figure:
    """Create a WeatherBench2-style scorecard heatmap.

    Args:
        data: Nested dict {model: {variable: {lead_time: value}}}
        baseline: Name of baseline model for comparison
        variables: List of variables to show
        lead_times: List of lead times (hours)
        title: Plot title
        metric: Metric name for labeling

    Returns:
        Plotly Figure object
    """
    # Get baseline values
    baseline_data = data.get(baseline, {})

    # Models to compare (excluding baseline)
    models = [m for m in data.keys() if m != baseline]

    # Calculate improvements
    z_values = []
    text_values = []
    hover_text = []

    for model in models:
        model_row_z = []
        model_row_text = []
        model_row_hover = []

        for var in variables:
            for lt in lead_times:
                model_val = data.get(model, {}).get(var, {}).get(lt)
                baseline_val = baseline_data.get(var, {}).get(lt)

                if model_val is not None and baseline_val is not None and baseline_val > 0:
                    # Calculate improvement (positive = better for RMSE/MAE)
                    improvement = (baseline_val - model_val) / baseline_val * 100
                    model_row_z.append(improvement)
                    model_row_text.append(f"{improvement:.1f}%")
                    model_row_hover.append(
                        f"<b>{model}</b><br>"
                        f"{var} @ {lt}h<br>"
                        f"Value: {model_val:.2f}<br>"
                        f"Baseline: {baseline_val:.2f}<br>"
                        f"Improvement: {improvement:.1f}%"
                    )
                else:
                    model_row_z.append(None)
                    model_row_text.append("N/A")
                    model_row_hover.append(f"{model}<br>{var} @ {lt}h<br>No data")

        z_values.append(model_row_z)
        text_values.append(model_row_text)
        hover_text.append(model_row_hover)

    # Create x-axis labels
    x_labels = [f"{var}\n{lt}h" for var in variables for lt in lead_times]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=models,
        text=text_values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertext=hover_text,
        hoverinfo="text",
        colorscale=[
            [0, "#d73027"],      # -20% or worse
            [0.25, "#fc8d59"],   # -10%
            [0.4, "#fee08b"],    # -2%
            [0.5, "#ffffbf"],    # 0%
            [0.6, "#d9ef8b"],    # +2%
            [0.75, "#91cf60"],   # +10%
            [1, "#1a9850"],      # +20% or better
        ],
        zmin=-20,
        zmax=20,
        colorbar=dict(
            title=f"% vs {baseline}",
            ticksuffix="%",
        ),
    ))

    # Add variable group separators
    for i in range(1, len(variables)):
        x_pos = i * len(lead_times) - 0.5
        fig.add_vline(x=x_pos, line_width=2, line_color="white")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Variable / Lead Time",
            tickangle=45,
        ),
        yaxis=dict(title="Model"),
        height=max(300, 100 + len(models) * 40),
    )

    return fig


def create_skill_vs_leadtime(
    data: Dict[str, Dict[int, float]],
    baseline_data: Optional[Dict[int, float]] = None,
    variable: str = "Z500",
    metric: str = "RMSE",
    units: str = "",
) -> go.Figure:
    """Create skill degradation curve vs lead time.

    Args:
        data: Dict {model: {lead_time: value}}
        baseline_data: Optional baseline values for skill score
        variable: Variable name for title
        metric: Metric name
        units: Units string

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    for model, values in data.items():
        lead_times = sorted(values.keys())
        metric_values = [values[lt] for lt in lead_times]

        fig.add_trace(go.Scatter(
            x=lead_times,
            y=metric_values,
            mode='lines+markers',
            name=model,
            line=dict(width=2.5, color=colors.get(model, "#000000")),
            marker=dict(size=8),
        ))

    # Add baseline reference if provided
    if baseline_data:
        lead_times = sorted(baseline_data.keys())
        baseline_values = [baseline_data[lt] for lt in lead_times]
        fig.add_trace(go.Scatter(
            x=lead_times,
            y=baseline_values,
            mode='lines',
            name='Baseline',
            line=dict(width=2, dash='dash', color='gray'),
        ))

    fig.update_layout(
        title=f"{variable} {metric} vs Lead Time",
        xaxis_title="Lead Time (hours)",
        yaxis_title=f"{metric} ({units})" if units else metric,
        height=400,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    return fig


def create_acc_threshold_plot(
    data: Dict[str, Dict[int, float]],
    threshold: float = 0.6,
    variable: str = "Z500",
) -> go.Figure:
    """Create ACC plot with useful skill threshold.

    Args:
        data: Dict {model: {lead_time: acc_value}}
        threshold: ACC threshold for useful skill
        variable: Variable name

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    for model, values in data.items():
        lead_times = sorted(values.keys())
        acc_values = [values[lt] for lt in lead_times]

        fig.add_trace(go.Scatter(
            x=lead_times,
            y=acc_values,
            mode='lines+markers',
            name=model,
            line=dict(width=2.5, color=colors.get(model, "#000000")),
            marker=dict(size=8),
        ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Useful skill (ACC={threshold})",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"{variable} Anomaly Correlation Coefficient",
        xaxis_title="Lead Time (hours)",
        yaxis_title="ACC",
        yaxis_range=[0, 1],
        height=400,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


# =============================================================================
# Regional Analysis
# =============================================================================

def create_regional_comparison_bar(
    data: Dict[str, Dict[str, float]],
    regions: List[str],
    variable: str = "Z500",
    metric: str = "RMSE",
) -> go.Figure:
    """Create regional comparison bar chart.

    Args:
        data: Dict {model: {region: value}}
        regions: List of region names
        variable: Variable name
        metric: Metric name

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    for model, region_values in data.items():
        values = [region_values.get(r, 0) for r in regions]
        fig.add_trace(go.Bar(
            name=model,
            x=regions,
            y=values,
            marker_color=colors.get(model, "#000000"),
        ))

    fig.update_layout(
        title=f"{variable} {metric} by Region",
        xaxis_title="Region",
        yaxis_title=metric,
        barmode='group',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_regional_radar(
    data: Dict[str, Dict[str, float]],
    regions: List[str],
    normalize: bool = True,
) -> go.Figure:
    """Create radar/spider plot for regional skills.

    Args:
        data: Dict {model: {region: skill_value}}
        regions: List of region names
        normalize: Whether to normalize values to 0-1

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    for model, region_values in data.items():
        values = [region_values.get(r, 0) for r in regions]

        if normalize:
            max_val = max(values) if max(values) > 0 else 1
            values = [v / max_val for v in values]

        # Close the polygon
        values = values + [values[0]]
        categories = regions + [regions[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model,
            line=dict(color=colors.get(model, "#000000")),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else None,
            ),
        ),
        showlegend=True,
        title="Regional Skill Comparison",
        height=450,
    )

    return fig


# =============================================================================
# Spectral Analysis
# =============================================================================

def create_power_spectrum(
    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_slopes: Optional[List[float]] = None,
    title: str = "Power Spectrum",
) -> go.Figure:
    """Create power spectrum plot in log-log space.

    Args:
        spectra: Dict {model: (wavenumbers, power)}
        reference_slopes: Optional list of reference slopes to show (e.g., [-3, -5/3])
        title: Plot title

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    for model, (wavenumbers, power) in spectra.items():
        fig.add_trace(go.Scatter(
            x=wavenumbers,
            y=power,
            mode='lines',
            name=model,
            line=dict(width=2, color=colors.get(model, "#000000")),
        ))

    # Add reference slopes
    if reference_slopes:
        k_ref = np.logspace(0, 2, 50)
        for slope in reference_slopes:
            power_ref = k_ref ** slope * 1e6
            slope_label = f"k^{slope}" if slope == int(slope) else f"k^{slope:.2f}"
            fig.add_trace(go.Scatter(
                x=k_ref,
                y=power_ref,
                mode='lines',
                name=slope_label,
                line=dict(width=2, dash='dash', color='gray'),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Zonal Wavenumber",
        yaxis_title="Power Spectral Density",
        xaxis_type="log",
        yaxis_type="log",
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    return fig


# =============================================================================
# Ensemble/Probabilistic Metrics
# =============================================================================

def create_spread_skill_plot(
    spread_data: Dict[str, Dict[int, float]],
    skill_data: Dict[str, Dict[int, float]],
) -> go.Figure:
    """Create spread-skill diagram for ensemble forecasts.

    Args:
        spread_data: Dict {model: {lead_time: spread}}
        skill_data: Dict {model: {lead_time: rmse}}

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Spread vs Lead Time", "Spread-Skill Ratio"],
    )

    for model in spread_data.keys():
        if model not in skill_data:
            continue

        lead_times = sorted(spread_data[model].keys())
        spreads = [spread_data[model][lt] for lt in lead_times]
        skills = [skill_data[model].get(lt, 0) for lt in lead_times]
        ratios = [s / sk if sk > 0 else 0 for s, sk in zip(spreads, skills)]

        color = colors.get(model, "#000000")

        # Spread curve
        fig.add_trace(
            go.Scatter(
                x=lead_times,
                y=spreads,
                mode='lines+markers',
                name=model,
                line=dict(color=color),
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Spread-skill ratio
        fig.add_trace(
            go.Scatter(
                x=lead_times,
                y=ratios,
                mode='lines+markers',
                name=model,
                line=dict(color=color),
                showlegend=False,
            ),
            row=1, col=2,
        )

    # Add ideal ratio line
    fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=400)
    fig.update_xaxes(title_text="Lead Time (hours)")
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Spread/Skill Ratio", row=1, col=2)

    return fig


def create_rank_histogram(
    histogram_data: Dict[str, np.ndarray],
    n_bins: int = 11,
) -> go.Figure:
    """Create rank histogram for ensemble calibration.

    Args:
        histogram_data: Dict {model: histogram_counts}
        n_bins: Number of bins

    Returns:
        Plotly Figure
    """
    colors = get_model_colors()

    fig = go.Figure()

    bin_edges = np.arange(n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for model, counts in histogram_data.items():
        # Normalize
        counts_norm = counts / np.sum(counts)

        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts_norm,
            name=model,
            marker_color=colors.get(model, "#000000"),
            opacity=0.7,
        ))

    # Add flat reference line
    fig.add_hline(
        y=1 / n_bins,
        line_dash="dash",
        line_color="red",
        annotation_text="Ideal (flat)",
    )

    fig.update_layout(
        title="Rank Histogram (Ensemble Calibration)",
        xaxis_title="Rank",
        yaxis_title="Frequency",
        barmode='group',
        height=350,
    )

    return fig


# =============================================================================
# Summary Statistics
# =============================================================================

def create_model_ranking_table(
    data: Dict[str, Dict[str, float]],
    metrics: List[str],
    higher_is_better: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Create model ranking table across metrics.

    Args:
        data: Dict {model: {metric: value}}
        metrics: List of metrics to rank
        higher_is_better: Dict {metric: bool} indicating if higher is better

    Returns:
        DataFrame with rankings
    """
    if higher_is_better is None:
        higher_is_better = {}

    models = list(data.keys())
    rankings = {m: {} for m in models}

    for metric in metrics:
        values = [(m, data[m].get(metric, float('inf'))) for m in models]

        # Sort based on whether higher is better
        ascending = not higher_is_better.get(metric, False)
        values.sort(key=lambda x: x[1], reverse=not ascending)

        for rank, (model, _) in enumerate(values, 1):
            rankings[model][metric] = rank

    # Calculate average rank
    for model in models:
        ranks = list(rankings[model].values())
        rankings[model]["Average Rank"] = np.mean(ranks)

    df = pd.DataFrame(rankings).T
    df = df.sort_values("Average Rank")

    return df


def create_improvement_summary(
    model_values: Dict[str, float],
    baseline_value: float,
    metric_name: str = "RMSE",
) -> Dict[str, Dict[str, Any]]:
    """Calculate improvement summary for multiple models.

    Args:
        model_values: Dict {model: value}
        baseline_value: Baseline value for comparison
        metric_name: Name of the metric

    Returns:
        Dict with improvement statistics
    """
    summary = {}

    for model, value in model_values.items():
        if baseline_value > 0:
            improvement = (baseline_value - value) / baseline_value * 100
        else:
            improvement = 0

        summary[model] = {
            "value": value,
            "improvement": improvement,
            "is_better": improvement > 0,
            "color": get_improvement_color(improvement),
        }

    return summary
