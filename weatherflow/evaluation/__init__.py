"""Evaluation utilities.

This module provides evaluation metrics for weather prediction models,
including WeatherBench2-compatible metrics and regional analysis.
"""

from .gaia.metrics import (
    EvaluationSelectionConfig,
    EvaluationSelectionMetadata,
    ReliabilityResult,
    acc,
    brier_score,
    crps_ensemble,
    mae,
    reliability_curve,
    rmse,
)

# WeatherBench2 compatible metrics
from .weatherbench2 import (
    # Core metrics
    rmse as wb2_rmse,
    mse as wb2_mse,
    mae as wb2_mae,
    bias as wb2_bias,
    acc as wb2_acc,
    wind_vector_rmse,
    seeps,
    # Probabilistic metrics
    crps,
    crps_skill,
    crps_spread,
    energy_score,
    ensemble_spread,
    spread_skill_ratio,
    rank_histogram,
    # Regions
    Region,
    RegionType,
    PREDEFINED_REGIONS,
    get_latitude_weights,
    spatial_average,
    # Derived variables
    compute_wind_speed,
    compute_vorticity,
    compute_divergence,
    compute_eddy_kinetic_energy,
    # Scorecard
    Scorecard,
    ScorecardEntry,
    generate_scorecard,
    # Evaluator
    WeatherBench2Evaluator,
    EvaluationConfig,
    EvaluationResult,
    # Convenience functions
    quick_evaluate,
    compare_models,
    # Constants
    HEADLINE_VARIABLES,
    STANDARD_LEAD_TIMES,
)

__all__ = [
    # GAIA metrics
    "EvaluationSelectionConfig",
    "EvaluationSelectionMetadata",
    "ReliabilityResult",
    "acc",
    "brier_score",
    "crps_ensemble",
    "mae",
    "reliability_curve",
    "rmse",
    # WeatherBench2 metrics
    "wb2_rmse",
    "wb2_mse",
    "wb2_mae",
    "wb2_bias",
    "wb2_acc",
    "wind_vector_rmse",
    "seeps",
    "crps",
    "crps_skill",
    "crps_spread",
    "energy_score",
    "ensemble_spread",
    "spread_skill_ratio",
    "rank_histogram",
    # Regions
    "Region",
    "RegionType",
    "PREDEFINED_REGIONS",
    "get_latitude_weights",
    "spatial_average",
    # Derived variables
    "compute_wind_speed",
    "compute_vorticity",
    "compute_divergence",
    "compute_eddy_kinetic_energy",
    # Scorecard
    "Scorecard",
    "ScorecardEntry",
    "generate_scorecard",
    # Evaluator
    "WeatherBench2Evaluator",
    "EvaluationConfig",
    "EvaluationResult",
    # Convenience functions
    "quick_evaluate",
    "compare_models",
    # Constants
    "HEADLINE_VARIABLES",
    "STANDARD_LEAD_TIMES",
]
