"""
WeatherBench2-Compatible Evaluation Metrics

This module implements evaluation metrics compatible with WeatherBench2,
the benchmark for next-generation data-driven weather models.

Reference: Rasp et al. (2023) "WeatherBench 2: A benchmark for the next
generation of data-driven global weather models"
https://arxiv.org/abs/2308.15560

Metrics implemented:
- Deterministic: RMSE, MSE, MAE, Bias, ACC, Wind Vector RMSE, SEEPS
- Probabilistic: CRPS, Energy Score, Ensemble Spread/Skill
- Regional: Global, Tropics, Extra-tropics, Northern/Southern Hemisphere
- Headline variables: Z500, T850, T2M, WS10, MSLP, Q700, TP24h
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Try to import xarray for better data handling
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# =============================================================================
# Constants and Configuration
# =============================================================================

# Earth radius for spatial calculations (meters)
EARTH_RADIUS = 6.371e6

# WeatherBench2 Headline Variables
HEADLINE_VARIABLES = {
    "z500": {
        "name": "Geopotential at 500hPa",
        "units": "m²/s²",
        "level": 500,
        "description": "Large-scale atmospheric circulation tracer",
    },
    "t850": {
        "name": "Temperature at 850hPa",
        "units": "K",
        "level": 850,
        "description": "Lower troposphere temperature",
    },
    "t2m": {
        "name": "2m Temperature",
        "units": "K",
        "level": "surface",
        "description": "Surface air temperature",
    },
    "ws10": {
        "name": "10m Wind Speed",
        "units": "m/s",
        "level": "surface",
        "description": "Surface wind speed magnitude",
    },
    "mslp": {
        "name": "Mean Sea Level Pressure",
        "units": "Pa",
        "level": "surface",
        "description": "Large-scale dynamics proxy",
    },
    "q700": {
        "name": "Specific Humidity at 700hPa",
        "units": "kg/kg",
        "level": 700,
        "description": "Moisture transport proxy",
    },
    "tp24h": {
        "name": "24h Total Precipitation",
        "units": "mm",
        "level": "surface",
        "description": "Daily precipitation accumulation",
    },
}

# Standard lead times for evaluation (hours)
STANDARD_LEAD_TIMES = [6, 12, 24, 48, 72, 120, 168, 240]


# =============================================================================
# Region Definitions
# =============================================================================

class RegionType(Enum):
    """Predefined region types for evaluation."""
    GLOBAL = "global"
    TROPICS = "tropics"
    EXTRA_TROPICS = "extra_tropics"
    NORTHERN_EXTRA_TROPICS = "northern_extra_tropics"
    SOUTHERN_EXTRA_TROPICS = "southern_extra_tropics"
    NORTHERN_HEMISPHERE = "northern_hemisphere"
    SOUTHERN_HEMISPHERE = "southern_hemisphere"
    LAND = "land"
    OCEAN = "ocean"


@dataclass
class Region:
    """Geographic region for evaluation.

    Attributes:
        name: Human-readable region name
        lat_min: Minimum latitude (-90 to 90)
        lat_max: Maximum latitude (-90 to 90)
        lon_min: Optional minimum longitude (-180 to 180 or 0 to 360)
        lon_max: Optional maximum longitude
        land_mask: Optional boolean array for land/ocean filtering
    """
    name: str
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    land_mask: Optional[NDArray[np.bool_]] = None

    def get_mask(
        self,
        latitudes: NDArray[np.float64],
        longitudes: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.bool_]:
        """Create boolean mask for this region.

        Args:
            latitudes: 1D array of latitudes
            longitudes: Optional 1D array of longitudes

        Returns:
            Boolean mask array (lat, lon) or (lat,) if no longitudes
        """
        lat_mask = (latitudes >= self.lat_min) & (latitudes <= self.lat_max)

        if longitudes is None or (self.lon_min is None and self.lon_max is None):
            if self.land_mask is not None:
                return lat_mask[:, np.newaxis] & self.land_mask
            return lat_mask

        # Handle longitude wrapping
        lon_mask = np.ones(len(longitudes), dtype=bool)
        if self.lon_min is not None and self.lon_max is not None:
            if self.lon_min <= self.lon_max:
                lon_mask = (longitudes >= self.lon_min) & (longitudes <= self.lon_max)
            else:
                # Wrapping case (e.g., 170 to -170)
                lon_mask = (longitudes >= self.lon_min) | (longitudes <= self.lon_max)

        # Combine masks
        mask = lat_mask[:, np.newaxis] & lon_mask[np.newaxis, :]

        if self.land_mask is not None:
            mask = mask & self.land_mask

        return mask


# Predefined regions matching WeatherBench2
PREDEFINED_REGIONS = {
    RegionType.GLOBAL: Region("Global", -90, 90),
    RegionType.TROPICS: Region("Tropics", -20, 20),
    RegionType.EXTRA_TROPICS: Region("Extra-tropics", -90, 90),  # Will exclude tropics
    RegionType.NORTHERN_EXTRA_TROPICS: Region("Northern Extra-tropics", 20, 90),
    RegionType.SOUTHERN_EXTRA_TROPICS: Region("Southern Extra-tropics", -90, -20),
    RegionType.NORTHERN_HEMISPHERE: Region("Northern Hemisphere", 0, 90),
    RegionType.SOUTHERN_HEMISPHERE: Region("Southern Hemisphere", -90, 0),
}


def get_extra_tropical_mask(
    latitudes: NDArray[np.float64],
    threshold: float = 20.0,
) -> NDArray[np.bool_]:
    """Get mask for extra-tropical regions (|lat| > threshold)."""
    return np.abs(latitudes) > threshold


# =============================================================================
# Spatial Weighting
# =============================================================================

def get_latitude_weights(
    latitudes: NDArray[np.float64],
    method: str = "cos",
) -> NDArray[np.float64]:
    """Compute latitude-based area weights.

    WeatherBench2 uses cosine weighting to account for the fact that
    grid cells near the equator represent larger areas than those near poles.

    Args:
        latitudes: Array of latitudes in degrees
        method: Weighting method - "cos" (default) or "area"

    Returns:
        Weight array normalized to sum to 1
    """
    lat_rad = np.deg2rad(latitudes)

    if method == "cos":
        weights = np.cos(lat_rad)
    elif method == "area":
        # More accurate area weighting using sin differences
        lat_bounds = np.zeros(len(latitudes) + 1)
        lat_bounds[1:-1] = (latitudes[:-1] + latitudes[1:]) / 2
        lat_bounds[0] = max(-90, latitudes[0] - (latitudes[1] - latitudes[0]) / 2)
        lat_bounds[-1] = min(90, latitudes[-1] + (latitudes[-1] - latitudes[-2]) / 2)
        lat_bounds_rad = np.deg2rad(lat_bounds)
        weights = np.abs(np.sin(lat_bounds_rad[1:]) - np.sin(lat_bounds_rad[:-1]))
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Normalize
    weights = weights / np.sum(weights)
    return weights


def spatial_average(
    data: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    lat_axis: int = -2,
    lon_axis: int = -1,
    region_mask: Optional[NDArray[np.bool_]] = None,
    weights: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Compute latitude-weighted spatial average.

    Args:
        data: Input array with lat/lon dimensions
        latitudes: 1D array of latitudes
        lat_axis: Axis index for latitude dimension
        lon_axis: Axis index for longitude dimension
        region_mask: Optional boolean mask (lat, lon)
        weights: Optional pre-computed latitude weights

    Returns:
        Array with spatial dimensions removed
    """
    if weights is None:
        weights = get_latitude_weights(latitudes)

    # Expand weights to broadcast with data
    weight_shape = [1] * data.ndim
    weight_shape[lat_axis] = len(latitudes)
    lat_weights = weights.reshape(weight_shape)

    if region_mask is not None:
        # Apply mask
        mask_shape = [1] * data.ndim
        mask_shape[lat_axis] = region_mask.shape[0]
        mask_shape[lon_axis] = region_mask.shape[1]
        mask = region_mask.reshape(mask_shape)

        masked_data = np.where(mask, data, np.nan)
        weighted = masked_data * lat_weights

        # Average over valid points
        sum_weights = np.sum(np.where(mask, lat_weights, 0), axis=(lat_axis, lon_axis))
        result = np.nansum(weighted, axis=(lat_axis, lon_axis)) / sum_weights
    else:
        weighted = data * lat_weights
        result = np.mean(weighted, axis=(lat_axis, lon_axis)) * len(latitudes)

    return result


# =============================================================================
# Deterministic Metrics
# =============================================================================

def rmse(
    forecast: ArrayLike,
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
    spatial_dims: bool = True,
) -> float:
    """Root Mean Square Error with optional latitude weighting.

    Args:
        forecast: Predicted values
        truth: Ground truth values
        latitudes: Latitudes for spatial weighting
        lat_axis: Latitude axis index
        lon_axis: Longitude axis index
        region: Optional region for masked evaluation
        spatial_dims: Whether data has spatial dimensions

    Returns:
        RMSE value
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    squared_error = (forecast - truth) ** 2

    if spatial_dims and latitudes is not None:
        region_mask = region.get_mask(latitudes) if region else None
        mse = spatial_average(squared_error, latitudes, lat_axis, lon_axis, region_mask)
        return float(np.sqrt(np.mean(mse)))
    else:
        return float(np.sqrt(np.mean(squared_error)))


def mse(
    forecast: ArrayLike,
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
    spatial_dims: bool = True,
) -> float:
    """Mean Square Error with optional latitude weighting."""
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    squared_error = (forecast - truth) ** 2

    if spatial_dims and latitudes is not None:
        region_mask = region.get_mask(latitudes) if region else None
        return float(np.mean(spatial_average(
            squared_error, latitudes, lat_axis, lon_axis, region_mask
        )))
    else:
        return float(np.mean(squared_error))


def mae(
    forecast: ArrayLike,
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
    spatial_dims: bool = True,
) -> float:
    """Mean Absolute Error with optional latitude weighting."""
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    abs_error = np.abs(forecast - truth)

    if spatial_dims and latitudes is not None:
        region_mask = region.get_mask(latitudes) if region else None
        return float(np.mean(spatial_average(
            abs_error, latitudes, lat_axis, lon_axis, region_mask
        )))
    else:
        return float(np.mean(abs_error))


def bias(
    forecast: ArrayLike,
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
    spatial_dims: bool = True,
) -> float:
    """Mean Bias (forecast - truth) with optional latitude weighting."""
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    error = forecast - truth

    if spatial_dims and latitudes is not None:
        region_mask = region.get_mask(latitudes) if region else None
        return float(np.mean(spatial_average(
            error, latitudes, lat_axis, lon_axis, region_mask
        )))
    else:
        return float(np.mean(error))


def acc(
    forecast: ArrayLike,
    truth: ArrayLike,
    climatology: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
    epsilon: float = 1e-10,
) -> float:
    """Anomaly Correlation Coefficient.

    ACC measures the correlation between forecast and truth anomalies
    relative to climatology. Values range from -1 to 1, with 1 being perfect.

    ACC = sum(forecast_anom * truth_anom) / sqrt(sum(forecast_anom^2) * sum(truth_anom^2))

    Args:
        forecast: Predicted values
        truth: Ground truth values
        climatology: Climatological mean values
        latitudes: Latitudes for spatial weighting
        lat_axis: Latitude axis index
        lon_axis: Longitude axis index
        region: Optional region for masked evaluation
        epsilon: Small value to prevent division by zero

    Returns:
        ACC value between -1 and 1
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    climatology = np.asarray(climatology, dtype=np.float64)

    # Compute anomalies
    forecast_anom = forecast - climatology
    truth_anom = truth - climatology

    if latitudes is not None:
        weights = get_latitude_weights(latitudes)
        region_mask = region.get_mask(latitudes) if region else None

        # Weighted correlation
        numerator = spatial_average(
            forecast_anom * truth_anom, latitudes, lat_axis, lon_axis, region_mask
        )
        forecast_var = spatial_average(
            forecast_anom ** 2, latitudes, lat_axis, lon_axis, region_mask
        )
        truth_var = spatial_average(
            truth_anom ** 2, latitudes, lat_axis, lon_axis, region_mask
        )

        denominator = np.sqrt(forecast_var * truth_var) + epsilon
        return float(np.mean(numerator / denominator))
    else:
        numerator = np.sum(forecast_anom * truth_anom)
        denominator = np.sqrt(
            np.sum(forecast_anom ** 2) * np.sum(truth_anom ** 2)
        ) + epsilon
        return float(numerator / denominator)


def wind_vector_rmse(
    u_forecast: ArrayLike,
    v_forecast: ArrayLike,
    u_truth: ArrayLike,
    v_truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    lat_axis: int = -2,
    lon_axis: int = -1,
    region: Optional[Region] = None,
) -> float:
    """Wind Vector RMSE.

    Combines u and v wind component errors as sqrt(u_err^2 + v_err^2),
    which is equivalent to the magnitude of the vector error.

    Args:
        u_forecast: Predicted u-component (east-west)
        v_forecast: Predicted v-component (north-south)
        u_truth: True u-component
        v_truth: True v-component
        latitudes: Latitudes for spatial weighting
        lat_axis: Latitude axis index
        lon_axis: Longitude axis index
        region: Optional region for masked evaluation

    Returns:
        Wind vector RMSE in m/s
    """
    u_forecast = np.asarray(u_forecast, dtype=np.float64)
    v_forecast = np.asarray(v_forecast, dtype=np.float64)
    u_truth = np.asarray(u_truth, dtype=np.float64)
    v_truth = np.asarray(v_truth, dtype=np.float64)

    # Vector squared error
    vector_se = (u_forecast - u_truth) ** 2 + (v_forecast - v_truth) ** 2

    if latitudes is not None:
        region_mask = region.get_mask(latitudes) if region else None
        mse_val = spatial_average(
            vector_se, latitudes, lat_axis, lon_axis, region_mask
        )
        return float(np.sqrt(np.mean(mse_val)))
    else:
        return float(np.sqrt(np.mean(vector_se)))


# =============================================================================
# Precipitation Metrics
# =============================================================================

def seeps(
    forecast: ArrayLike,
    truth: ArrayLike,
    dry_threshold: float = 0.25,
    light_threshold_percentile: float = 67,
    climatology_dry_fraction: Optional[float] = None,
    climatology_light_threshold: Optional[float] = None,
) -> float:
    """Stable Equitable Error in Probability Space (SEEPS).

    SEEPS is the standard metric for precipitation in WeatherBench2,
    based on Rodwell et al. (2010). It categorizes precipitation into
    dry/light/heavy and computes a scoring matrix.

    Args:
        forecast: Predicted precipitation values
        truth: Ground truth precipitation values
        dry_threshold: Threshold for dry conditions (mm)
        light_threshold_percentile: Percentile for light/heavy boundary
        climatology_dry_fraction: Pre-computed climatological dry fraction
        climatology_light_threshold: Pre-computed light precipitation threshold

    Returns:
        SEEPS score (lower is better, 0 is perfect)
    """
    forecast = np.asarray(forecast, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()

    # Compute climatological thresholds if not provided
    if climatology_dry_fraction is None:
        climatology_dry_fraction = np.mean(truth <= dry_threshold)

    if climatology_light_threshold is None:
        wet_values = truth[truth > dry_threshold]
        if len(wet_values) > 0:
            climatology_light_threshold = np.percentile(wet_values, light_threshold_percentile)
        else:
            climatology_light_threshold = dry_threshold * 2

    # Categorize forecast and truth
    def categorize(values):
        cats = np.zeros_like(values, dtype=int)
        cats[values <= dry_threshold] = 0  # Dry
        cats[(values > dry_threshold) & (values <= climatology_light_threshold)] = 1  # Light
        cats[values > climatology_light_threshold] = 2  # Heavy
        return cats

    forecast_cat = categorize(forecast)
    truth_cat = categorize(truth)

    # SEEPS scoring matrix
    # Rows: forecast category, Columns: truth category
    p1 = climatology_dry_fraction
    p3 = 1 - climatology_dry_fraction - (1 - climatology_dry_fraction) * (light_threshold_percentile / 100)

    # Scoring matrix (penalizes misclassifications)
    scoring_matrix = np.array([
        [0, 1 / (1 - p1), 4 / (1 - p1)],  # Forecast dry
        [1 / p1, 0, 1 / (1 - p1)],         # Forecast light
        [4 / p1, 1 / p1, 0],               # Forecast heavy
    ])

    # Compute SEEPS
    scores = scoring_matrix[forecast_cat, truth_cat]
    return float(np.mean(scores))


# =============================================================================
# Probabilistic Metrics
# =============================================================================

def crps(
    ensemble: ArrayLike,
    truth: ArrayLike,
    ensemble_axis: int = 0,
) -> float:
    """Continuous Ranked Probability Score.

    CRPS measures the integrated squared difference between the forecast
    cumulative distribution and the observation. Lower is better.

    CRPS = E|X - Y| - 0.5 * E|X - X'|

    where X, X' are ensemble members and Y is the observation.

    Args:
        ensemble: Ensemble forecast with shape (n_members, ...)
        truth: Ground truth values
        ensemble_axis: Axis containing ensemble members

    Returns:
        CRPS value (lower is better)
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    # Move ensemble axis to front
    ensemble = np.moveaxis(ensemble, ensemble_axis, 0)
    n_members = ensemble.shape[0]

    # Term 1: Mean absolute error between ensemble and truth
    mae_term = np.mean(np.abs(ensemble - truth), axis=0)

    # Term 2: Mean pairwise absolute difference within ensemble
    # Use efficient ranking-based computation
    ensemble_sorted = np.sort(ensemble, axis=0)
    weights = 2 * np.arange(1, n_members + 1) - n_members - 1
    spread_term = np.sum(weights[:, np.newaxis] * ensemble_sorted, axis=0) / (n_members * (n_members - 1))

    crps_values = mae_term - np.abs(spread_term)
    return float(np.mean(crps_values))


def crps_skill(
    ensemble: ArrayLike,
    truth: ArrayLike,
    ensemble_axis: int = 0,
) -> float:
    """CRPS Skill component - ensemble-observation distance."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    ensemble = np.moveaxis(ensemble, ensemble_axis, 0)
    return float(np.mean(np.abs(ensemble - truth)))


def crps_spread(
    ensemble: ArrayLike,
    ensemble_axis: int = 0,
) -> float:
    """CRPS Spread component - inter-ensemble distances."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    ensemble = np.moveaxis(ensemble, ensemble_axis, 0)
    n_members = ensemble.shape[0]

    ensemble_sorted = np.sort(ensemble, axis=0)
    weights = 2 * np.arange(1, n_members + 1) - n_members - 1
    spread = np.sum(weights[:, np.newaxis] * ensemble_sorted, axis=0) / (n_members * (n_members - 1))

    return float(np.mean(np.abs(spread)))


def energy_score(
    ensemble: ArrayLike,
    truth: ArrayLike,
    ensemble_axis: int = 0,
    variable_axis: int = 1,
) -> float:
    """Energy Score for multivariate ensemble forecasts.

    Extends CRPS to multivariate settings using L2 norms.

    ES = E||X - Y||₂ - 0.5 * E||X - X'||₂

    Args:
        ensemble: Ensemble forecast (n_members, n_variables, ...)
        truth: Ground truth (n_variables, ...)
        ensemble_axis: Axis containing ensemble members
        variable_axis: Axis containing variables

    Returns:
        Energy Score (lower is better)
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    ensemble = np.moveaxis(ensemble, ensemble_axis, 0)
    n_members = ensemble.shape[0]

    # L2 norm of error vectors
    error_norms = np.sqrt(np.sum((ensemble - truth) ** 2, axis=variable_axis))
    mae_term = np.mean(error_norms, axis=0)

    # Pairwise L2 norms within ensemble
    pairwise_sum = 0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            diff = ensemble[i] - ensemble[j]
            pairwise_sum += np.sqrt(np.sum(diff ** 2, axis=variable_axis - 1))

    spread_term = 2 * pairwise_sum / (n_members * (n_members - 1))

    return float(np.mean(mae_term - 0.5 * spread_term))


def ensemble_spread(
    ensemble: ArrayLike,
    ensemble_axis: int = 0,
) -> float:
    """Ensemble spread (standard deviation across members)."""
    ensemble = np.asarray(ensemble, dtype=np.float64)
    return float(np.mean(np.std(ensemble, axis=ensemble_axis, ddof=1)))


def spread_skill_ratio(
    ensemble: ArrayLike,
    truth: ArrayLike,
    ensemble_axis: int = 0,
) -> float:
    """Spread-Skill Ratio.

    Ratio of ensemble spread to RMSE of ensemble mean.
    Ideally should be close to 1 for well-calibrated ensembles.

    Args:
        ensemble: Ensemble forecast
        truth: Ground truth
        ensemble_axis: Axis containing ensemble members

    Returns:
        Spread-skill ratio (ideally ~1)
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    # Spread
    spread = ensemble_spread(ensemble, ensemble_axis)

    # Skill (RMSE of ensemble mean)
    ensemble_mean = np.mean(ensemble, axis=ensemble_axis)
    skill = float(np.sqrt(np.mean((ensemble_mean - truth) ** 2)))

    if skill < 1e-10:
        return float('inf')

    return spread / skill


def rank_histogram(
    ensemble: ArrayLike,
    truth: ArrayLike,
    ensemble_axis: int = 0,
    n_bins: Optional[int] = None,
) -> NDArray[np.float64]:
    """Compute rank histogram for ensemble calibration assessment.

    For a well-calibrated ensemble, the histogram should be flat.
    U-shaped indicates under-dispersion, dome-shaped indicates over-dispersion.

    Args:
        ensemble: Ensemble forecast (n_members, ...)
        truth: Ground truth
        ensemble_axis: Axis containing ensemble members
        n_bins: Number of histogram bins (default: n_members + 1)

    Returns:
        Normalized histogram counts
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    ensemble = np.moveaxis(ensemble, ensemble_axis, 0)
    n_members = ensemble.shape[0]

    if n_bins is None:
        n_bins = n_members + 1

    # Flatten spatial dimensions
    ensemble_flat = ensemble.reshape(n_members, -1)
    truth_flat = truth.ravel()

    # Compute ranks
    combined = np.vstack([truth_flat[np.newaxis, :], ensemble_flat])
    ranks = np.argsort(np.argsort(combined, axis=0), axis=0)[0]  # Rank of truth

    # Add small random perturbation for ties
    ranks = ranks + np.random.uniform(-0.1, 0.1, ranks.shape)

    # Compute histogram
    hist, _ = np.histogram(ranks, bins=n_bins, range=(0, n_members + 1))

    return hist / np.sum(hist)


# =============================================================================
# Derived Variables
# =============================================================================

def compute_wind_speed(
    u: ArrayLike,
    v: ArrayLike,
) -> NDArray[np.float64]:
    """Compute wind speed magnitude from u and v components."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.sqrt(u ** 2 + v ** 2)


def compute_geostrophic_wind(
    geopotential: ArrayLike,
    latitudes: NDArray[np.float64],
    dx: float,
    dy: float,
    f0: float = 1e-4,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute geostrophic wind from geopotential.

    u_g = -(1/f) * dΦ/dy
    v_g = (1/f) * dΦ/dx

    Args:
        geopotential: Geopotential field (lat, lon)
        latitudes: Latitude values
        dx: Grid spacing in x (meters)
        dy: Grid spacing in y (meters)
        f0: Coriolis parameter (default mid-latitude value)

    Returns:
        Tuple of (u_g, v_g) geostrophic wind components
    """
    phi = np.asarray(geopotential, dtype=np.float64)

    # Coriolis parameter varying with latitude
    omega = 7.292e-5  # Earth's rotation rate
    f = 2 * omega * np.sin(np.deg2rad(latitudes))[:, np.newaxis]
    f = np.where(np.abs(f) < 1e-10, np.sign(f) * 1e-10, f)  # Avoid division by zero

    # Gradients
    dphi_dy = np.gradient(phi, dy, axis=0)
    dphi_dx = np.gradient(phi, dx, axis=1)

    u_g = -dphi_dy / f
    v_g = dphi_dx / f

    return u_g, v_g


def compute_vorticity(
    u: ArrayLike,
    v: ArrayLike,
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Compute relative vorticity from wind components.

    ζ = dv/dx - du/dy
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    dv_dx = np.gradient(v, dx, axis=-1)
    du_dy = np.gradient(u, dy, axis=-2)

    return dv_dx - du_dy


def compute_divergence(
    u: ArrayLike,
    v: ArrayLike,
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Compute horizontal divergence from wind components.

    D = du/dx + dv/dy
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    du_dx = np.gradient(u, dx, axis=-1)
    dv_dy = np.gradient(v, dy, axis=-2)

    return du_dx + dv_dy


def compute_eddy_kinetic_energy(
    u: ArrayLike,
    v: ArrayLike,
    u_mean: Optional[ArrayLike] = None,
    v_mean: Optional[ArrayLike] = None,
) -> NDArray[np.float64]:
    """Compute eddy kinetic energy.

    EKE = 0.5 * (u'^2 + v'^2)

    where u' and v' are deviations from the mean.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u_mean is None:
        u_mean = np.mean(u, axis=-1, keepdims=True)
    if v_mean is None:
        v_mean = np.mean(v, axis=-1, keepdims=True)

    u_prime = u - u_mean
    v_prime = v - v_mean

    return 0.5 * (u_prime ** 2 + v_prime ** 2)


# =============================================================================
# Spectral Analysis
# =============================================================================

def zonal_power_spectrum(
    data: ArrayLike,
    lat_axis: int = -2,
    lon_axis: int = -1,
) -> NDArray[np.float64]:
    """Compute zonal (east-west) power spectrum.

    Args:
        data: Input field with longitude dimension
        lat_axis: Latitude axis index
        lon_axis: Longitude axis index

    Returns:
        Power spectrum as function of zonal wavenumber
    """
    data = np.asarray(data, dtype=np.float64)

    # FFT along longitude
    fft_result = np.fft.fft(data, axis=lon_axis)
    power = np.abs(fft_result) ** 2

    # Average over latitudes
    power_avg = np.mean(power, axis=lat_axis)

    # Return one-sided spectrum (positive wavenumbers only)
    n_lon = data.shape[lon_axis]
    return power_avg[..., :n_lon // 2 + 1]


def spectral_slope(
    spectrum: ArrayLike,
    wavenumber_range: Optional[Tuple[int, int]] = None,
) -> float:
    """Compute spectral slope in log-log space.

    For turbulent cascades, expected slopes are:
    - k^-3 for enstrophy cascade
    - k^-5/3 for energy cascade

    Args:
        spectrum: Power spectrum values
        wavenumber_range: Optional (min_k, max_k) for fitting

    Returns:
        Spectral slope (negative for typical atmospheric spectra)
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)

    wavenumbers = np.arange(1, len(spectrum))
    log_k = np.log(wavenumbers)
    log_power = np.log(spectrum[1:] + 1e-20)  # Avoid log(0)

    if wavenumber_range is not None:
        k_min, k_max = wavenumber_range
        mask = (wavenumbers >= k_min) & (wavenumbers <= k_max)
        log_k = log_k[mask]
        log_power = log_power[mask]

    # Linear regression in log-log space
    slope, _ = np.polyfit(log_k, log_power, 1)

    return float(slope)


# =============================================================================
# Scorecard Generation
# =============================================================================

@dataclass
class ScorecardEntry:
    """Single entry in a scorecard."""
    variable: str
    lead_time: int
    metric_value: float
    baseline_value: float
    percent_improvement: float
    region: str = "global"

    @property
    def is_better(self) -> bool:
        """Whether the model is better than baseline."""
        return self.percent_improvement > 0


@dataclass
class Scorecard:
    """WeatherBench2-style scorecard for model evaluation."""
    model_name: str
    baseline_name: str
    entries: List[ScorecardEntry] = field(default_factory=list)

    def add_entry(
        self,
        variable: str,
        lead_time: int,
        metric_value: float,
        baseline_value: float,
        region: str = "global",
    ) -> None:
        """Add a scorecard entry."""
        if baseline_value > 0:
            # For RMSE/MAE, negative improvement is better
            percent_improvement = (baseline_value - metric_value) / baseline_value * 100
        else:
            percent_improvement = 0.0

        entry = ScorecardEntry(
            variable=variable,
            lead_time=lead_time,
            metric_value=metric_value,
            baseline_value=baseline_value,
            percent_improvement=percent_improvement,
            region=region,
        )
        self.entries.append(entry)

    def get_entries_by_variable(self, variable: str) -> List[ScorecardEntry]:
        """Get all entries for a specific variable."""
        return [e for e in self.entries if e.variable == variable]

    def get_entries_by_lead_time(self, lead_time: int) -> List[ScorecardEntry]:
        """Get all entries for a specific lead time."""
        return [e for e in self.entries if e.lead_time == lead_time]

    def summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics."""
        improvements = [e.percent_improvement for e in self.entries]
        return {
            "mean_improvement": float(np.mean(improvements)),
            "median_improvement": float(np.median(improvements)),
            "best_improvement": float(np.max(improvements)),
            "worst_improvement": float(np.min(improvements)),
            "percent_better": float(np.mean([1 if e.is_better else 0 for e in self.entries])) * 100,
        }

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame([
                {
                    "variable": e.variable,
                    "lead_time": e.lead_time,
                    "metric_value": e.metric_value,
                    "baseline_value": e.baseline_value,
                    "percent_improvement": e.percent_improvement,
                    "region": e.region,
                }
                for e in self.entries
            ])
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")


def generate_scorecard(
    model_forecasts: Dict[str, Dict[int, ArrayLike]],
    baseline_forecasts: Dict[str, Dict[int, ArrayLike]],
    truth: Dict[str, ArrayLike],
    climatology: Optional[Dict[str, ArrayLike]] = None,
    latitudes: Optional[NDArray[np.float64]] = None,
    model_name: str = "Model",
    baseline_name: str = "IFS HRES",
    metrics: List[str] = ["rmse"],
    regions: List[RegionType] = [RegionType.GLOBAL],
) -> Scorecard:
    """Generate a WeatherBench2-style scorecard.

    Args:
        model_forecasts: Dict of {variable: {lead_time: forecast_array}}
        baseline_forecasts: Dict of {variable: {lead_time: forecast_array}}
        truth: Dict of {variable: truth_array}
        climatology: Optional dict of {variable: climatology_array} for ACC
        latitudes: Latitude array for spatial weighting
        model_name: Name of the model being evaluated
        baseline_name: Name of the baseline model
        metrics: List of metrics to compute ("rmse", "mae", "acc")
        regions: List of regions to evaluate

    Returns:
        Scorecard object with all entries
    """
    scorecard = Scorecard(model_name=model_name, baseline_name=baseline_name)

    for variable in model_forecasts.keys():
        if variable not in truth:
            continue

        truth_var = truth[variable]

        for lead_time in model_forecasts[variable].keys():
            model_fc = model_forecasts[variable][lead_time]
            baseline_fc = baseline_forecasts.get(variable, {}).get(lead_time)

            if baseline_fc is None:
                continue

            for region_type in regions:
                region = PREDEFINED_REGIONS.get(region_type)
                region_name = region_type.value if region_type else "global"

                for metric in metrics:
                    if metric == "rmse":
                        model_val = rmse(model_fc, truth_var, latitudes, region=region)
                        baseline_val = rmse(baseline_fc, truth_var, latitudes, region=region)
                    elif metric == "mae":
                        model_val = mae(model_fc, truth_var, latitudes, region=region)
                        baseline_val = mae(baseline_fc, truth_var, latitudes, region=region)
                    elif metric == "acc" and climatology and variable in climatology:
                        model_val = acc(model_fc, truth_var, climatology[variable], latitudes, region=region)
                        baseline_val = acc(baseline_fc, truth_var, climatology[variable], latitudes, region=region)
                        # For ACC, higher is better, so flip the improvement calculation
                        scorecard.add_entry(
                            variable=f"{variable}_{metric}",
                            lead_time=lead_time,
                            metric_value=model_val,
                            baseline_value=baseline_val,
                            region=region_name,
                        )
                        # Override percent improvement for ACC (higher is better)
                        scorecard.entries[-1] = ScorecardEntry(
                            variable=f"{variable}_{metric}",
                            lead_time=lead_time,
                            metric_value=model_val,
                            baseline_value=baseline_val,
                            percent_improvement=(model_val - baseline_val) / max(baseline_val, 0.01) * 100,
                            region=region_name,
                        )
                        continue
                    else:
                        continue

                    scorecard.add_entry(
                        variable=f"{variable}_{metric}",
                        lead_time=lead_time,
                        metric_value=model_val,
                        baseline_value=baseline_val,
                        region=region_name,
                    )

    return scorecard


# =============================================================================
# Evaluation Runner
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for WeatherBench2-style evaluation."""
    variables: List[str] = field(default_factory=lambda: ["z500", "t850", "t2m"])
    lead_times: List[int] = field(default_factory=lambda: [24, 72, 120, 240])
    metrics: List[str] = field(default_factory=lambda: ["rmse", "mae", "acc"])
    regions: List[RegionType] = field(default_factory=lambda: [RegionType.GLOBAL])
    compute_spectral: bool = False
    compute_ensemble_metrics: bool = False


@dataclass
class EvaluationResult:
    """Results from evaluation."""
    config: EvaluationConfig
    metrics: Dict[str, Dict[str, Dict[int, float]]]  # {region: {variable_metric: {lead_time: value}}}
    scorecard: Optional[Scorecard] = None
    spectral_analysis: Optional[Dict[str, Any]] = None

    def get_metric(
        self,
        variable: str,
        metric: str,
        lead_time: int,
        region: str = "global",
    ) -> Optional[float]:
        """Get a specific metric value."""
        key = f"{variable}_{metric}"
        return self.metrics.get(region, {}).get(key, {}).get(lead_time)

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = ["WeatherBench2 Evaluation Results", "=" * 40]

        for region, region_metrics in self.metrics.items():
            lines.append(f"\nRegion: {region}")
            lines.append("-" * 20)

            for var_metric, lead_values in region_metrics.items():
                lines.append(f"  {var_metric}:")
                for lt, val in sorted(lead_values.items()):
                    lines.append(f"    {lt}h: {val:.4f}")

        if self.scorecard:
            stats = self.scorecard.summary_stats()
            lines.append(f"\nScorecard Summary:")
            lines.append(f"  Mean improvement: {stats['mean_improvement']:.1f}%")
            lines.append(f"  Percent better: {stats['percent_better']:.1f}%")

        return "\n".join(lines)


class WeatherBench2Evaluator:
    """Main evaluator class for WeatherBench2-compatible evaluation."""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        latitudes: Optional[NDArray[np.float64]] = None,
        longitudes: Optional[NDArray[np.float64]] = None,
    ):
        self.config = config or EvaluationConfig()
        self.latitudes = latitudes
        self.longitudes = longitudes
        self._lat_weights = None

        if latitudes is not None:
            self._lat_weights = get_latitude_weights(latitudes)

    def evaluate(
        self,
        forecasts: Dict[str, Dict[int, ArrayLike]],
        truth: Dict[str, ArrayLike],
        climatology: Optional[Dict[str, ArrayLike]] = None,
        baseline_forecasts: Optional[Dict[str, Dict[int, ArrayLike]]] = None,
        model_name: str = "Model",
        baseline_name: str = "IFS HRES",
    ) -> EvaluationResult:
        """Run full evaluation.

        Args:
            forecasts: Model forecasts {variable: {lead_time: array}}
            truth: Ground truth {variable: array}
            climatology: Climatological means {variable: array}
            baseline_forecasts: Optional baseline for scorecard
            model_name: Name of model being evaluated
            baseline_name: Name of baseline model

        Returns:
            EvaluationResult with all computed metrics
        """
        results: Dict[str, Dict[str, Dict[int, float]]] = {}

        for region_type in self.config.regions:
            region = PREDEFINED_REGIONS.get(region_type)
            region_name = region_type.value
            results[region_name] = {}

            for variable in self.config.variables:
                if variable not in forecasts or variable not in truth:
                    continue

                truth_var = np.asarray(truth[variable])

                for metric_name in self.config.metrics:
                    key = f"{variable}_{metric_name}"
                    results[region_name][key] = {}

                    for lead_time in self.config.lead_times:
                        if lead_time not in forecasts[variable]:
                            continue

                        fc = np.asarray(forecasts[variable][lead_time])

                        if metric_name == "rmse":
                            val = rmse(fc, truth_var, self.latitudes, region=region)
                        elif metric_name == "mae":
                            val = mae(fc, truth_var, self.latitudes, region=region)
                        elif metric_name == "bias":
                            val = bias(fc, truth_var, self.latitudes, region=region)
                        elif metric_name == "acc" and climatology and variable in climatology:
                            clim = np.asarray(climatology[variable])
                            val = acc(fc, truth_var, clim, self.latitudes, region=region)
                        else:
                            continue

                        results[region_name][key][lead_time] = val

        # Generate scorecard if baseline provided
        scorecard = None
        if baseline_forecasts is not None:
            scorecard = generate_scorecard(
                model_forecasts=forecasts,
                baseline_forecasts=baseline_forecasts,
                truth=truth,
                climatology=climatology,
                latitudes=self.latitudes,
                model_name=model_name,
                baseline_name=baseline_name,
                metrics=self.config.metrics,
                regions=self.config.regions,
            )

        return EvaluationResult(
            config=self.config,
            metrics=results,
            scorecard=scorecard,
        )

    def evaluate_ensemble(
        self,
        ensemble_forecasts: Dict[str, Dict[int, ArrayLike]],
        truth: Dict[str, ArrayLike],
        ensemble_axis: int = 0,
    ) -> Dict[str, Dict[str, Dict[int, float]]]:
        """Evaluate ensemble forecasts with probabilistic metrics.

        Args:
            ensemble_forecasts: Ensemble forecasts {variable: {lead_time: array}}
            truth: Ground truth {variable: array}
            ensemble_axis: Axis containing ensemble members

        Returns:
            Dict of probabilistic metrics
        """
        results = {}

        for variable in ensemble_forecasts.keys():
            if variable not in truth:
                continue

            truth_var = np.asarray(truth[variable])
            results[variable] = {}

            for lead_time, ens_fc in ensemble_forecasts[variable].items():
                ens_fc = np.asarray(ens_fc)

                results[variable][lead_time] = {
                    "crps": crps(ens_fc, truth_var, ensemble_axis),
                    "spread": ensemble_spread(ens_fc, ensemble_axis),
                    "spread_skill_ratio": spread_skill_ratio(ens_fc, truth_var, ensemble_axis),
                }

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_evaluate(
    forecast: ArrayLike,
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
) -> Dict[str, float]:
    """Quick evaluation with standard metrics.

    Args:
        forecast: Predicted values
        truth: Ground truth values
        latitudes: Optional latitudes for spatial weighting

    Returns:
        Dict with rmse, mae, and bias values
    """
    return {
        "rmse": rmse(forecast, truth, latitudes),
        "mae": mae(forecast, truth, latitudes),
        "bias": bias(forecast, truth, latitudes),
    }


def compare_models(
    model_forecasts: Dict[str, ArrayLike],
    truth: ArrayLike,
    latitudes: Optional[NDArray[np.float64]] = None,
    baseline_key: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same data.

    Args:
        model_forecasts: Dict of {model_name: forecast_array}
        truth: Ground truth array
        latitudes: Optional latitudes for spatial weighting
        baseline_key: Optional model name to use as baseline for % improvements

    Returns:
        Dict of {model_name: {metric_name: value}}
    """
    results = {}

    for model_name, forecast in model_forecasts.items():
        results[model_name] = quick_evaluate(forecast, truth, latitudes)

    if baseline_key and baseline_key in results:
        baseline = results[baseline_key]
        for model_name in results:
            if model_name != baseline_key:
                for metric in ["rmse", "mae"]:
                    improvement = (baseline[metric] - results[model_name][metric]) / baseline[metric] * 100
                    results[model_name][f"{metric}_improvement_%"] = improvement

    return results
