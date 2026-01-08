import numpy as np

from weatherflow.evaluation.gaia.metrics import (
    EvaluationSelectionConfig,
    acc,
    brier_score,
    crps_ensemble,
    mae,
    reliability_curve,
    rmse,
)


def test_rmse_mae() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([2.0, 2.0, 4.0])
    expected_rmse = np.sqrt(((pred - target) ** 2).mean())
    expected_mae = np.abs(pred - target).mean()

    assert rmse(pred, target) == expected_rmse
    assert mae(pred, target) == expected_mae


def test_acc_perfect_correlation() -> None:
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([2.0, 4.0, 6.0, 8.0])

    assert np.isclose(acc(pred, target), 1.0)


def test_crps_ensemble() -> None:
    ensemble = np.array([[0.0], [2.0]])
    observations = np.array([1.0])
    expected_crps = 0.5

    assert np.isclose(
        crps_ensemble(ensemble, observations, ensemble_axis=0), expected_crps
    )


def test_reliability_and_brier_score() -> None:
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])
    observations = np.array([0.0, 0.0, 1.0, 1.0])

    expected_brier = np.mean((probabilities - observations) ** 2)
    assert brier_score(probabilities, observations) == expected_brier

    result = reliability_curve(probabilities, observations, bins=2)
    assert result.counts.sum() == 4
    assert np.isclose(result.forecast_probabilities[0], np.mean([0.1, 0.4]))
    assert np.isclose(result.observed_frequencies[1], 1.0)


def test_selection_config_applies_axes() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [2.0, 2.0]])
    selection = EvaluationSelectionConfig(
        level_axis=0,
        variable_axis=1,
        level_indices=[0],
        variable_indices=[1],
    )

    expected_rmse = np.sqrt(((pred[0, 1] - target[0, 1]) ** 2).mean())
    assert rmse(pred, target, selection=selection) == expected_rmse
