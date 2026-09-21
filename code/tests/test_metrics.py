from __future__ import annotations

import numpy as np

from robird.metrics import (
    aggregate_subset_records,
    brier_score,
    budget_curve,
    expected_calibration_error,
    negative_log_likelihood,
    topk_accuracy,
)


def test_perfect_probabilities_have_perfect_discrimination_and_calibration() -> None:
    probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    labels = np.asarray([0, 1])
    assert topk_accuracy(probabilities, labels, 1) == 1.0
    assert negative_log_likelihood(probabilities, labels) == 0.0
    assert brier_score(probabilities, labels) == 0.0
    assert expected_calibration_error(probabilities, labels, bins=5) == 0.0


def test_subset_predictions_are_averaged_before_group_scoring() -> None:
    records = [
        {"observation_id": 1, "budget": 1, "label": 0, "probabilities": [0.9, 0.1]},
        {"observation_id": 1, "budget": 1, "label": 0, "probabilities": [0.3, 0.7]},
    ]
    aggregated = aggregate_subset_records(records)
    assert len(aggregated) == 1
    assert aggregated[0]["subset_count"] == 2
    np.testing.assert_allclose(aggregated[0]["probabilities"], [0.6, 0.4])


def test_budget_curve_counts_corrections_without_subset_weighting() -> None:
    records = [
        {"observation_id": 1, "budget": 1, "label": 0, "probabilities": [0.4, 0.6]},
        {"observation_id": 2, "budget": 1, "label": 1, "probabilities": [0.1, 0.9]},
        {"observation_id": 3, "budget": 1, "label": 0, "probabilities": [0.9, 0.1]},
        {"observation_id": 1, "budget": 2, "label": 0, "probabilities": [0.8, 0.2]},
        {"observation_id": 2, "budget": 2, "label": 1, "probabilities": [0.2, 0.8]},
    ]
    report = budget_curve(records, ece_bins=5)
    assert report["by_budget"]["1"]["macro_top1"] == 0.75
    assert report["common_cohort_by_budget"]["1"]["macro_top1"] == 0.5
    assert report["by_budget"]["2"]["macro_top1"] == 1.0
    assert report["common_cohort_groups"] == 2
    assert report["normalized_macro_aubc"] == 0.75
    assert report["first_to_last"]["corrections"] == 1
    assert report["first_to_last"]["regressions"] == 0
