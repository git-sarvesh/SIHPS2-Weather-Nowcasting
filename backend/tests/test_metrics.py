"""Correctness tests for the verification metrics.

The contingency-table metrics are checked against hand-computed values so a
sign error or an inverted ratio cannot pass silently, and the undefined cases
are checked to return ``None`` with a reason rather than a misleading zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.training.metrics import (
    brier_score,
    categorical_metrics,
    contingency_table,
    crps,
    evaluate_fields,
    evaluate_hazard,
    reliability_curve,
    safe_ratio,
    summarise,
)


def test_safe_ratio_returns_none_on_zero_denominator() -> None:
    assert safe_ratio(1.0, 2.0) == 0.5
    assert safe_ratio(0.0, 0.0) is None
    assert safe_ratio(5.0, 0.0) is None


def test_contingency_table_counts_are_correct() -> None:
    truth = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    hits, misses, false_alarms, correct_negatives = contingency_table(pred, truth)

    assert (hits, misses, false_alarms, correct_negatives) == (2.0, 2.0, 1.0, 3.0)


def test_categorical_metrics_match_hand_computed_values() -> None:
    # 2 hits, 2 misses, 1 false alarm, 3 correct negatives.
    truth = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    result = categorical_metrics(pred, truth)

    assert result["hits"] == 2.0
    assert result["misses"] == 2.0
    assert result["false_alarms"] == 1.0
    assert result["correct_negatives"] == 3.0
    assert result["csi"] == pytest.approx(2.0 / 5.0)          # hits/(H+M+FA)
    assert result["pod"] == pytest.approx(2.0 / 4.0)          # hits/(H+M)
    assert result["far"] == pytest.approx(1.0 / 3.0)          # FA/(H+FA)
    assert result["f1"] == pytest.approx(4.0 / 7.0)           # 2H/(2H+M+FA)
    assert result["defined"] is True
    assert result["undefined_reason"] is None


def test_perfect_forecast_scores_one() -> None:
    truth = np.array([1, 1, 0, 0, 1, 0])
    result = categorical_metrics(truth, truth)

    assert result["csi"] == pytest.approx(1.0)
    assert result["pod"] == pytest.approx(1.0)
    assert result["far"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(1.0)


def test_undefined_metrics_are_null_with_reasons() -> None:
    """No observed events and no forecasts: every ratio has a 0 denominator."""
    quiet = categorical_metrics(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool))
    assert quiet["pod"] is None       # hits / observed
    assert quiet["far"] is None       # false alarms / forecasts
    assert quiet["csi"] is None       # hits / (hits + misses + false alarms)
    assert quiet["f1"] is None
    assert quiet["defined"] is False
    assert set(quiet["undefined_reason"]) == {"csi", "pod", "far", "f1"}
    # Counts are still reported, so the reason for the null scores is visible.
    assert quiet["observed_events"] == 0.0
    assert quiet["forecast_events"] == 0.0
    assert quiet["correct_negatives"] == 4.0

    # Events present but nothing forecast: FAR is 0/0 while POD is a real 0.0.
    never = categorical_metrics(np.zeros(4, dtype=bool), np.ones(4, dtype=bool))
    assert never["far"] is None
    assert never["pod"] == pytest.approx(0.0)
    assert never["csi"] == pytest.approx(0.0)  # 0 hits / 4 misses
    assert "far" in never["undefined_reason"]

    # Forecasts present but no events observed: POD is 0/0, FAR is a real 1.0.
    noisy = categorical_metrics(np.ones(4, dtype=bool), np.zeros(4, dtype=bool))
    assert noisy["pod"] is None
    assert noisy["far"] == pytest.approx(1.0)
    assert "pod" in noisy["undefined_reason"]


def test_validity_mask_excludes_entries() -> None:
    truth = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0])
    valid = np.array([True, False, True, True])  # second entry is missing data

    result = categorical_metrics(pred, truth, valid=valid)

    assert result["n_valid"] == 3
    assert result["hits"] == 1.0
    assert result["misses"] == 0.0
    # The missed event must NOT be counted, because its observation is invalid.
    assert result["observed_events"] == 1.0


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        categorical_metrics(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="validity mask"):
        categorical_metrics(np.zeros(3), np.zeros(3), valid=np.ones(2, dtype=bool))


def test_brier_score_is_zero_for_a_perfect_forecast() -> None:
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    result = brier_score(truth, truth)

    assert result["brier"] == pytest.approx(0.0)
    assert result["brier_climatology"] > 0.0
    assert result["brier_skill_score"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(0.5)
    assert result["defined"] is True


def test_brier_score_hand_computation() -> None:
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    prob = np.array([0.8, 0.4, 0.6, 0.1])
    result = brier_score(prob, truth)

    expected = float(np.mean((prob - truth) ** 2))
    assert result["brier"] == pytest.approx(expected)
    # A perfect constant-forecast-at-climatology baseline scores 0.25 here.
    assert result["brier_climatology"] == pytest.approx(0.25)
    assert result["brier_skill_score"] == pytest.approx(1.0 - expected / 0.25)
    assert result["mean_forecast"] == pytest.approx(float(prob.mean()))


def test_brier_skill_score_is_none_without_events() -> None:
    """No events => climatological Brier is 0 => skill is undefined, not 0."""
    result = brier_score(np.array([0.1, 0.2]), np.array([0.0, 0.0]))

    assert result["brier_skill_score"] is None
    assert "brier_skill_score" in result["undefined_reason"]


def test_brier_handles_no_valid_observations() -> None:
    result = brier_score(np.array([]), np.array([]))

    assert result["brier"] is None
    assert result["defined"] is False
    assert result["undefined_reason"] == "no valid observations"


def test_reliability_curve_is_perfectly_calibrated_when_matching() -> None:
    # Ten bins, each with a constant forecast equal to the observed frequency.
    truth = np.repeat([0.0, 1.0], 50)
    prob = truth.copy()
    result = reliability_curve(prob, truth, n_bins=10)

    assert result["reliability"] == pytest.approx(0.0, abs=1e-12)
    assert result["resolution"] == pytest.approx(0.25, abs=1e-12)
    assert result["uncertainty"] == pytest.approx(0.25)
    assert result["defined"] is True
    # Bins cover the whole probability range.
    assert result["bins"][0]["lower"] == 0.0
    assert result["bins"][-1]["upper"] == 1.0


def test_reliability_curve_reports_bias() -> None:
    """Over-confident 1.0 forecasts against 50% events show a large reliability term."""
    truth = np.tile([0.0, 1.0], 50)
    prob = np.ones(100)
    result = reliability_curve(prob, truth, n_bins=10)

    # reliability = E[(fc - obs)^2] = (1.0 - 0.5)^2 = 0.25
    assert result["reliability"] == pytest.approx(0.25)
    assert result["resolution"] == pytest.approx(0.0)
    assert result["base_rate"] == pytest.approx(0.5)


def test_reliability_curve_handles_empty_input() -> None:
    result = reliability_curve(np.array([]), np.array([]))

    assert result["defined"] is False
    assert result["bins"] == []


# --------------------------------------------------------------------------- #
# CRPS
# --------------------------------------------------------------------------- #
def test_crps_is_zero_for_a_degenerate_ensemble_on_the_observation() -> None:
    obs = np.array([1.0, 0.0, 1.0])
    samples = np.stack([obs, obs])  # (S, n_points), no spread

    result = crps(samples, obs)
    assert result["defined"] is True
    assert result["crps"] == pytest.approx(0.0, abs=1e-12)
    assert result["n_samples"] == 2


def test_crps_penalises_a_spread_missed_event() -> None:
    obs = np.array([1.0, 1.0])
    # Ensemble that never reaches the observed 1.0 -> strictly positive CRPS.
    low = np.zeros((4, 2))
    result = crps(low, obs)

    assert result["defined"] is True
    assert result["crps"] > 0.0


def test_crps_requires_at_least_two_members() -> None:
    result = crps(np.array([0.5, 0.5]), np.array([0.0, 1.0]))

    assert result["defined"] is False
    assert "at least 2" in result["undefined_reason"]


def test_crps_supports_batched_ensembles() -> None:
    obs = np.zeros((2, 3, 4, 4))
    samples = np.random.RandomState(0).rand(5, 2, 3, 4, 4)  # (S, B, T, H, W)

    result = crps(samples, obs, sample_axis=0)
    assert result["defined"] is True
    assert result["n_samples"] == 5
    assert result["crps"] >= 0.0


# --------------------------------------------------------------------------- #
# per-hazard / per-lead aggregation
# --------------------------------------------------------------------------- #
def _stack(n_steps: int = 3, size: int = 4) -> dict[str, np.ndarray]:
    return {
        "thunderstorm": np.full((n_steps, size, size), 0.9),
        "cloudburst": np.zeros((n_steps, size, size)),
        "flood": np.full((n_steps, size, size), 0.1),
    }


def test_evaluate_hazard_reports_all_metric_blocks() -> None:
    prob = np.array([0.9, 0.9, 0.1, 0.1])
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    result = evaluate_hazard(prob, truth, threshold=0.5, n_bins=2)

    assert set(result) == {"threshold", "categorical", "brier", "reliability", "crps"}
    assert result["threshold"] == 0.5
    assert result["categorical"]["hits"] == 1.0
    assert result["categorical"]["false_alarms"] == 1.0
    # No ensemble supplied -> CRPS explicitly undefined.
    assert result["crps"]["defined"] is False
    assert "mc-samples" in result["crps"]["undefined_reason"]


def test_evaluate_hazard_uses_the_requested_threshold() -> None:
    prob = np.array([0.4, 0.4, 0.4])
    truth = np.array([1.0, 0.0, 0.0])

    low = evaluate_hazard(prob, truth, threshold=0.3)["categorical"]
    high = evaluate_hazard(prob, truth, threshold=0.6)["categorical"]

    assert low["forecast_events"] == 3.0
    assert high["forecast_events"] == 0.0
    assert high["pod"] == pytest.approx(0.0)


def test_evaluate_fields_reports_per_lead_times() -> None:
    n_steps, size = 3, 2
    predictions = {name: np.full((n_steps, size, size), 0.9) for name in
                   ("thunderstorm", "cloudburst", "flood")}
    targets = {
        "thunderstorm": np.tile(np.array([[1.0, 0.0], [0.0, 0.0]]), (n_steps, 1, 1)),
        "cloudburst": np.zeros((n_steps, size, size)),
        "flood": np.ones((n_steps, size, size)),
    }
    report = evaluate_fields(predictions, targets, lead_hours=[0.5, 1.0, 1.5], n_bins=2)

    assert set(report["pooled"]) == {"thunderstorm", "cloudburst", "flood"}
    assert len(report["by_lead"]["thunderstorm"]) == n_steps
    for step, entry in enumerate(report["by_lead"]["thunderstorm"]):
        assert entry["step"] == step
        assert entry["lead_hours"] == [0.5, 1.0, 1.5][step]
    assert report["lead_hours"] == [0.5, 1.0, 1.5]
    assert report["thresholds"]["thunderstorm"] == 0.5


def test_evaluate_fields_records_undefined_metrics() -> None:
    """A lead time with no observed events must be reported, not hidden."""
    predictions = {"thunderstorm": np.full((1, 2, 2), 0.05)}
    targets = {"thunderstorm": np.zeros((1, 2, 2))}
    report = evaluate_fields(predictions, targets, lead_hours=[0.5], n_bins=2)

    undefined = {(u["hazard"], u["metric"]) for u in report["undefined_metrics"]}
    assert ("thunderstorm", "pod") in undefined
    assert ("thunderstorm", "brier_skill_score") in undefined
    # A defined metric must not be listed.
    assert ("thunderstorm", "brier") not in undefined
    assert all(entry["reason"] for entry in report["undefined_metrics"])


def test_evaluate_fields_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="prediction shape"):
        evaluate_fields(
            {"thunderstorm": np.zeros((2, 2, 2))}, {"thunderstorm": np.zeros((3, 2, 2))}
        )


def test_evaluate_fields_passes_ensembles_for_crps() -> None:
    n_steps, size = 2, 2
    predictions = {"thunderstorm": np.full((n_steps, size, size), 0.6)}
    targets = {
        "thunderstorm": np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
    }
    ensembles = {"thunderstorm": np.random.RandomState(1).rand(4, n_steps, size, size)}

    report = evaluate_fields(
        predictions, targets, lead_hours=[0.5, 1.0], n_bins=2, ensembles=ensembles
    )
    assert report["pooled"]["thunderstorm"]["crps"]["defined"] is True
    assert report["by_lead"]["thunderstorm"][0]["crps"]["defined"] is True


def test_summarise_orders_worst_lead_times_first() -> None:
    n_steps, size = 2, 2
    predictions = {
        "thunderstorm": np.full((n_steps, size, size), 0.9),
        "cloudburst": np.zeros((n_steps, size, size)),
        "flood": np.zeros((n_steps, size, size)),
    }
    targets = {name: np.ones((n_steps, size, size)) for name in predictions}
    report = evaluate_fields(predictions, targets, lead_hours=[0.5, 1.0], n_bins=2)
    summary = summarise(report, n_worst=3)

    assert "worst_lead_times" in summary
    assert summary["n_undefined_metrics"] == len(report["undefined_metrics"])
    csis = [row["csi"] for row in summary["worst_lead_times"]]
    assert csis == sorted(csis)  # ascending CSI == worst first


def test_summary_report_is_json_serialisable() -> None:
    import json

    n_steps, size = 2, 2
    predictions = {name: np.full((n_steps, size, size), 0.1) for name in
                   ("thunderstorm", "cloudburst", "flood")}
    targets = {name: np.zeros((n_steps, size, size)) for name in predictions}
    report = evaluate_fields(predictions, targets, lead_hours=[0.5, 1.0], n_bins=2)

    encoded = json.dumps(report, default=str)
    # No observed events anywhere -> POD must survive JSON as null, not 0.0.
    assert json.loads(encoded)["pooled"]["thunderstorm"]["categorical"]["pod"] is None
