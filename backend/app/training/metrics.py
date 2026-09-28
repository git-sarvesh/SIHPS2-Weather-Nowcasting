"""Verification metrics for probabilistic nowcasts.

Implemented here (NumPy only, no external dependency):

* **Categorical** - CSI (critical success index / threat score), POD (hit rate),
  FAR (false alarm ratio) and F1, from the 2x2 contingency table.
* **Probabilistic** - Brier score, Brier skill score against climatology, and
  reliability-curve statistics (bias, resolution, mean forecast probability).
* **Ensemble** - CRPS via the unbiased estimator, reused from
  :mod:`app.models.uncertainty` so training and evaluation agree.

Undefined cases are explicit, never silently coerced to zero: every ratio with
an empty denominator returns ``None`` together with a ``defined`` flag and a
human-readable ``undefined_reason``. Missing observations must be passed as a
validity mask and are excluded rather than counted as "no event".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.models.uncertainty import crps_ensemble

__all__ = [
    "brier_score",
    "categorical_metrics",
    "contingency_table",
    "crps",
    "evaluate_fields",
    "evaluate_hazard",
    "reliability_curve",
    "safe_ratio",
    "summarise",
]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is 0."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def contingency_table(
    predictions: np.ndarray, targets: np.ndarray, valid: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """Return ``(hits, misses, false_alarms, correct_negatives)``.

    Parameters
    ----------
    predictions:
        Binary forecast (``True`` counts as a forecast event).
    targets:
        Binary truth.
    valid:
        Optional boolean mask of valid (observed) entries; invalid entries are
        excluded from all counts.
    """
    pred = np.asarray(predictions).astype(bool).ravel()
    truth = np.asarray(targets).astype(bool).ravel()
    if pred.size != truth.size:
        raise ValueError(f"shape mismatch: {pred.size} predictions vs {truth.size} targets")
    if valid is not None:
        mask = np.asarray(valid).astype(bool).ravel()
        if mask.size != pred.size:
            raise ValueError("validity mask must match the prediction size")
        pred, truth = pred[mask], truth[mask]
    hits = float(np.count_nonzero(pred & truth))
    misses = float(np.count_nonzero(~pred & truth))
    false_alarms = float(np.count_nonzero(pred & ~truth))
    correct_negatives = float(np.count_nonzero(~pred & ~truth))
    return hits, misses, false_alarms, correct_negatives


def categorical_metrics(
    predictions: np.ndarray, targets: np.ndarray, *, valid: np.ndarray | None = None
) -> dict[str, Any]:
    """CSI, POD, FAR and F1 with explicit undefined handling."""
    hits, misses, false_alarms, correct_negatives = contingency_table(predictions, targets, valid)
    observed = hits + misses
    forecast = hits + false_alarms
    total = hits + misses + false_alarms + correct_negatives
    csi = safe_ratio(hits, hits + misses + false_alarms)
    pod = safe_ratio(hits, observed)
    far = safe_ratio(false_alarms, forecast)
    f1 = safe_ratio(2.0 * hits, 2.0 * hits + misses + false_alarms)
    reasons: dict[str, str] = {}
    if csi is None:
        reasons["csi"] = "no forecasts and no observed events (0/0)"
    if pod is None:
        reasons["pod"] = "no observed events (0/0)"
    if far is None:
        reasons["far"] = "no forecasts issued (0/0)"
    if f1 is None:
        reasons["f1"] = "no hits, misses or false alarms (0/0)"
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "observed_events": observed,
        "forecast_events": forecast,
        "n_valid": total,
        "csi": csi,
        "pod": pod,
        "far": far,
        "f1": f1,
        "defined": csi is not None,
        "undefined_reason": reasons or None,
    }


def brier_score(
    probabilities: np.ndarray, targets: np.ndarray, *, valid: np.ndarray | None = None
) -> dict[str, Any]:
    """Brier score, climatological Brier score and Brier skill score.

    ``BSS = 1 - BS / BS_ref`` with ``BS_ref`` the Brier score of the
    climatological (sample-mean) forecast. ``BSS <= 0`` means the forecast is no
    better than that constant baseline; the value is reported, not hidden.
    """
    prob = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(targets, dtype=np.float64).ravel()
    if prob.size != truth.size:
        raise ValueError(f"shape mismatch: {prob.size} probabilities vs {truth.size} targets")
    if valid is not None:
        mask = np.asarray(valid).astype(bool).ravel()
        prob, truth = prob[mask], truth[mask]
    n = prob.size
    if n == 0:
        return {
            "brier": None,
            "brier_climatology": None,
            "brier_skill_score": None,
            "n_valid": 0,
            "base_rate": None,
            "mean_forecast": None,
            "defined": False,
            "undefined_reason": "no valid observations",
        }
    prob = np.clip(prob, 0.0, 1.0)
    bs = float(np.mean((prob - truth) ** 2))
    base_rate = float(truth.mean())
    bs_ref = float(np.mean((base_rate - truth) ** 2))
    bss = None if bs_ref == 0 else 1.0 - bs / bs_ref
    reasons: dict[str, str] = {}
    if bss is None:
        reasons["brier_skill_score"] = (
            "climatological Brier score is 0 (no events observed), so skill is undefined"
        )
    return {
        "brier": bs,
        "brier_climatology": bs_ref,
        "brier_skill_score": bss,
        "n_valid": int(n),
        "base_rate": base_rate,
        "mean_forecast": float(prob.mean()),
        "defined": True,
        "undefined_reason": reasons or None,
    }


def reliability_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 10,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reliability diagram data plus the Brier decomposition (Murphy 1973).

    ``reliability`` (small = well calibrated) and ``resolution`` (large = sharp
    forecasts) are reported together, since neither alone describes skill.
    """
    prob = np.asarray(probabilities, dtype=np.float64).ravel()
    truth = np.asarray(targets, dtype=np.float64).ravel()
    if prob.size != truth.size:
        raise ValueError("probabilities and targets must have the same size")
    if valid is not None:
        mask = np.asarray(valid).astype(bool).ravel()
        prob, truth = prob[mask], truth[mask]
    if prob.size == 0:
        return {
            "bins": [],
            "base_rate": None,
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
            "defined": False,
            "undefined_reason": "no valid observations",
        }
    prob = np.clip(prob, 0.0, 1.0)
    base_rate = float(truth.mean())
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    bins: list[dict[str, Any]] = []
    reliability_component = 0.0
    resolution_component = 0.0
    for index in range(len(edges) - 1):
        lo, hi = edges[index], edges[index + 1]
        last = index == len(edges) - 2
        in_bin = (prob >= lo) & (prob <= hi) if last else (prob >= lo) & (prob < hi)
        count = int(np.count_nonzero(in_bin))
        entry: dict[str, Any] = {
            "bin_index": index,
            "lower": float(lo),
            "upper": float(hi),
            "count": count,
            "mean_forecast": float(prob[in_bin].mean()) if count else None,
            "observed_frequency": float(truth[in_bin].mean()) if count else None,
        }
        if count:
            # Murphy (1973): the reliability term compares the mean forecast with
            # the *observed* frequency of the bin; using the overall base rate
            # here would score a perfectly deterministic forecast as unreliable.
            reliability_component += count * (entry["mean_forecast"] - entry["observed_frequency"]) ** 2
            resolution_component += count * (entry["observed_frequency"] - base_rate) ** 2
        bins.append(entry)
    total = float(prob.size)
    return {
        "bins": bins,
        "base_rate": base_rate,
        "reliability": reliability_component / total,
        "resolution": resolution_component / total,
        "uncertainty": float(np.mean((truth - base_rate) ** 2)),
        "defined": True,
        "undefined_reason": None,
    }


def crps(
    samples: np.ndarray,
    observations: np.ndarray,
    *,
    sample_axis: int = 0,
) -> dict[str, Any]:
    """Ensemble CRPS (unbiased estimator from :mod:`app.models.uncertainty`).

    Only meaningful where the output representation is an ensemble, so callers
    must supply at least two members; a single-member input is reported as
    undefined rather than approximated. ``sample_axis`` selects the ensemble axis
    (``(S, ...)`` by default, ``(B, S, ...)`` for batched predictions) and every
    other axis is matched against the observation points.
    """
    ens = np.asarray(samples, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64)
    if ens.ndim < 2 or ens.shape[int(sample_axis)] < 2:
        axis = int(sample_axis) if sample_axis < ens.ndim else 0
        return {
            "crps": None,
            "n_samples": int(ens.shape[axis]) if ens.ndim else 0,
            "defined": False,
            "undefined_reason": "CRPS needs an ensemble of at least 2 members",
        }
    if obs.ndim == 0:
        return {
            "crps": None,
            "n_samples": int(ens.shape[int(sample_axis)]),
            "defined": False,
            "undefined_reason": "observations must be an array, not a scalar",
        }
    # Move the sample axis last, then flatten the remaining axes to points.
    aligned = np.moveaxis(ens, int(sample_axis), -1)
    n_points = int(np.prod(obs.shape))
    aligned = aligned.reshape(-1, aligned.shape[-1])
    if aligned.shape[0] != n_points:
        return {
            "crps": None,
            "n_samples": int(ens.shape[int(sample_axis)]),
            "defined": False,
            "undefined_reason": (
                f"ensemble point count {aligned.shape[0]} != observation count {n_points}"
            ),
        }
    try:
        value = crps_ensemble(aligned.T, obs)
    except ValueError as exc:
        return {
            "crps": None,
            "n_samples": int(ens.shape[int(sample_axis)]),
            "defined": False,
            "undefined_reason": str(exc),
        }
    return {
        "crps": value,
        "n_samples": int(aligned.shape[-1]),
        "defined": True,
        "undefined_reason": None,
    }


def evaluate_hazard(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    samples: np.ndarray | None = None,
    sample_axis: int = 0,
) -> dict[str, Any]:
    """All metrics for one hazard field (one or more lead times).

    Parameters
    ----------
    probabilities:
        Forecast probability in ``[0, 1]``.
    targets:
        Binary (or soft) truth of the same shape.
    valid:
        Optional validity mask; invalid entries are excluded everywhere.
    threshold:
        Event threshold for the categorical scores.
    samples:
        Optional ensemble enabling CRPS.
    sample_axis:
        Which axis of ``samples`` is the ensemble axis.
    """
    prob = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    truth = np.asarray(targets, dtype=np.float64)
    categorical = categorical_metrics(prob >= float(threshold), truth >= 0.5, valid=valid)
    brier = brier_score(prob, truth, valid=valid)
    reliability = reliability_curve(prob, truth, n_bins=n_bins, valid=valid)
    payload: dict[str, Any] = {
        "threshold": float(threshold),
        "categorical": categorical,
        "brier": brier,
        "reliability": reliability,
    }
    if samples is not None:
        payload["crps"] = crps(
            np.asarray(samples, dtype=np.float64), truth, sample_axis=sample_axis
        )
    else:
        payload["crps"] = {
            "crps": None,
            "n_samples": 0,
            "defined": False,
            "undefined_reason": "no ensemble supplied; run with --mc-samples to enable CRPS",
        }
    return payload


def evaluate_fields(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    lead_hours: list[float] | None = None,
    thresholds: dict[str, float] | None = None,
    valid: dict[str, np.ndarray] | None = None,
    n_bins: int = 10,
    ensembles: dict[str, np.ndarray] | None = None,
    ensemble_sample_axis: int = 0,
) -> dict[str, Any]:
    """Evaluate every hazard, reporting pooled *and* per-lead-time scores.

    ``predictions``/``targets`` are ``(T, H, W)`` stacks. The pooled entry
    flattens all lead times; the ``by_lead`` entries keep them separate, because
    skill degrades with lead time and a pooled number hides that.

    ``ensemble_sample_axis`` tells the function which axis of the supplied
    ensembles is the sample axis: ``0`` for ``(S, T, H, W)`` or ``1`` for the
    batched ``(B, S, T, H, W)`` that :mod:`app.training.evaluate` produces.
    """
    thresholds = dict(thresholds or {"thunderstorm": 0.5, "cloudburst": 0.5, "flood": 0.5})
    n_bins = max(2, int(n_bins))
    sample_axis = int(ensemble_sample_axis)
    hazards = [name for name in ("thunderstorm", "cloudburst", "flood") if name in predictions]
    report: dict[str, Any] = {
        "pooled": {},
        "by_lead": {},
        "lead_hours": list(lead_hours or []),
        "thresholds": thresholds,
        "undefined_metrics": [],
    }
    for hazard in hazards:
        pred_stack = np.asarray(predictions[hazard], dtype=np.float64)
        truth_stack = np.asarray(targets[hazard], dtype=np.float64)
        if truth_stack.shape != pred_stack.shape:
            raise ValueError(
                f"{hazard}: prediction shape {pred_stack.shape} != target shape {truth_stack.shape}"
            )
        mask_stack = np.asarray(valid[hazard]).astype(bool) if valid and hazard in valid else None
        sample_stack = (
            np.asarray(ensembles[hazard], dtype=np.float64) if ensembles and hazard in ensembles else None
        )
        n_steps = pred_stack.shape[0] if pred_stack.ndim >= 3 else 1
        report["pooled"][hazard] = evaluate_hazard(
            pred_stack.reshape(-1),
            truth_stack.reshape(-1),
            valid=None if mask_stack is None else mask_stack.reshape(-1),
            threshold=thresholds.get(hazard, 0.5),
            n_bins=n_bins,
            samples=None if sample_stack is None else sample_stack,
            sample_axis=sample_axis,
        )
        per_lead: list[dict[str, Any]] = []
        for step in range(n_steps):
            entry = evaluate_hazard(
                pred_stack[step],
                truth_stack[step],
                valid=None if mask_stack is None else mask_stack[step],
                threshold=thresholds.get(hazard, 0.5),
                n_bins=n_bins,
                samples=None if sample_stack is None else sample_stack[:, step],
            )
            entry["step"] = step
            if lead_hours and step < len(lead_hours):
                entry["lead_hours"] = lead_hours[step]
            per_lead.append(entry)
            for block_name, keys in (
                ("categorical", ("csi", "pod", "far", "f1")),
                ("brier", ("brier", "brier_skill_score")),
            ):
                block = entry[block_name]
                for key in keys:
                    if block.get(key) is None:
                        report["undefined_metrics"].append(
                            {
                                "hazard": hazard,
                                "scope": "by_lead",
                                "step": step,
                                "metric": key,
                                "reason": block.get("undefined_reason"),
                            }
                        )
        report["by_lead"][hazard] = per_lead
    return report


def summarise(report: dict[str, Any], *, n_worst: int = 5) -> dict[str, Any]:
    """Compact summary: the weakest lead times and the undefined-metric count."""
    rows: list[dict[str, Any]] = []
    for hazard, entries in (report.get("by_lead") or {}).items():
        for entry in entries:
            rows.append(
                {
                    "hazard": hazard,
                    "step": entry.get("step"),
                    "lead_hours": entry.get("lead_hours"),
                    "csi": entry["categorical"].get("csi"),
                    "brier_skill_score": entry["brier"].get("brier_skill_score"),
                    "n_valid": entry["categorical"].get("n_valid"),
                }
            )
    defined = sorted(
        (row for row in rows if row["csi"] is not None), key=lambda row: row["csi"]
    )
    undefined = report.get("undefined_metrics") or []
    return {
        "worst_lead_times": defined[: max(0, int(n_worst))],
        "n_undefined_metrics": len(undefined),
        "undefined_metrics": undefined[: max(0, int(n_worst))],
    }
