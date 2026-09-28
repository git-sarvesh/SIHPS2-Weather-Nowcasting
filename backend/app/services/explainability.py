"""Explainability service for the multi-hazard nowcaster."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.logging_conf import get_logger
from app.physics import (
    CHANNELS,
    CONVECTIVE_CHANNELS,
    MOISTURE_CHANNELS,
    N_CHANNELS,
    TERRAIN_CHANNELS,
    channel_importance_template,
    channel_index,
)

logger = get_logger("services.explainability")

__all__ = [
    "GradCAMExplainer",
    "PhysicalConsistencyChecker",
    "WhatIfSimulator",
    "AttributionResult",
    "ConsistencyResult",
]

@dataclass(slots=True)
class AttributionResult:
    """Attribution and saliency maps for a single forecast step and hazard."""

    hazard: str
    gradcam_map: np.ndarray
    channel_attributions: dict[str, float]
    top_channels: list[tuple[str, float]]
    step: int
    model_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard": self.hazard,
            "gradcam_map": self.gradcam_map.tolist(),
            "channel_attributions": {k: round(float(v), 4) for k, v in self.channel_attributions.items()},
            "top_channels": [(k, round(float(v), 4)) for k, v in self.top_channels],
            "step": self.step,
            "model_version": self.model_version,
        }


@dataclass(slots=True)
class ConsistencyResult:
    """Physical consistency audit for model predictions."""

    consistent: bool
    score: float
    checks: dict[str, dict[str, Any]]
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "score": round(self.score, 4),
            "checks": self.checks,
            "violations": self.violations,
        }


class GradCAMExplainer:
    """Computes spatial Grad-CAM activations and input channel attributions."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def explain(
        self,
        x: np.ndarray,
        terrain: np.ndarray | None = None,
        *,
        hazard: str = "cloudburst",
        step: int = -1,
        target_point: tuple[int, int] | None = None,
    ) -> AttributionResult:
        if hasattr(self.model, "backbone") and hasattr(self.model, "heads"):
            return self._explain_torch(x, terrain, hazard=hazard, step=step, target_point=target_point)
        return self._explain_reference(x, terrain, hazard=hazard, step=step, target_point=target_point)

    def _explain_torch(
        self,
        x: np.ndarray,
        terrain: np.ndarray | None,
        *,
        hazard: str,
        step: int,
        target_point: tuple[int, int] | None,
    ) -> AttributionResult:
        import torch

        self.model.eval()
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        if x_tensor.ndim == 4:
            x_tensor = x_tensor.unsqueeze(0)
        x_tensor.requires_grad_(True)

        t_tensor = None
        if terrain is not None:
            t_tensor = torch.as_tensor(terrain, dtype=torch.float32)
            if t_tensor.ndim == 3:
                t_tensor = t_tensor.unsqueeze(0)

        out = self.model(x_tensor, t_tensor)
        heads = out["heads"]
        if hazard == "thunderstorm":
            target_prob = heads.thunderstorm_prob
        elif hazard == "flood":
            target_prob = heads.flood_prob
        else:
            target_prob = heads.cloudburst_prob

        target_slice = target_prob[0, step]
        if target_point is not None:
            r, c = target_point
            score = target_slice[r, c]
        else:
            score = target_slice.mean()

        self.model.zero_grad()
        if x_tensor.grad is not None:
            x_tensor.grad.zero_()

        score.backward(retain_graph=True)

        if x_tensor.grad is not None:
            input_grad = x_tensor.grad[0].detach().cpu().numpy()
            raw_channel_scores = np.mean(np.abs(input_grad), axis=(0, 2, 3))
            spatial_grad = np.max(np.abs(input_grad), axis=(0, 1))
            spatial_min, spatial_max = float(spatial_grad.min()), float(spatial_grad.max())
            cam_map = (
                (spatial_grad - spatial_min) / (spatial_max - spatial_min + 1e-8)
                if spatial_max > spatial_min
                else np.zeros_like(spatial_grad)
            )
        else:
            raw_channel_scores = np.ones(N_CHANNELS)
            cam_map = np.zeros(x_tensor.shape[3:], dtype=np.float32)

        total_score = float(np.sum(raw_channel_scores)) or 1.0
        channel_attrs = {
            spec.name: float(raw_channel_scores[i] / total_score)
            for i, spec in enumerate(CHANNELS)
        }
        sorted_channels = sorted(channel_attrs.items(), key=lambda kv: kv[1], reverse=True)

        return AttributionResult(
            hazard=hazard,
            gradcam_map=cam_map,
            channel_attributions=channel_attrs,
            top_channels=sorted_channels[:5],
            step=step,
            model_version=getattr(self.model, "model_version", "torch-v1"),
        )

    def _explain_reference(
        self,
        x: np.ndarray,
        terrain: np.ndarray | None,
        *,
        hazard: str,
        step: int,
        target_point: tuple[int, int] | None,
    ) -> AttributionResult:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 5:
            arr = arr[0]
        t_in, c, h, w = arr.shape

        eps = 1e-3
        base_dict = self.model.deterministic_forward(arr, terrain=terrain)
        base_pred = base_dict[hazard][step]

        channel_scores = np.zeros(N_CHANNELS, dtype=np.float64)
        for ch_idx in range(c):
            perturbed = arr.copy()
            perturbed[:, ch_idx, :, :] += eps
            perturbed_pred = self.model.deterministic_forward(perturbed, terrain=terrain)[hazard][step]
            channel_scores[ch_idx] = np.mean(np.abs(perturbed_pred - base_pred)) / eps

        total = float(np.sum(channel_scores)) or 1.0
        channel_attrs = {
            spec.name: float(channel_scores[i] / total)
            for i, spec in enumerate(CHANNELS)
        }
        sorted_channels = sorted(channel_attrs.items(), key=lambda kv: kv[1], reverse=True)

        spatial_min, spatial_max = float(base_pred.min()), float(base_pred.max())
        cam_map = (
            (base_pred - spatial_min) / (spatial_max - spatial_min + 1e-8)
            if spatial_max > spatial_min
            else np.zeros_like(base_pred)
        )

        return AttributionResult(
            hazard=hazard,
            gradcam_map=cam_map,
            channel_attributions=channel_attrs,
            top_channels=sorted_channels[:5],
            step=step,
            model_version=getattr(self.model, "model_version", "numpy-reference"),
        )


class PhysicalConsistencyChecker:
    """Verifies that ML predictions conform to atmospheric and hydrological physics."""

    def __init__(
        self,
        ctt_cold_threshold_k: float = 235.0,
        cooling_rate_threshold_k_h: float = -10.0,
        iwv_min_mm: float = 35.0,
    ) -> None:
        self.ctt_cold_threshold = ctt_cold_threshold_k
        self.cooling_threshold = cooling_rate_threshold_k_h
        self.iwv_min = iwv_min_mm

    def check(
        self,
        predictions: dict[str, np.ndarray],
        x_physical: np.ndarray,
        terrain: Any | None = None,
        attributions: AttributionResult | None = None,
    ) -> ConsistencyResult:
        arr = np.asarray(x_physical, dtype=np.float64)
        if arr.ndim == 5:
            arr = arr[0]

        ctt = arr[-1, channel_index("ctt")]
        cooling = arr[-1, channel_index("ctt_cooling_rate")]
        iwv = arr[-1, channel_index("iwv")]

        cb_prob = np.asarray(predictions["cloudburst"][-1])
        flood_prob = np.asarray(predictions["flood"][-1])

        checks: dict[str, dict[str, Any]] = {}
        violations: list[str] = []

        high_cb_mask = cb_prob > 0.6
        if np.any(high_cb_mask):
            ctt_in_storm = ctt[high_cb_mask]
            cooling_in_storm = cooling[high_cb_mask]
            cold_or_growing = (ctt_in_storm <= self.ctt_cold_threshold) | (cooling_in_storm <= self.cooling_threshold)
            valid_fraction = float(np.mean(cold_or_growing))
            status = valid_fraction >= 0.8
            checks["convective_temperature"] = {
                "passed": status,
                "conforming_fraction": round(valid_fraction, 3),
            }
            if not status:
                violations.append("Cloudburst predicted over warm/non-cooling clouds")
        else:
            checks["convective_temperature"] = {"passed": True, "conforming_fraction": 1.0}

        if np.any(high_cb_mask):
            iwv_in_storm = iwv[high_cb_mask]
            moist_fraction = float(np.mean(iwv_in_storm >= self.iwv_min))
            status = moist_fraction >= 0.75
            checks["column_moisture"] = {
                "passed": status,
                "conforming_fraction": round(moist_fraction, 3),
            }
            if not status:
                violations.append("Cloudburst predicted in atmospheric column lacking sufficient moisture")
        else:
            checks["column_moisture"] = {"passed": True, "conforming_fraction": 1.0}

        high_flood_mask = flood_prob > 0.5
        if np.any(high_flood_mask) and terrain is not None:
            flow_acc = np.asarray(terrain.flow_accumulation)
            high_flow = flow_acc[high_flood_mask]
            flow_conforming = float(np.mean(high_flow > 5))
            status = flow_conforming >= 0.6
            checks["hydrological_routing"] = {
                "passed": status,
                "conforming_fraction": round(flow_conforming, 3),
            }
            if not status:
                violations.append("Flash flood alert outside natural drainage channels or river valleys")
        else:
            checks["hydrological_routing"] = {"passed": True, "conforming_fraction": 1.0}

        if attributions is not None:
            top_channel_names = [ch for ch, _ in attributions.top_channels[:3]]
            relevant_set = set(CONVECTIVE_CHANNELS) | set(MOISTURE_CHANNELS)
            has_relevant = any(ch in relevant_set for ch in top_channel_names)
            checks["saliency_relevance"] = {
                "passed": has_relevant,
                "top_channels": top_channel_names,
            }
            if not has_relevant:
                violations.append("Model decisions dominated by non-meteorological artifact channels")

        passed_count = sum(1 for c in checks.values() if c.get("passed", False))
        total_count = max(len(checks), 1)
        score = passed_count / total_count
        consistent = len(violations) == 0

        return ConsistencyResult(
            consistent=consistent,
            score=score,
            checks=checks,
            violations=violations,
        )


class WhatIfSimulator:
    """Counterfactual scenario analysis for interactive sensitivity exploration."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def simulate(
        self,
        x_normalised: np.ndarray,
        terrain: np.ndarray | None = None,
        *,
        perturbations: dict[str, float],
        lead_hours: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    ) -> dict[str, Any]:
        arr = np.asarray(x_normalised, dtype=np.float32).copy()
        if arr.ndim == 4:
            arr = arr[np.newaxis, ...]

        if hasattr(self.model, "backbone"):
            import torch
            with torch.no_grad():
                x_t = torch.as_tensor(arr)
                t_t = torch.as_tensor(terrain).unsqueeze(0) if terrain is not None and terrain.ndim == 3 else None
                base_out = self.model(x_t, t_t)["heads"]
                base_ts = base_out.thunderstorm_prob[0].cpu().numpy()
                base_cb = base_out.cloudburst_prob[0].cpu().numpy()
                base_fl = base_out.flood_prob[0].cpu().numpy()
        else:
            base_pred = self.model.deterministic_forward(arr[0], terrain=terrain)
            base_ts = base_pred["thunderstorm"]
            base_cb = base_pred["cloudburst"]
            base_fl = base_pred["flood"]

        perturbed_x = arr.copy()
        for ch_name, delta in perturbations.items():
            idx = channel_index(ch_name)
            perturbed_x[:, :, idx, :, :] = np.clip(perturbed_x[:, :, idx, :, :] + delta, 0.0, 1.0)

        if hasattr(self.model, "backbone"):
            import torch
            with torch.no_grad():
                x_pert_t = torch.as_tensor(perturbed_x)
                pert_out = self.model(x_pert_t, t_t)["heads"]
                pert_ts = pert_out.thunderstorm_prob[0].cpu().numpy()
                pert_cb = pert_out.cloudburst_prob[0].cpu().numpy()
                pert_fl = pert_out.flood_prob[0].cpu().numpy()
        else:
            pert_pred = self.model.deterministic_forward(perturbed_x[0], terrain=terrain)
            pert_ts = pert_pred["thunderstorm"]
            pert_cb = pert_pred["cloudburst"]
            pert_fl = pert_pred["flood"]

        return {
            "baseline": {
                "thunderstorm_max": float(base_ts.max()),
                "cloudburst_max": float(base_cb.max()),
                "flood_max": float(base_fl.max()),
            },
            "counterfactual": {
                "thunderstorm_max": float(pert_ts.max()),
                "cloudburst_max": float(pert_cb.max()),
                "flood_max": float(pert_fl.max()),
            },
            "delta": {
                "thunderstorm": float(pert_ts.max() - base_ts.max()),
                "cloudburst": float(pert_cb.max() - base_cb.max()),
                "flood": float(pert_fl.max() - base_fl.max()),
            },
            "perturbations_applied": perturbations,
        }

