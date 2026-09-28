"""Shared inference service backing the FastAPI layer.

This module wires together the *existing* components - synthetic ingestion,
terrain processing, the CNN-ConvLSTM multi-task model, the risk engine and the
explainability service - behind one object that the API routes call.

Design constraints
------------------
* **Explicitly synthetic demo mode.** Every response carries ``demo_mode`` /
  ``is_synthetic`` flags plus the experimental disclaimer. Nothing here is an
  operational weather warning and no forecasting-accuracy claim is made: the
  model is randomly initialised unless a trained checkpoint exists, and the
  inputs are synthetic fields.
* **Lazy runtime.** Model, terrain and the synthetic event are built on first
  use so importing :mod:`app.main` stays cheap.
* **No invented numbers.** Every field returned here is produced by the
  existing model / risk engine / explainability code paths.

This phase intentionally does not add persistence (SQLAlchemy), Celery
scheduling, live MOSDAC/IMDAA ingestion or a trained checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.grid import GridSpec
from app.ingestion.base import EXPERIMENTAL_DISCLAIMER, ObservationCube, WindowSpec
from app.ingestion.synthetic import (
    SYNTHETIC_ATTRIBUTION,
    EventSpec,
    SyntheticINSATConnector,
    build_demo_terrain,
    default_event_catalog,
)
from app.logging_conf import get_logger
from app.models.network import TASK_NAMES
from app.models.registry import build_model, load_checkpoint, resolve_backend
from app.services.explainability import GradCAMExplainer, PhysicalConsistencyChecker, WhatIfSimulator
from app.services.risk_engine import RiskEngine, RiskResult, RiskWeights

logger = get_logger("services.inference")


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, used to pin a checkpoint in a persisted forecast run."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

__all__ = [
    "InferenceService",
    "ModelUnavailableError",
    "UnknownEventError",
    "get_inference_service",
    "reset_inference_service",
]

#: Default synthetic event when a request does not name one (Uttarkashi cloudburst).
DEFAULT_EVENT_INDEX = 2

#: Hazard identifiers accepted by the forecast and explain routes.
SUPPORTED_HAZARDS: tuple[str, ...] = tuple(TASK_NAMES)


class ModelUnavailableError(RuntimeError):
    """Raised when no usable predictor can be constructed."""


class UnknownEventError(LookupError):
    """Raised when a requested synthetic event id does not exist."""


@dataclass(slots=True)
class _Runtime:
    """Heavy cached objects backing the service (built once, reused)."""

    grid: GridSpec
    terrain: Any
    exposure: np.ndarray
    model: Any
    backend: str
    model_version: str
    describe: dict[str, Any]
    cube: ObservationCube
    spec: EventSpec
    window: WindowSpec
    #: ``(1, T_in, C, H, W)`` normalised model input.
    x_model: np.ndarray
    #: ``(1, T_in, C, H, W)`` physical-unit copy (XAI / consistency checks).
    x_physical: np.ndarray
    #: ``(1, 4, H, W)`` terrain tensor for the flood head late fusion.
    terrain_tensor: np.ndarray
    #: Lead time (h) of each forecast step emitted by the model.
    lead_hours: list[float]
    init_time: datetime
    risk_engine: RiskEngine
    created_at: float = field(default_factory=time.time)

    @property
    def input_shape(self) -> list[int]:
        return [int(v) for v in self.x_model.shape]

    @property
    def n_steps(self) -> int:
        return len(self.lead_hours)


class InferenceService:
    """Single entry point used by every API route.

    Parameters
    ----------
    settings:
        Optional settings override (tests inject a coarser grid this way).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._runtime: _Runtime | None = None
        self._unavailable_reason: str | None = None

    # ------------------------------------------------------------------ utils
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def demo_mode(self) -> bool:
        return bool(self._settings.demo_mode)

    @property
    def disclaimer(self) -> str:
        return EXPERIMENTAL_DISCLAIMER

    def attribution(self) -> dict[str, Any]:
        """Provenance block attached to *every* response."""
        return {
            "demo_mode": self.demo_mode,
            "is_synthetic": True,
            "data_source": "SIHPS synthetic demo generator (not an observation)",
            "attribution": SYNTHETIC_ATTRIBUTION,
            "disclaimer": self.disclaimer,
            "accuracy_claim": (
                "No forecasting accuracy is claimed. The model is randomly initialised "
                "unless a trained checkpoint is present, and no independent observational "
                "validation has been performed."
            ),
        }

    def invalidate(self) -> None:
        """Drop the cached runtime so the next call rebuilds it."""
        self._runtime = None
        self._unavailable_reason = None

    # ---------------------------------------------------------------- runtime
    def _event_spec(self, event_id: str | None) -> EventSpec:
        catalog = default_event_catalog()
        if event_id is None:
            return catalog[min(DEFAULT_EVENT_INDEX, len(catalog) - 1)]
        for spec in catalog:
            if spec.event_id == event_id:
                return spec
        raise UnknownEventError(f"unknown synthetic event_id {event_id!r}")

    def _build_model(self) -> tuple[str, Any, bool]:
        """Load a trained checkpoint when present, else build a fresh model."""
        backend = resolve_backend(self._settings.model_backend)
        directory = self._settings.model_dir_path / self._settings.model_version
        try:
            model = load_checkpoint(directory, backend=backend, map_location=self._settings.device)
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not kill the API
            logger.warning("checkpoint load failed, using fresh weights", extra={"error": str(exc)})
            model = None
        if model is not None:
            return backend, model, True
        if not self.demo_mode:
            raise ModelUnavailableError(
                f"no trained checkpoint found at '{directory}'. Run training first, or start "
                "the API with SIHPS_DEMO_MODE=true for the clearly-labelled synthetic demo."
            )
        # Demo mode: untrained weights. The response metadata states this plainly.
        return backend, build_model(preset="lite", backend=backend), False

    def ensure_runtime(self, event_id: str | None = None) -> _Runtime:
        """Return the cached runtime, building it on first use."""
        if self._runtime is not None and event_id is None:
            return self._runtime
        try:
            runtime = self._build_runtime(event_id)
        except (ModelUnavailableError, UnknownEventError):
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as 503 by the routes
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            logger.exception("inference runtime unavailable")
            raise ModelUnavailableError(f"could not initialise the demo runtime: {exc}") from exc
        if event_id is None:
            self._runtime = runtime
        return runtime

    def _build_runtime(self, event_id: str | None) -> _Runtime:
        settings = self._settings
        spec = self._event_spec(event_id)
        grid = settings.grid
        logger.info("building demo terrain", extra={"grid": grid.to_dict()})
        terrain, exposure = build_demo_terrain(grid, seed=spec.seed)

        connector = SyntheticINSATConnector(grid, terrain)
        cube = connector.fetch(spec, exposure=exposure)

        window = WindowSpec(seq_len=settings.sequence_length, horizon=settings.forecast_steps)
        if window.total_frames > cube.n_frames:
            raise ModelUnavailableError(
                f"synthetic event has {cube.n_frames} frames, but the configured window "
                f"needs {window.total_frames}"
            )
        x_norm, _labels, x_phys = cube.window(window, 0)
        # ``ObservationCube.window`` yields ``(C, T, H, W)``; the network wants
        # ``(B, T, C, H, W)``.
        x_model = np.ascontiguousarray(x_norm.transpose(1, 0, 2, 3)[None], dtype=np.float32)
        x_physical = np.ascontiguousarray(x_phys.transpose(1, 0, 2, 3)[None], dtype=np.float32)
        terrain_tensor = np.ascontiguousarray(terrain.to_feature_stack()[None], dtype=np.float32)

        backend, model, trained = self._build_model()
        model_version = str(getattr(model, "model_version", settings.model_version))
        describe: dict[str, Any] = dict(model.describe()) if hasattr(model, "describe") else {}
        describe.setdefault("model_version", model_version)
        describe["trained_checkpoint_loaded"] = bool(trained)
        describe["backend"] = backend

        risk_engine = RiskEngine(
            terrain,
            exposure=exposure,
            weights=RiskWeights(*settings.risk_weight_tuple),
            thresholds=settings.threshold_tuple,
            exposure_weights=settings.exposure_weight_tuple,
            disclaimer=self.disclaimer,
        )
        return _Runtime(
            grid=grid,
            terrain=terrain,
            exposure=exposure,
            model=model,
            backend=backend,
            model_version=model_version,
            describe=describe,
            cube=cube,
            spec=spec,
            window=window,
            x_model=x_model,
            x_physical=x_physical,
            terrain_tensor=terrain_tensor,
            lead_hours=list(window.horizon_hours),
            init_time=cube.newest_time(),
            risk_engine=risk_engine,
        )

    # ------------------------------------------------------------------ meta
    def health(self) -> dict[str, Any]:
        """Component health. Never raises: reports the failure reason instead."""
        settings = self._settings
        payload: dict[str, Any] = {
            "status": "ok",
            "app_name": settings.app_name,
            "env": settings.env,
            "runtime_loaded": self._runtime is not None,
            "components": {
                "grid": {"status": "ok", **settings.grid.to_dict()},
                "terrain": {"status": "unknown", "loaded": False},
                "model": {
                    "status": "unknown",
                    "loaded": False,
                    "requested_backend": settings.model_backend,
                    "resolved_backend": None,
                    "model_version": settings.model_version,
                },
                "risk_engine": {"status": "unknown", "loaded": False},
                "explainability": {"status": "unknown", "loaded": False},
            },
            "settings": settings.public_dict(),
            **self.attribution(),
        }
        if self._unavailable_reason:
            payload["status"] = "degraded"
            payload["error"] = self._unavailable_reason
            payload["components"]["model"]["status"] = "unavailable"
            return payload

        try:
            runtime = self.ensure_runtime()
        except ModelUnavailableError as exc:
            payload["status"] = "degraded"
            payload["error"] = str(exc)
            payload["components"]["model"]["status"] = "unavailable"
            return payload

        payload["runtime_loaded"] = True
        payload["components"]["terrain"] = {
            "status": "ok",
            "loaded": True,
            **runtime.terrain.statistics(),
        }
        payload["components"]["model"] = {
            "status": "ok",
            "loaded": True,
            "requested_backend": settings.model_backend,
            "resolved_backend": runtime.backend,
            "model_version": runtime.model_version,
            "trained_checkpoint_loaded": bool(runtime.describe.get("trained_checkpoint_loaded")),
            "input_shape": runtime.input_shape,
        }
        payload["components"]["risk_engine"] = {
            "status": "ok",
            "loaded": True,
            "weights": list(settings.risk_weight_tuple),
            "thresholds": list(settings.threshold_tuple),
        }
        payload["components"]["explainability"] = {
            "status": "ok",
            "loaded": True,
            "hazards": list(SUPPORTED_HAZARDS),
        }
        payload["demo_event"] = runtime.spec.to_dict()
        return payload

    def describe_model(self) -> dict[str, Any]:
        """``/model/describe`` payload."""
        runtime = self.ensure_runtime()
        return {
            "model_version": runtime.model_version,
            "backend": runtime.backend,
            "describe": runtime.describe,
            "grid": runtime.grid.to_dict(),
            "input_shape": runtime.input_shape,
            "terrain_shape": [int(v) for v in runtime.terrain_tensor.shape],
            "lead_times_h": runtime.lead_hours,
            "n_forecast_steps": runtime.n_steps,
            "hazards": list(SUPPORTED_HAZARDS),
            "synthetic_event": runtime.spec.to_dict(),
            "provenance": runtime.cube.provenance_summary(),
            **self.attribution(),
        }

    # ------------------------------------------------------------- inference
    def _model_inputs(self, runtime: _Runtime):
        """Backend-appropriate model inputs.

        The PyTorch network takes normalised ``(1, T, C, H, W)`` tensors, while
        the NumPy reference takes *physical* ``(T, C, H, W)`` arrays. Both use the
        ``(4, H, W)`` terrain tensor for the flood head.
        """
        if runtime.backend == "torch":
            import torch

            return (
                torch.as_tensor(runtime.x_model, dtype=torch.float32),
                torch.as_tensor(runtime.terrain_tensor, dtype=torch.float32),
            )
        return runtime.x_physical[0], runtime.terrain_tensor[0]

    @staticmethod
    def _has_batch_axis(arrays: dict[str, np.ndarray]) -> bool:
        """The PyTorch contract keeps a leading batch axis; NumPy's does not."""
        return any(np.asarray(v).ndim >= 5 or np.asarray(v).shape[0] == 1 for v in arrays.values())

    def _deterministic(self, runtime: _Runtime) -> dict[str, np.ndarray]:
        """One deterministic pass as ``(T, H, W)`` per hazard field."""
        model = runtime.model
        x, terrain_t = self._model_inputs(runtime)
        if hasattr(model, "deterministic_forward"):
            out = {key: np.asarray(value) for key, value in model.deterministic_forward(x, terrain_t).items()}
        else:  # torch module without the convenience wrapper
            import torch

            with torch.no_grad():
                heads = model(x, terrain_t)["heads"]
            out = {
                "thunderstorm": heads.thunderstorm_prob.cpu().numpy(),
                "cloudburst": heads.cloudburst_prob.cpu().numpy(),
                "flood": heads.flood_prob.cpu().numpy(),
            }
        if out and self._has_batch_axis(out):
            out = {key: value[0] for key, value in out.items()}
        self._align_lead_hours(runtime, out["thunderstorm"].shape[0])
        return out

    @staticmethod
    def _align_lead_hours(runtime: _Runtime, n_steps: int) -> None:
        """Trim lead times when a backend emits fewer steps than configured."""
        if 0 < n_steps < len(runtime.lead_hours):
            logger.warning(
                "backend emitted fewer forecast steps than configured",
                extra={"emitted": int(n_steps), "configured": len(runtime.lead_hours)},
            )
            runtime.lead_hours = runtime.lead_hours[:n_steps]

    def _ensemble(self, runtime: _Runtime, n_samples: int) -> dict[str, np.ndarray] | None:
        """MC-dropout ensemble ``(S, T, H, W)``, or ``None`` when unsupported."""
        model = runtime.model
        if not hasattr(model, "mc_forward"):
            return None
        samples = max(2, int(n_samples))
        x, terrain_t = self._model_inputs(runtime)
        out = {key: np.asarray(value) for key, value in model.mc_forward(x, terrain_t, n_samples=samples).items()}
        return out

    def _resolve_lead(self, runtime: _Runtime, lead_hours: float | None) -> int:
        """Map a requested lead time (hours) onto a forecast-step index."""
        if lead_hours is None:
            return runtime.n_steps - 1
        distances = [abs(h - float(lead_hours)) for h in runtime.lead_hours]
        closest = int(np.argmin(distances))
        if distances[closest] > 1e-6:
            raise ValueError(f"lead_hours must be one of {runtime.lead_hours}; got {lead_hours}")
        return closest

    def _risk(
        self,
        runtime: _Runtime,
        prediction: dict[str, np.ndarray],
        step: int,
        *,
        uncertainty: dict[str, np.ndarray] | None = None,
    ) -> RiskResult:
        return runtime.risk_engine.compute(
            prediction,
            init_time=runtime.init_time,
            lead_hours=runtime.lead_hours,
            step=step,
            model_version=runtime.model_version,
            uncertainty=uncertainty,
            metadata={"event_id": runtime.spec.event_id, "event_kind": runtime.spec.kind},
        )

    @staticmethod
    def _field_stats(stack: np.ndarray, step: int) -> dict[str, float]:
        arr = np.asarray(stack)[step]
        return {
            "max": round(float(arr.max()), 6),
            "mean": round(float(arr.mean()), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
        }

    @staticmethod
    def _valid_time(runtime: _Runtime, step: int) -> str:
        return (runtime.init_time + timedelta(hours=runtime.lead_hours[step])).isoformat()

    # ------------------------------------------------------ public accessors
    # Used by the persistence layer and the Celery tasks so they do not have to
    # reach into private helpers.
    def deterministic_fields(self, runtime: _Runtime | None = None) -> dict[str, np.ndarray]:
        """Hazard probability fields ``(T, H, W)`` for the cached/default runtime."""
        runtime = runtime or self.ensure_runtime()
        return self._deterministic(runtime)

    def resolve_lead(self, runtime: _Runtime | None = None, lead_hours: float | None = None) -> int:
        """Forecast-step index for ``lead_hours`` on the cached/default runtime."""
        runtime = runtime or self.ensure_runtime()
        return self._resolve_lead(runtime, lead_hours)

    def risk_fields(
        self,
        runtime: _Runtime | None = None,
        *,
        lead_hours: float | None = None,
        prediction: dict[str, np.ndarray] | None = None,
        uncertainty: dict[str, np.ndarray] | None = None,
    ):
        """Terrain-aware :class:`~app.services.risk_engine.RiskResult` for one step."""
        runtime = runtime or self.ensure_runtime()
        prediction = prediction or self._deterministic(runtime)
        step = self._resolve_lead(runtime, lead_hours)
        return self._risk(runtime, prediction, step, uncertainty=uncertainty)

    def active_checkpoint_info(self) -> dict[str, Any]:
        """Identity and provenance of the checkpoint backing this service.

        Reports whether a *trained* checkpoint was loaded, the model version, the
        training provenance if present, and the calibration status. Never claims
        observational validation: ``is_synthetic`` and ``validation_status`` are
        carried straight through from the artefact.
        """
        runtime = self.ensure_runtime()
        directory = self.settings.model_dir_path / runtime.model_version
        checkpoint: Path | None = directory / "nowcast_model.pt"
        meta: dict[str, Any] = {}
        if checkpoint.exists():
            try:
                meta = json.loads((directory / "training_meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        calibration: dict[str, Any] = {}
        calibration_path = directory / "calibration.json"
        if calibration_path.exists():
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                calibration = {}
        return {
            "model_version": runtime.model_version,
            "backend": runtime.backend,
            "checkpoint_path": str(checkpoint) if checkpoint.exists() else None,
            "checkpoint_present": checkpoint.exists(),
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.exists() else None,
            "trained_checkpoint_loaded": bool(
                runtime.describe.get("trained_checkpoint_loaded")
            ),
            "training_provenance": meta or None,
            "training_is_synthetic": meta.get("is_synthetic") if meta else None,
            "training_validation_status": meta.get("validation_status") if meta else None,
            "calibration": calibration or None,
            "calibration_present": bool(calibration),
            "is_synthetic": self.demo_mode,
            "observational_validation": False,
            "validation_status": meta.get("validation_status") if meta else (
                "no trained checkpoint loaded; synthetic demo weights"
            ),
        }

    # ---------------------------------------------------------------- routes
    def forecast(
        self,
        *,
        event_id: str | None = None,
        lead_hours: float | None = None,
        include_uncertainty: bool = False,
        mc_samples: int | None = None,
    ) -> dict[str, Any]:
        """Multi-hazard forecast summary for the synthetic demo event."""
        runtime = self.ensure_runtime(event_id)
        prediction = self._deterministic(runtime)
        step = self._resolve_lead(runtime, lead_hours)

        uncertainty_payload: dict[str, Any] | None = None
        risk_uncertainty: dict[str, np.ndarray] | None = None
        if include_uncertainty:
            samples = int(mc_samples or self._settings.mc_samples)
            ensemble = self._ensemble(runtime, samples)
            if ensemble is None:
                uncertainty_payload = {
                    "method": "unavailable",
                    "n_samples": int(samples),
                    "note": "active model backend does not implement MC sampling",
                }
            else:
                risk_uncertainty = {
                    hazard: np.asarray(ensemble[hazard])[:, step].std(axis=0)
                    for hazard in SUPPORTED_HAZARDS
                }
                uncertainty_payload = {
                    "method": "MC-dropout + variational bottleneck",
                    "n_samples": int(samples),
                    "spread_at_selected_step": {
                        hazard: {
                            "mean_std": round(float(risk_uncertainty[hazard].mean()), 6),
                            "max_std": round(float(risk_uncertainty[hazard].max()), 6),
                        }
                        for hazard in SUPPORTED_HAZARDS
                    },
                    "interval_90_at_selected_step": {
                        hazard: {
                            "lower": round(
                                float(np.percentile(np.asarray(ensemble[hazard])[:, step], 5)), 6
                            ),
                            "upper": round(
                                float(np.percentile(np.asarray(ensemble[hazard])[:, step], 95)), 6
                            ),
                        }
                        for hazard in SUPPORTED_HAZARDS
                    },
                }

        result = self._risk(runtime, prediction, step, uncertainty=risk_uncertainty)
        per_step = [
            {
                "step": index,
                "lead_hours": lead,
                "valid_time": self._valid_time(runtime, index),
                **{
                    hazard: self._field_stats(prediction[hazard], index)
                    for hazard in SUPPORTED_HAZARDS
                },
            }
            for index, lead in enumerate(runtime.lead_hours)
        ]
        selected = per_step[step]
        return {
            "event_id": runtime.spec.event_id,
            "event_kind": runtime.spec.kind,
            "model_version": runtime.model_version,
            "init_time": runtime.init_time.isoformat(),
            "grid": runtime.grid.to_dict(),
            "lead_times_h": runtime.lead_hours,
            "selected": {
                "step": step,
                "lead_hours": runtime.lead_hours[step],
                "valid_time": selected["valid_time"],
            },
            "fields": {hazard: selected[hazard] for hazard in SUPPORTED_HAZARDS},
            "per_step": per_step,
            "risk": {
                "lead_hours": result.lead_hours,
                "summary": result.summary(),
                "thresholds": list(runtime.risk_engine.thresholds),
                "weights": list(runtime.risk_engine.weights.normalised()),
            },
            "uncertainty": uncertainty_payload,
            **self.attribution(),
        }

    def point_risk(
        self,
        *,
        lat: float,
        lon: float,
        event_id: str | None = None,
        lead_hours: float | None = None,
    ) -> dict[str, Any]:
        """Terrain-aware compound risk at one geographic point."""
        runtime = self.ensure_runtime(event_id)
        if not runtime.grid.contains(float(lat), float(lon)):
            raise ValueError(
                f"({lat}, {lon}) is outside the model AOI "
                f"[{runtime.grid.min_lon}, {runtime.grid.max_lon}] x "
                f"[{runtime.grid.min_lat}, {runtime.grid.max_lat}]"
            )
        prediction = self._deterministic(runtime)
        step = self._resolve_lead(runtime, lead_hours)
        result = self._risk(runtime, prediction, step)
        payload = runtime.risk_engine.point_risk(result, lat=float(lat), lon=float(lon))
        payload.update({"event_id": runtime.spec.event_id, "grid": runtime.grid.to_dict()})
        payload.update(self.attribution())
        return payload

    def risk_geojson(
        self,
        *,
        event_id: str | None = None,
        lead_hours: float | None = None,
        min_category: int = 1,
        risk_field: str = "overall",
        max_features: int = 2000,
    ) -> dict[str, Any]:
        """Vectorised risk polygons as a GeoJSON ``FeatureCollection``."""
        runtime = self.ensure_runtime(event_id)
        prediction = self._deterministic(runtime)
        step = self._resolve_lead(runtime, lead_hours)
        result = self._risk(runtime, prediction, step)
        collection = runtime.risk_engine.to_geojson(
            result,
            min_category=int(min_category),
            risk_field=str(risk_field),
            max_features=int(max_features),
        )
        metadata = dict(collection.get("metadata") or {})
        labelling = self.attribution()
        for feature in collection.get("features") or []:
            properties = feature.setdefault("properties", {})
            properties.update(labelling)
        metadata.update(
            {
                "event_id": runtime.spec.event_id,
                "selected_step": step,
                "lead_hours": runtime.lead_hours[step],
                **labelling,
            }
        )
        collection["metadata"] = metadata
        return collection

    def explain(
        self,
        *,
        hazard: str = "cloudburst",
        step: int | None = None,
        event_id: str | None = None,
        include_consistency: bool = True,
        include_what_if: bool = False,
        perturbations: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Grad-CAM attribution, plus optional consistency audit and what-if."""
        runtime = self.ensure_runtime(event_id)
        if hazard not in SUPPORTED_HAZARDS:
            raise ValueError(f"hazard must be one of {list(SUPPORTED_HAZARDS)}; got {hazard!r}")
        # The backend may emit fewer steps than configured; ``_deterministic`` is
        # what reconciles ``lead_hours`` with the real output width, and Grad-CAM
        # indexes the same tensor. Run it first so the step bounds below are the
        # steps that actually exist rather than the configured maximum.
        prediction = self._deterministic(runtime)
        resolved_step = runtime.n_steps - 1 if step is None else int(step)
        if not 0 <= resolved_step < runtime.n_steps:
            raise ValueError(f"step must be in [0, {runtime.n_steps - 1}]; got {step}")

        explainer = GradCAMExplainer(runtime.model)
        attribution = explainer.explain(
            runtime.x_model,
            runtime.terrain_tensor,
            hazard=hazard,
            step=resolved_step,
        )
        cam = np.asarray(attribution.gradcam_map)
        payload: dict[str, Any] = {
            "event_id": runtime.spec.event_id,
            "hazard": hazard,
            "step": resolved_step,
            "lead_hours": runtime.lead_hours[resolved_step],
            "attribution_result": attribution.to_dict(),
            "gradcam_map": {
                "shape": [int(v) for v in cam.shape],
                "min": round(float(cam.min()), 6),
                "max": round(float(cam.max()), 6),
                "mean": round(float(cam.mean()), 6),
            },
            "model_version": runtime.model_version,
            **self.attribution(),
        }
        if include_consistency:
            prediction = self._deterministic(runtime)
            checker = PhysicalConsistencyChecker()
            consistency = checker.check(
                prediction,
                runtime.x_physical,
                terrain=runtime.terrain,
                attributions=attribution,
            )
            payload["physical_consistency"] = consistency.to_dict()
        if include_what_if:
            simulator = WhatIfSimulator(runtime.model)
            payload["what_if"] = simulator.simulate(
                runtime.x_model,
                runtime.terrain_tensor,
                perturbations=dict(perturbations or {"iwv": 0.1, "ctt": -0.1}),
            )
        return payload


_SERVICE: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Process-wide :class:`InferenceService` singleton (FastAPI dependency)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = InferenceService()
    return _SERVICE


def reset_inference_service() -> None:
    """Drop the singleton (used by tests)."""
    global _SERVICE
    _SERVICE = None
