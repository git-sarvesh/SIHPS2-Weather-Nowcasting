"""Pure-NumPy reference nowcaster (fallback backend, no PyTorch required).

Why this exists: the API, risk engine, XAI layer and dashboard must run on an
interpreter where no PyTorch wheel is available. This module provides physically
motivated per-cell predictors (cloud-top temperature and its trend, water-vapour
anomaly, column moisture, CAPE, split-window difference, terrain exposure) feeding
a small multi-layer perceptron with a shared trunk and three heads, trained with
an Adam optimiser implemented in NumPy. MC-dropout style sampling (Bernoulli
masks) exposes the same uncertainty interface as the PyTorch model.

It is deliberately a **reference implementation**: when PyTorch is available (the
supported configuration, used by the Docker stack) the ConvLSTM + Cross-Hazard
Attention model is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.logging_conf import get_logger
from app.nputils import sigmoid, softmax
from app.physics import channel_index

logger = get_logger("models.reference")

#: Per-cell features consumed by the NumPy MLP (order matters for checkpoints).
FEATURE_NAMES: tuple[str, ...] = (
    "ctt_now",
    "ctt_min",
    "ctt_change",
    "wv_anomaly",
    "iwv",
    "cape",
    "mir_bt",
    "vis_refl",
    "split_window_diff",
    "elevation",
    "slope",
    "flow_accumulation",
)

__all__ = ["NumpyReferenceNowcaster", "ReferenceConfig", "FEATURE_NAMES"]


@dataclass(slots=True)
class ReferenceConfig:
    """Architecture and feature normalisation of the reference MLP."""

    hidden: int = 48
    rain_classes: int = 4
    dropout: float = 0.10
    seed: int = 1234
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None

    def ensure_stats(self) -> np.ndarray:
        """Return (and lazily initialise) the feature standardisation vectors."""
        if self.feature_mean is None or self.feature_std is None:
            return np.zeros(len(FEATURE_NAMES))
        return self.feature_mean


def _init(rng: np.random.RandomState, fan_in: int, fan_out: int) -> tuple[np.ndarray, np.ndarray]:
    """He-style dense-layer initialisation."""
    return rng.randn(fan_out, fan_in) * np.sqrt(2.0 / max(fan_in, 1)), np.zeros(fan_out)


class NumpyReferenceNowcaster:
    """Feature-based MLP nowcaster implemented with NumPy only."""

    backend = "numpy"

    def __init__(
        self, config: ReferenceConfig | None = None, *, version: str = "sihps-numpy-reference-v0.1.0"
    ) -> None:
        self.config = config or ReferenceConfig()
        self.model_version = version
        rng = np.random.RandomState(self.config.seed)
        hidden = self.config.hidden
        n_features = len(FEATURE_NAMES)
        self.params: dict[str, np.ndarray] = {}
        self.params["W1"], self.params["b1"] = _init(rng, n_features, hidden)
        self.params["W2"], self.params["b2"] = _init(rng, hidden, hidden)
        self.params["Wts"], self.params["bts"] = _init(rng, hidden, 1)
        self.params["Wcb"], self.params["bcb"] = _init(rng, hidden, self.config.rain_classes)
        self.params["Wfl"], self.params["bfl"] = _init(rng, hidden, 1)
        if self.config.feature_mean is None:
            self.config.feature_mean = np.zeros(n_features)
        if self.config.feature_std is None:
            self.config.feature_std = np.ones(n_features)
        self.trained = False

    # ------------------------------------------------------------- features
    @staticmethod
    def features(channels: np.ndarray, terrain: np.ndarray | None = None, *, step: int = -1) -> np.ndarray:
        """Build a ``(H*W, F)`` per-cell feature matrix from a ``(T, C, H, W)`` window.

        Parameters
        ----------
        channels:
            Physical-unit input window (``T`` frames, 12 channels).
        terrain:
            Optional ``(4, H, W)`` terrain tensor (elevation, slope, flow, land use).
        step:
            Index of the "current" frame (default: the last frame).
        """
        arr = np.asarray(channels, dtype=np.float64)
        if arr.ndim != 4:
            raise ValueError(f"expected (T, C, H, W), got {arr.shape}")
        ctt = arr[:, channel_index("ctt")]
        tir1 = arr[:, channel_index("tir1_bt")]
        tir2 = arr[:, channel_index("tir2_bt")]
        wv = arr[:, channel_index("wv_bt")]
        index = min(max(step, 0), arr.shape[0] - 1) if step >= 0 else arr.shape[0] - 1
        columns = [
            ctt[index],
            ctt[: index + 1].min(axis=0),
            ctt[index] - ctt[0],
            wv[index] - wv[: index + 1].mean(axis=0),
            arr[index, channel_index("iwv")],
            arr[index, channel_index("cape")],
            arr[index, channel_index("mir_bt")],
            arr[index, channel_index("vis_refl")],
            tir1[index] - tir2[index],
        ]
        if terrain is None:
            columns.extend([np.zeros_like(ctt[index])] * 3)
        else:
            terrain_arr = np.asarray(terrain, dtype=np.float64)
            columns.extend([terrain_arr[0], terrain_arr[1], terrain_arr[2]])
        stacked = np.stack([np.asarray(col, dtype=np.float64) for col in columns], axis=-1)
        return stacked.reshape(-1, stacked.shape[-1])

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Standardise features with the stored training statistics."""
        mean = self.config.feature_mean if self.config.feature_mean is not None else 0.0
        std = self.config.feature_std if self.config.feature_std is not None else 1.0
        return (x - np.asarray(mean)[None, :]) / np.maximum(np.asarray(std)[None, :], 1e-6)

    # -------------------------------------------------------------- forward
    def _forward(
        self, x: np.ndarray, *, dropout_masks: bool = False, rng: np.random.RandomState | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shared trunk plus the three heads (optional Bernoulli dropout)."""
        rng = rng or np.random.RandomState(0)
        scale = 1.0 / max(1e-6, 1.0 - self.config.dropout)
        z1 = self.params["W1"] @ x.T + self.params["b1"][:, None]
        h1 = np.maximum(z1, 0.0)
        if dropout_masks:
            h1 = h1 * (rng.rand(*h1.shape) > self.config.dropout) * scale
        z2 = self.params["W2"] @ h1 + self.params["b2"][:, None]
        h2 = np.maximum(z2, 0.0)
        if dropout_masks:
            h2 = h2 * (rng.rand(*h2.shape) > self.config.dropout) * scale
        ts = sigmoid(self.params["Wts"] @ h2 + self.params["bts"][:, None])[0]
        rain = softmax((self.params["Wcb"] @ h2 + self.params["bcb"][:, None]).T, axis=-1)
        flood = sigmoid(self.params["Wfl"] @ h2 + self.params["bfl"][:, None])[0]
        return ts, rain, flood

    def _predict_step(
        self, channels: np.ndarray, terrain: np.ndarray | None, step: int, *, dropout: bool, rng
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict one lead step; returns ``(ts, rain, flood)`` on the raster grid."""
        arr = np.asarray(channels, dtype=np.float64)
        height, width = arr.shape[-2:]
        features = self.normalise(self.features(arr, terrain, step=step))
        ts, rain, flood = self._forward(features, dropout_masks=dropout, rng=rng)
        return (
            ts.reshape(height, width),
            rain.reshape(height, width, -1).transpose(2, 0, 1),
            flood.reshape(height, width),
        )

    def deterministic_forward(
        self, channels: np.ndarray, terrain: np.ndarray | None = None, *, n_steps: int = 12
    ) -> dict[str, np.ndarray]:
        """Hazard fields for each lead time (same contract as the PyTorch model)."""
        n_frames = np.asarray(channels).shape[0]
        steps = min(int(n_steps), max(1, n_frames))
        ts_out, rain_out, flood_out = [], [], []
        for step in range(steps):
            ts, rain, flood = self._predict_step(channels, terrain, step, dropout=False, rng=None)
            ts_out.append(ts)
            rain_out.append(rain)
            flood_out.append(flood)
        rain_stack = np.stack(rain_out)
        zeros = np.zeros((steps, *np.asarray(channels).shape[-2:]), dtype=np.float32)
        return {
            "thunderstorm": np.stack(ts_out),
            "cloudburst": rain_stack[:, -1],
            "flood": np.stack(flood_out),
            "rain_prob": rain_stack,
            "attention_thunderstorm_to_cloudburst": zeros,
            "attention_cloudburst_to_flood": zeros,
        }

    def mc_forward(
        self,
        channels: np.ndarray,
        terrain: np.ndarray | None = None,
        *,
        n_samples: int = 20,
        n_steps: int = 12,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """MC-dropout sampling with the same return contract as the PyTorch model."""
        rng = np.random.RandomState(seed)
        n_frames = np.asarray(channels).shape[0]
        steps = min(int(n_steps), max(1, n_frames))
        collected: dict[str, list[np.ndarray]] = {
            "thunderstorm": [],
            "cloudburst": [],
            "flood": [],
            "rain_prob": [],
        }
        for _ in range(max(1, int(n_samples))):
            ts_run, rain_run, flood_run = [], [], []
            for step in range(steps):
                ts, rain, flood = self._predict_step(channels, terrain, step, dropout=True, rng=rng)
                ts_run.append(ts)
                rain_run.append(rain)
                flood_run.append(flood)
            rain_stack = np.stack(rain_run)
            collected["thunderstorm"].append(np.stack(ts_run))
            collected["cloudburst"].append(rain_stack[:, -1])
            collected["flood"].append(np.stack(flood_run))
            collected["rain_prob"].append(rain_stack)
        result = {key: np.stack(values) for key, values in collected.items()}
        zeros = np.zeros((1, steps, *np.asarray(channels).shape[-2:]), dtype=np.float32)
        result["attention_thunderstorm_to_cloudburst"] = zeros
        result["attention_cloudburst_to_flood"] = zeros
        return result

    # ------------------------------------------------------------ training
    def fit(
        self,
        features: np.ndarray,
        targets: dict[str, np.ndarray],
        *,
        epochs: int = 60,
        batch_size: int = 4096,
        lr: float = 3e-3,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train the heads with Adam (NumPy implementation).

        ``targets`` must contain ``thunderstorm`` ``(N,)`` binary labels,
        ``rain_class`` ``(N,)`` integer classes and ``flood`` ``(N,)`` binary labels
        aligned with the rows of ``features``. Class imbalance is compensated with
        inverse-frequency weights, mirroring the focal / weighted-CE losses of the
        PyTorch path.
        """
        x_raw = np.asarray(features, dtype=np.float64)
        self.config.feature_mean = x_raw.mean(axis=0)
        self.config.feature_std = np.maximum(x_raw.std(axis=0), 1e-6)
        x = self.normalise(x_raw)
        y_ts = np.asarray(targets["thunderstorm"], dtype=np.float64)
        y_rain = np.asarray(targets["rain_class"], dtype=np.int64)
        y_flood = np.asarray(targets["flood"], dtype=np.float64)
        n = x.shape[0]
        rng = np.random.RandomState(self.config.seed)
        moments = {key: (np.zeros_like(value), np.zeros_like(value)) for key, value in self.params.items()}
        history: dict[str, float] = {}
        for epoch in range(int(epochs)):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, int(batch_size)):
                idx = order[start : start + int(batch_size)]
                loss, grads = self._loss_and_grads(x[idx], y_ts[idx], y_rain[idx], y_flood[idx])
                total += loss * idx.size
                for key, grad in grads.items():
                    m, v = moments[key]
                    m = m * 0.9 + 0.1 * grad
                    v = v * 0.999 + 0.001 * grad**2
                    m_hat = m / (1.0 - 0.9 ** (epoch + 1))
                    v_hat = v / (1.0 - 0.999 ** (epoch + 1))
                    self.params[key] = self.params[key] - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    moments[key] = (m, v)
            history[f"epoch_{epoch}"] = total / max(n, 1)
            if verbose and epoch % 10 == 0:
                logger.info("reference training", extra={"epoch": epoch, "loss": total / max(n, 1)})
        self.trained = True
        return history

    def _loss_and_grads(
        self, x: np.ndarray, y_ts: np.ndarray, y_rain: np.ndarray, y_flood: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Weighted BCE + class-weighted CE + BCE, with manual backpropagation."""
        batch = x.shape[0]
        z1 = self.params["W1"] @ x.T + self.params["b1"][:, None]
        h1 = np.maximum(z1, 0.0)
        z2 = self.params["W2"] @ h1 + self.params["b2"][:, None]
        h2 = np.maximum(z2, 0.0)
        ts_logit = self.params["Wts"] @ h2 + self.params["bts"][:, None]
        rain_logit = self.params["Wcb"] @ h2 + self.params["bcb"][:, None]
        flood_logit = self.params["Wfl"] @ h2 + self.params["bfl"][:, None]

        ts_prob = sigmoid(ts_logit)[0]
        rain_prob = softmax(rain_logit.T, axis=-1)
        flood_prob = sigmoid(flood_logit)[0]

        pos_ts = float(np.clip(y_ts.mean(), 1e-3, 1 - 1e-3))
        pos_flood = float(np.clip(y_flood.mean(), 1e-3, 1 - 1e-3))
        w_ts = (1.0 - pos_ts) / pos_ts
        w_flood = (1.0 - pos_flood) / pos_flood

        d_ts = (ts_prob - y_ts) * batch
        d_ts = np.where(y_ts > 0.5, d_ts * w_ts, d_ts)
        counts = np.bincount(y_rain, minlength=self.config.rain_classes).astype(np.float64)
        class_weights = counts.sum() / np.maximum(counts, 1.0)
        sample_weights = class_weights[y_rain] / class_weights.mean()
        d_rain = (rain_prob - np.eye(self.config.rain_classes)[y_rain]).T
        d_rain = d_rain * sample_weights[None, :] * batch
        d_flood = (flood_prob - y_flood) * batch
        d_flood = np.where(y_flood > 0.5, d_flood * w_flood, d_flood)

        grads: dict[str, np.ndarray] = {
            "Wts": d_ts @ h2.T,
            "bts": d_ts.sum(axis=1),
            "Wcb": d_rain @ h2.T,
            "bcb": d_rain.sum(axis=1),
            "Wfl": d_flood @ h2.T,
            "bfl": d_flood.sum(axis=1),
        }
        d_h2 = self.params["Wts"].T @ d_ts + self.params["Wcb"].T @ d_rain + self.params["Wfl"].T @ d_flood
        d_z2 = d_h2 * (z2 > 0)
        grads["W2"] = d_z2 @ h1.T
        grads["b2"] = d_z2.sum(axis=1)
        d_h1 = self.params["W2"].T @ d_z2
        d_z1 = d_h1 * (z1 > 0)
        grads["W1"] = d_z1 @ x
        grads["b1"] = d_z1.sum(axis=1)
        for key in grads:
            grads[key] = grads[key] / batch

        eps = 1e-9
        bce_ts = -np.mean(y_ts * np.log(ts_prob + eps) + (1 - y_ts) * np.log(1 - ts_prob + eps))
        bce_flood = -np.mean(y_flood * np.log(flood_prob + eps) + (1 - y_flood) * np.log(1 - flood_prob + eps))
        ce = -np.mean(np.log(np.clip(rain_prob[np.arange(batch), y_rain], eps, None)))
        return float(bce_ts + bce_flood + ce), grads

    # ------------------------------------------------------------- plumbing
    def state_dict(self) -> dict[str, np.ndarray]:
        """Serialisable parameter dictionary."""
        payload = {key: value for key, value in self.params.items()}
        payload["feature_mean"] = np.asarray(self.config.feature_mean)
        payload["feature_std"] = np.asarray(self.config.feature_std)
        return payload

    def load_state_dict(self, payload: dict[str, np.ndarray]) -> None:
        """Restore parameters from :meth:`state_dict`."""
        for key in list(self.params):
            if key in payload:
                self.params[key] = np.asarray(payload[key], dtype=np.float64)
        if "feature_mean" in payload:
            self.config.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        if "feature_std" in payload:
            self.config.feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
        self.trained = True

    def describe(self) -> dict[str, Any]:
        """Model metadata (mirrors :meth:`MultiTaskNowcastNet.describe`)."""
        return {
            "model_version": self.model_version,
            "backend": self.backend,
            "n_parameters": int(sum(value.size for value in self.params.values())),
            "features": list(FEATURE_NAMES),
            "trained": self.trained,
        }

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"NumpyReferenceNowcaster[{self.model_version}] features={len(FEATURE_NAMES)} "
            f"hidden={self.config.hidden} trained={self.trained}"
        )
