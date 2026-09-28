"""Uncertainty modelling: variational bottleneck, MC dropout and calibration (Part 2.4).

Three complementary mechanisms:

1. :class:`VariationalBottleneck` - a learned Gaussian latent (reparameterisation
   trick) inside the shared encoder, so the network learns a distribution over
   its spatiotemporal features.
2. :func:`enable_mc_dropout` - keeps dropout active at inference time so repeated
   stochastic forward passes sample the predictive distribution.
3. :class:`TemperatureScaler` - post-hoc calibration fitted on the validation set.

Together they yield per-pixel mean, standard deviation, quantiles and a 90 %
confidence interval for every hazard, which the API and dashboard surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from app.nputils import logit as np_logit

__all__ = [
    "VariationalBottleneck",
    "enable_mc_dropout",
    "TemperatureScaler",
    "PredictionDistribution",
    "summarise_samples",
    "crps_ensemble",
    "gaussian_kl",
]


def gaussian_kl(mu: Tensor, logvar: Tensor, *, free_bits: float = 0.05) -> Tensor:
    """KL divergence to ``N(0, I)``, averaged over the batch, with a free-bits floor.

    ``free_bits`` prevents posterior collapse by not penalising latent dimensions
    whose KL is already below the floor (Kingma et al. 2016).
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    kl_per_dim = kl.mean(dim=(2, 3))
    return torch.clamp(kl_per_dim, min=free_bits).sum(dim=1).mean()


class VariationalBottleneck(nn.Module):
    """Gaussian latent bottleneck (reparameterisation trick) on a strided grid.

    The bottleneck runs at reduced resolution to keep the memory footprint low
    while still injecting uncertainty into the shared representation.
    """

    def __init__(self, channels: int, latent_dim: int = 16, *, stride: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.stride = stride
        self.pool = nn.AvgPool2d(stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity()
        self.to_params = nn.Conv2d(channels, 2 * latent_dim, 3, padding=1)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, channels // 4 or 1)), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(features_with_latent, mu, logvar)``."""
        pooled = self.pool(features)
        mu, logvar = torch.chunk(self.to_params(pooled), 2, dim=1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        decoded = self.from_latent(z)
        if decoded.shape[-2:] != features.shape[-2:]:
            decoded = nn.functional.interpolate(decoded, size=features.shape[-2:], mode="nearest")
        return features + decoded, mu, logvar


def enable_mc_dropout(module: nn.Module, *, enabled: bool = True) -> None:
    """Toggle training mode for dropout layers only (MC dropout at inference).

    Normalisation layers keep their inference behaviour; only ``Dropout*``
    modules are switched to training mode so stochastic passes differ.
    """
    for child in module.modules():
        if isinstance(child, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            child.train(enabled)


@dataclass(slots=True)
class PredictionDistribution:
    """Ensemble summary for one probabilistic field."""

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray          # 5 % quantile  -> 90 % CI lower bound
    upper: np.ndarray          # 95 % quantile -> 90 % CI upper bound
    n_samples: int

    @property
    def ci_width(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def confidence(self) -> np.ndarray:
        """Confidence proxy in ``[0, 1]`` derived from the 90 % interval width."""
        return np.clip(1.0 - np.clip(self.ci_width, 0.0, 1.0), 0.0, 1.0)

    def at(self, row: int, col: int) -> dict[str, float]:
        """Point summary used by ``/api/v1/risk/point``."""
        return {
            "mean": float(self.mean[row, col]),
            "std": float(self.std[row, col]),
            "ci90_lower": float(self.lower[row, col]),
            "ci90_upper": float(self.upper[row, col]),
            "n_samples": int(self.n_samples),
        }

    def to_dict(self, *, downsample: int = 1) -> dict[str, Any]:
        """Serialisable summary, optionally strided to limit payload size."""
        step = max(1, int(downsample))
        sl = (slice(None, None, step), slice(None, None, step))
        return {
            "mean": self.mean[sl].round(4).tolist(),
            "std": self.std[sl].round(4).tolist(),
            "ci90_lower": self.lower[sl].round(4).tolist(),
            "ci90_upper": self.upper[sl].round(4).tolist(),
            "n_samples": self.n_samples,
        }


def summarise_samples(samples: np.ndarray, *, confidence_level: float = 0.90) -> PredictionDistribution:
    """Reduce an ensemble ``(S, ...)`` into mean/std/interval statistics."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim < 1 or arr.shape[0] < 2:
        raise ValueError("need at least two samples along the first axis")
    lower_q = 100.0 * (1.0 - confidence_level) / 2.0
    upper_q = 100.0 - lower_q
    return PredictionDistribution(
        mean=arr.mean(axis=0),
        std=arr.std(axis=0),
        lower=np.percentile(arr, lower_q, axis=0),
        upper=np.percentile(arr, upper_q, axis=0),
        n_samples=int(arr.shape[0]),
    )


def crps_ensemble(samples: np.ndarray, observation: np.ndarray) -> float:
    """Continuous Ranked Probability Score (unbiased ensemble estimator).

    ``CRPS = mean|X - y| - 0.5 * mean|X - X'|`` where ``X, X'`` are independent
    ensemble members. Used to compare the ConvLSTM and transformer variants
    (Part 2.5) and to score probabilistic nowcasts (Part 8.3).
    """
    ens = np.asarray(samples, dtype=np.float64)
    obs = np.asarray(observation, dtype=np.float64).ravel()
    flat = ens.reshape(ens.shape[0], -1) if ens.ndim > 1 else ens[:, None]
    if flat.shape[1] != obs.size:
        raise ValueError("ensemble and observation have incompatible shapes")
    term1 = np.mean(np.abs(flat - obs[None, :]), axis=0)
    term2 = 0.5 * np.mean(np.abs(flat[:, None, :] - flat[None, :, :]), axis=(0, 1))
    return float(np.mean(term1 - term2))


class TemperatureScaler:
    """Post-hoc probability calibration (Guo et al. 2017).

    Logits are divided by a single temperature ``T > 0`` fitted by minimising the
    validation negative log-likelihood (1-D monotone search, no optimiser needed).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """Fit the temperature on validation probabilities and labels."""
        p = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), 1e-6, 1 - 1e-6)
        y = np.asarray(targets, dtype=np.float64).ravel()
        if p.size != y.size:
            raise ValueError("probabilities and targets must have the same size")
        logits = np_logit(p)
        best_t, best_nll = 1.0, np.inf
        for temperature in np.exp(np.linspace(np.log(0.25), np.log(8.0), 60)):
            scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
            nll = -np.mean(y * np.log(scaled + 1e-12) + (1 - y) * np.log(1 - scaled + 1e-12))
            if nll < best_nll:
                best_t, best_nll = float(temperature), float(nll)
        self.temperature = best_t
        return best_t

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature to a probability field."""
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-np_logit(p) / max(self.temperature, 1e-3)))

    def to_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemperatureScaler":
        return cls(float(payload.get("temperature", 1.0)))
