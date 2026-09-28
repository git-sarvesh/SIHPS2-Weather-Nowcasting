"""Multi-task loss for the thunderstorm / cloudburst / flood nowcaster.

Design follows the head definitions in :mod:`app.models.multitask`:

* **Head 1 - thunderstorm.** Rare binary event (deep convection covers a small
  fraction of cells), so a *focal* binary cross-entropy (Lin et al. 2017) with a
  configurable ``gamma`` down-weights easy negatives.
* **Head 2 - rainfall class / cloudburst.** Four-class cross-entropy with class
  weights that counteract the strong ``no_rain`` imbalance, plus an auxiliary
  binary term on ``P(extreme)`` because that is the field the risk engine
  consumes.
* **Head 3 - flood.** Binary cross-entropy on the hard flood event *and* a soft
  term on ``flood_soft``, which is dense and therefore gives the head a usable
  gradient where the binary event is absent.
* **Variational regulariser.** The backbone KL term is added with a small
  weight so ensemble spread stays meaningful without dominating the task loss.

Lead-time weighting is supported because skill decays with lead time; by
default every lead time contributes equally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["LossWeights", "MultiTaskLoss", "class_balanced_weights", "focal_bce_loss"]


def _align_to_logits(tensor: Tensor, reference: Tensor) -> Tensor:
    """Broadcast a target onto the head-logit layout.

    The binary heads emit ``(B, T, 1, H, W)`` logits (see
    :class:`~app.models.multitask.HeadOutputs`) while the dataset yields
    ``(B, T, H, W)`` labels, so the target is unsqueezed when a singleton channel
    axis is present.
    """
    if tensor.shape == reference.shape:
        return tensor
    if tensor.dim() == reference.dim() - 1 and tensor.shape == reference[:, :, 0].shape:
        return tensor.unsqueeze(2)
    return tensor


def focal_bce_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.5,
    weight: Tensor | None = None,
) -> Tensor:
    """Focal binary cross-entropy, accepting either target layout.

    Targets may be ``(B, T, H, W)`` or ``(B, T, 1, H, W)`` to match the head
    logits. ``gamma = 0`` reduces to ordinary weighted BCE, which keeps the
    behaviour testable against a reference implementation.
    """
    targets = _align_to_logits(targets, logits)
    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1).to(flat_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_targets, reduction="none")
    probs = torch.sigmoid(flat_logits)
    p_t = probs * flat_targets + (1.0 - probs) * (1.0 - flat_targets)
    modulating = (1.0 - p_t).clamp_min(1e-8).pow(float(gamma))
    if alpha is not None:
        modulating = modulating * (alpha * flat_targets + (1.0 - alpha) * (1.0 - flat_targets))
    if weight is not None:
        modulating = modulating * weight.reshape(-1).to(flat_logits.dtype)
    return (bce * modulating).mean()


def class_balanced_weights(counts: Tensor, *, clip: float = 20.0) -> Tensor:
    """Inverse-frequency class weights from a ``(n_classes,)`` count vector.

    Zero-count classes get the maximum weight (so they are not silently ignored)
    and the weights are normalised to mean 1 to keep the loss scale stable.
    """
    counts = counts.to(torch.float64).clamp_min(0.0)
    total = float(counts.sum())
    if total <= 0:
        return torch.ones_like(counts, dtype=torch.float32)
    weights = total / (counts * float(len(counts)))
    weights = torch.where(counts > 0, weights, torch.full_like(weights, float(clip)))
    weights = (weights / float(clip)).clamp(1.0 / float(clip), 1.0)
    weights = weights / float(weights.mean())
    return weights.to(torch.float32)


@dataclass(slots=True)
class LossWeights:
    """Relative weights of the loss terms (all default to 1.0)."""

    thunderstorm: float = 1.0
    rain_class: float = 1.0
    cloudburst: float = 0.5
    flood: float = 1.0
    flood_soft: float = 0.5
    kl: float = 1e-3

    def to_dict(self) -> dict[str, float]:
        return {
            "thunderstorm": self.thunderstorm,
            "rain_class": self.rain_class,
            "cloudburst": self.cloudburst,
            "flood": self.flood,
            "flood_soft": self.flood_soft,
            "kl": self.kl,
        }


@dataclass(slots=True)
class MultiTaskLoss:
    """Combined loss over the three heads plus the variational KL term.

    Parameters
    ----------
    weights:
        Term weights (see :class:`LossWeights`).
    focal_gamma:
        Focusing parameter for the thunderstorm and cloudburst heads.
    positive_weight:
        ``pos_weight`` for the plain-BCE flood term, useful for rare events.
    rain_class_weights:
        Optional per-class weights for the rainfall head.
    lead_weights:
        Optional ``(T_out,)`` relative lead-time weights (applied to the
        element-wise terms; the class-balanced and BCE means are re-normalised).
    """

    weights: LossWeights = field(default_factory=LossWeights)
    focal_gamma: float = 2.0
    positive_weight: float = 1.0
    rain_class_weights: Tensor | None = None
    lead_weights: Tensor | None = None

    # ------------------------------------------------------------------ terms
    def thunderstorm_term(self, logits: Tensor, targets: Tensor) -> Tensor:
        return float(self.weights.thunderstorm) * focal_bce_loss(
            logits, targets, gamma=self.focal_gamma
        )

    def rain_term(self, rain_logits: Tensor, rain_class: Tensor) -> Tensor:
        """Weighted cross-entropy on the 4-class rainfall head."""
        logits = rain_logits.permute(0, 1, 3, 4, 2)  # (B, T, H, W, K)
        target = rain_class.long().unsqueeze(-1)
        weight = None
        if self.rain_class_weights is not None:
            weight = self.rain_class_weights.to(logits.device).reshape(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            weight=weight,
            reduction="mean",
        )
        return float(self.weights.rain_class) * loss

    def cloudburst_term(self, rain_logits: Tensor, cloudburst: Tensor) -> Tensor:
        """Binary term on ``P(extreme)`` taken from the rainfall logits."""
        extreme_logits = rain_logits[:, :, -1]  # (B, T, H, W)
        return float(self.weights.cloudburst) * focal_bce_loss(
            extreme_logits, cloudburst, gamma=self.focal_gamma
        )

    def flood_term(self, logits: Tensor, flood: Tensor, flood_soft: Tensor) -> Tensor:
        flood = _align_to_logits(flood, logits)
        flood_soft = _align_to_logits(flood_soft, logits)
        hard = F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            flood.reshape(-1).to(logits.dtype),
            reduction="mean",
            pos_weight=torch.tensor(float(self.positive_weight), device=logits.device),
        )
        soft = F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            flood_soft.reshape(-1).to(logits.dtype).clamp(0.0, 1.0),
            reduction="mean",
        )
        return float(self.weights.flood) * hard + float(self.weights.flood_soft) * soft

    def kl_term(self, kl: Tensor | None) -> Tensor:
        if kl is None:
            return torch.zeros((), dtype=torch.float32)
        return float(self.weights.kl) * kl.mean()

    def __call__(
        self, outputs: dict[str, Any], targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Return ``(total_loss, per_term_dict)``.

        ``targets`` must provide ``thunderstorm``, ``rain_class``, ``cloudburst``
        and ``flood``; ``flood_soft`` is optional (defaults to zeros).
        """
        heads = outputs["heads"]
        terms: dict[str, Tensor] = {
            "thunderstorm": self.thunderstorm_term(
                heads.thunderstorm_logits, targets["thunderstorm"]
            ),
            "rain_class": self.rain_term(heads.rain_logits, targets["rain_class"]),
            "cloudburst": self.cloudburst_term(heads.rain_logits, targets["cloudburst"]),
            "flood": self.flood_term(
                heads.flood_logits,
                targets["flood"],
                targets.get("flood_soft", torch.zeros_like(targets["flood"])),
            ),
            "kl": self.kl_term(outputs.get("kl")),
        }
        total = sum(terms.values())
        return total, {name: float(value.detach()) for name, value in terms.items()}
