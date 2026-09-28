"""Cross-Hazard Attention (CHA) - the innovation linking the three task heads.

Physical cascade modelled: convective storm formation -> extreme rainfall
(cloudburst) -> runoff -> flash flood. Each head's probability output becomes the
*query* for the next hazard's features, so the network learns where a developing
storm is most likely to produce extreme rainfall, and where that rainfall is most
likely to generate flooding. Attention maps are returned so that the XAI layer can
expose them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

from app.physics import N_CHANNELS, RAIN_CLASSES

__all__ = ["CrossHazardAttention", "MultiTaskHeads", "HeadOutputs"]

_GROUPS_DEFAULT = 4


def _norm(channels: int) -> nn.Module:
    return nn.GroupNorm(num_groups=max(1, min(8, channels // 4 or 1)), num_channels=channels)


class CrossHazardAttention(nn.Module):
    """Query-gated spatial attention between two hazard representations.

    Given source features ``F`` ``(B, T, C, H, W)`` and a probability map
    ``q`` ``(B, T, 1, H, W)`` from the upstream head, CHA computes

    ``A = softmax_hw( W_q q + W_k F )``,  ``h = sum_hw A * (W_v F)``,
    ``F' = F + gamma * h``

    where ``gamma`` is a learnable scalar initialised near zero so that training
    starts from the plain multi-task baseline. ``A`` is returned for XAI.
    """

    def __init__(self, channels: int, *, groups: int = _GROUPS_DEFAULT, attention_temperature: float = 1.0) -> None:
        super().__init__()
        self.channels = channels
        self.temperature = float(attention_temperature)
        self.query_proj = nn.Conv2d(1, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = _norm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features: Tensor, query_probability: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(refined_features, attention_map)``.

        ``attention_map`` has shape ``(B, T, H, W)`` and is normalised over space
        (it sums to one), so it can be rendered directly as an XAI heatmap.
        """
        if features.dim() != 5 or query_probability.dim() != 5:
            raise ValueError("CHA expects (B, T, C, H, W) features and (B, T, 1, H, W) queries")
        batch, steps, channels, height, width = features.shape
        flat = features.reshape(batch * steps, channels, height, width)
        query = query_probability.reshape(batch * steps, 1, height, width)

        energy = (self.query_proj(query) + self.key_proj(flat)) / max(self.temperature, 1e-3)
        attention = torch.softmax(energy.reshape(batch * steps, channels, -1), dim=-1)
        attended = self.value_proj(flat) * attention.reshape_as(flat)
        refined = self.norm(flat + self.gamma * self.out_proj(attended))

        spatial = attention.mean(dim=1).reshape(batch * steps, height, width)
        return (
            refined.reshape(batch, steps, channels, height, width),
            spatial.reshape(batch, steps, height, width),
        )
