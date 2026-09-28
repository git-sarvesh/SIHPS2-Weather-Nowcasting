"""Multi-task prediction heads with Cross-Hazard Attention (Parts 2.2 & 2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from app.models.blocks import ConvNormAct
from app.models.cha import CrossHazardAttention
from app.physics import RAIN_CLASSES

__all__ = ["HeadOutputs", "MultiTaskHeads"]


@dataclass(slots=True)
class HeadOutputs:
    """Per-lead-time logits, probabilities and cross-hazard attention maps."""

    thunderstorm_logits: Tensor              # (B, T, 1, H, W)
    rain_logits: Tensor                      # (B, T, 4, H, W)
    flood_logits: Tensor                     # (B, T, 1, H, W)
    thunderstorm_prob: Tensor                # (B, T, H, W)
    rain_prob: Tensor                        # (B, T, 4, H, W)
    flood_prob: Tensor                       # (B, T, H, W)
    cloudburst_prob: Tensor                  # (B, T, H, W) = P(extreme rainfall)
    flood_features: Tensor                   # (B, T, C, H, W) - Grad-CAM target
    cloudburst_features: Tensor              # (B, T, C, H, W) - Grad-CAM target
    attention_thunderstorm_to_cloudburst: Tensor | None = None   # (B, T, H, W)
    attention_cloudburst_to_flood: Tensor | None = None          # (B, T, H, W)


class MultiTaskHeads(nn.Module):
    """Three heads on shared features, chained by Cross-Hazard Attention.

    Head 1 - thunderstorm probability (binary, focal loss).
    Head 2 - cloudburst / extreme rainfall probability (4 classes, weighted CE).
    Head 3 - flash-flood risk (shared features + terrain via late fusion).

    The cascade is explicit: the thunderstorm probability gates the cloudburst
    features, and P(extreme rainfall) gates the flood features.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        terrain_channels: int = 4,
        hidden: int = 16,
        n_rain_classes: int = len(RAIN_CLASSES),
        dropout: float = 0.10,
        use_cha: bool = True,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.terrain_channels = terrain_channels
        self.n_rain_classes = n_rain_classes
        self.hidden = hidden
        self.use_cha = use_cha

        self.thunderstorm_head = nn.Sequential(
            ConvNormAct(feature_channels, hidden, 3),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
        )
        self.cloudburst_head = nn.Sequential(
            ConvNormAct(feature_channels, hidden, 3),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, n_rain_classes, 3, padding=1, bias=True),
        )
        self.terrain_encoder = nn.Sequential(
            ConvNormAct(terrain_channels, hidden, 3),
            ConvNormAct(hidden, hidden, 3),
        )
        self.flood_head = nn.Sequential(
            ConvNormAct(feature_channels + hidden, hidden, 3),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
        )
        self.cha_thunderstorm_to_cloudburst = (
            CrossHazardAttention(feature_channels) if use_cha else None
        )
        self.cha_cloudburst_to_flood = CrossHazardAttention(feature_channels) if use_cha else None

    # ------------------------------------------------------------------ fwd
    def forward(self, features: Tensor, terrain: Tensor | None = None) -> HeadOutputs:
        """Apply the three heads (cascaded by CHA) to shared features.

        Parameters
        ----------
        features:
            ``(B, T, C, H, W)`` spatiotemporal features from the backbone.
        terrain:
            Optional ``(B, 4, H, W)`` terrain stack (elevation, slope, flow
            accumulation, land use) used by the flood head's late fusion.
        """
        if features.dim() != 5:
            raise ValueError(f"heads expect (B, T, C, H, W), got {tuple(features.shape)}")
        batch, steps, channels, height, width = features.shape
        flat_features = features.reshape(batch * steps, channels, height, width)

        thunderstorm_logits = self.thunderstorm_head(flat_features)
        thunderstorm_logits = thunderstorm_logits.reshape(batch, steps, 1, height, width)
        thunderstorm_prob = torch.sigmoid(thunderstorm_logits)

        if self.cha_thunderstorm_to_cloudburst is not None:
            cloudburst_features, attention_ts_cb = self.cha_thunderstorm_to_cloudburst(
                features, thunderstorm_prob.detach()
            )
        else:
            cloudburst_features = features
            attention_ts_cb = torch.zeros((batch, steps, height, width), device=features.device)

        rain_logits = self.cloudburst_head(
            cloudburst_features.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, self.n_rain_classes, height, width)
        rain_prob = torch.softmax(rain_logits, dim=2)
        cloudburst_prob = rain_prob[:, :, -1]  # P(extreme rainfall)

        if self.cha_cloudburst_to_flood is not None:
            flood_features, attention_cb_flood = self.cha_cloudburst_to_flood(
                cloudburst_features, cloudburst_prob.unsqueeze(2).detach()
            )
        else:
            flood_features = cloudburst_features
            attention_cb_flood = torch.zeros((batch, steps, height, width), device=features.device)

        if terrain is not None:
            encoded_terrain = self.terrain_encoder(terrain)               # (B, hidden, H, W)
            terrain_features = encoded_terrain.unsqueeze(1).expand(-1, steps, -1, -1, -1)
            terrain_features = terrain_features.reshape(batch * steps, self.hidden, height, width)
        else:
            terrain_features = torch.zeros(
                (batch * steps, self.hidden, height, width), device=features.device, dtype=features.dtype
            )

        fused = torch.cat(
            [flood_features.reshape(batch * steps, channels, height, width), terrain_features], dim=1
        )
        flood_logits = self.flood_head(fused).reshape(batch, steps, 1, height, width)

        return HeadOutputs(
            thunderstorm_logits=thunderstorm_logits,
            rain_logits=rain_logits,
            flood_logits=flood_logits,
            thunderstorm_prob=thunderstorm_prob.reshape(batch, steps, height, width),
            rain_prob=rain_prob,
            flood_prob=torch.sigmoid(flood_logits).reshape(batch, steps, height, width),
            cloudburst_prob=cloudburst_prob,
            flood_features=flood_features,
            cloudburst_features=cloudburst_features,
            attention_thunderstorm_to_cloudburst=attention_ts_cb,
            attention_cloudburst_to_flood=attention_cb_flood,
        )
