"""Unified nowcasting network: backbone + multi-task heads + MC inference (Parts 2.1-2.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from app.models.backbone import BackboneConfig, CNNConvLSTMBackbone
from app.models.multitask import HeadOutputs, MultiTaskHeads
from app.models.uncertainty import enable_mc_dropout

__all__ = ["NowcastNetConfig", "MultiTaskNowcastNet", "TASK_NAMES"]

#: Task identifiers shared by the API, risk engine and XAI layer.
TASK_NAMES: tuple[str, str, str] = ("thunderstorm", "cloudburst", "flood")


@dataclass(slots=True)
class NowcastNetConfig:
    """Configuration of the complete multi-task nowcasting model."""

    backbone: BackboneConfig = field(default_factory=lambda: BackboneConfig.preset("full"))
    terrain_channels: int = 4
    head_hidden: int = 16
    use_cha: bool = True
    model_version: str = "sihps-convlstm-cha-v0.1.0"
    rain_classes: Sequence[str] = ("no_rain", "light", "heavy", "extreme")

    @classmethod
    def preset(cls, name: str = "full", **overrides: Any) -> "NowcastNetConfig":
        """Build a config from a named backbone preset (``full`` | ``lite``)."""
        config = cls(backbone=BackboneConfig.preset(name))
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def to_dict(self) -> dict[str, Any]:
        backbone = dataclasses.asdict(self.backbone)
        backbone["convlstm_channels"] = list(backbone["convlstm_channels"])
        return {
            "model_version": self.model_version,
            "terrain_channels": self.terrain_channels,
            "head_hidden": self.head_hidden,
            "use_cha": self.use_cha,
            "rain_classes": list(self.rain_classes),
            "backbone": backbone,
        }


class MultiTaskNowcastNet(nn.Module):
    """End-to-end model: normalised multi-channel tensor in, hazard fields out."""

    def __init__(self, config: NowcastNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NowcastNetConfig()
        self.backbone = CNNConvLSTMBackbone(self.config.backbone)
        self.heads = MultiTaskHeads(
            self.backbone.recurrent.out_channels,
            terrain_channels=self.config.terrain_channels,
            hidden=self.config.head_hidden,
            n_rain_classes=len(self.config.rain_classes),
            dropout=self.config.backbone.dropout,
            use_cha=self.config.use_cha,
        )
        self.model_version = self.config.model_version

    def forward(self, x: Tensor, terrain: Tensor | None = None) -> dict[str, Any]:
        """Full forward pass.

        Parameters
        ----------
        x:
            ``(B, T_in, C, H, W)`` normalised inputs (``T_in = 6``, ``C = 12``).
        terrain:
            Optional ``(B, 4, H, W)`` terrain tensor for the flood head.

        Returns
        -------
        dict with head outputs, backbone features (Grad-CAM targets) and the KL term.
        """
        backbone_out = self.backbone(x)
        heads_out: HeadOutputs = self.heads(backbone_out.features, terrain)
        return {
            "heads": heads_out,
            "backbone": backbone_out,
            "kl": backbone_out.kl,
            "features": backbone_out.features,
            "model_version": self.model_version,
        }

    # ------------------------------------------------------- MC inference
    @torch.no_grad()
    def mc_forward(
        self,
        x: Tensor,
        terrain: Tensor | None = None,
        *,
        n_samples: int = 20,
        batch_chunk: int = 4,
    ) -> dict[str, np.ndarray]:
        """MC-dropout + variational predictive distribution (Part 2.4).

        All stochastic passes run as a single batched forward (chunked along the
        batch axis) so the 20-sample ensemble stays fast on CPU. Dropout masks and
        latent samples are independent per batch element, hence per sample.

        Returns
        -------
        dict of ensembles with shapes ``(n_samples, T_out, H, W)`` plus
        ``rain_prob`` ``(n_samples, T_out, 4, H, W)`` and attention maps.
        """
        was_training = self.training
        self.eval()
        enable_mc_dropout(self, enabled=True)
        ensembles: dict[str, list[np.ndarray]] = {name: [] for name in TASK_NAMES}
        rain_chunks: list[np.ndarray] = []
        attention: dict[str, list[np.ndarray]] = {"thunderstorm_to_cloudburst": [], "cloudburst_to_flood": []}
        remaining = max(1, int(n_samples))
        try:
            while remaining > 0:
                chunk = min(int(batch_chunk), remaining)
                remaining -= chunk
                expanded_x = x[:1].expand(chunk, *x.shape[1:]).contiguous()
                expanded_terrain = None
                if terrain is not None:
                    expanded_terrain = terrain[:1].expand(chunk, *terrain.shape[1:]).contiguous()
                out = self.forward(expanded_x, expanded_terrain)
                heads: HeadOutputs = out["heads"]
                ensembles["thunderstorm"].append(heads.thunderstorm_prob.cpu().numpy())
                ensembles["cloudburst"].append(heads.cloudburst_prob.cpu().numpy())
                ensembles["flood"].append(heads.flood_prob.cpu().numpy())
                rain_chunks.append(heads.rain_prob.cpu().numpy())
                if heads.attention_thunderstorm_to_cloudburst is not None:
                    attention["thunderstorm_to_cloudburst"].append(
                        heads.attention_thunderstorm_to_cloudburst.cpu().numpy()
                    )
                    attention["cloudburst_to_flood"].append(
                        heads.attention_cloudburst_to_flood.cpu().numpy()
                    )
        finally:
            enable_mc_dropout(self, enabled=False)
            self.train(was_training)

        result: dict[str, np.ndarray] = {name: np.concatenate(chunks, axis=0) for name, chunks in ensembles.items()}
        result["rain_prob"] = np.concatenate(rain_chunks, axis=0)
        for key, chunks in attention.items():
            result[f"attention_{key}"] = np.concatenate(chunks, axis=0) if chunks else np.zeros((1,))
        return result

    @torch.no_grad()
    def deterministic_forward(self, x: Tensor, terrain: Tensor | None = None) -> dict[str, np.ndarray]:
        """Single deterministic pass (fast interactive path, no MC sampling)."""
        was_training = self.training
        self.eval()
        try:
            out = self.forward(x, terrain)
        finally:
            self.train(was_training)
        heads: HeadOutputs = out["heads"]
        zeros = np.zeros_like(heads.flood_prob.cpu().numpy())
        return {
            "thunderstorm": heads.thunderstorm_prob.cpu().numpy(),
            "cloudburst": heads.cloudburst_prob.cpu().numpy(),
            "flood": heads.flood_prob.cpu().numpy(),
            "rain_prob": heads.rain_prob.cpu().numpy(),
            "attention_thunderstorm_to_cloudburst": (
                heads.attention_thunderstorm_to_cloudburst.cpu().numpy()
                if heads.attention_thunderstorm_to_cloudburst is not None
                else zeros
            ),
            "attention_cloudburst_to_flood": (
                heads.attention_cloudburst_to_flood.cpu().numpy()
                if heads.attention_cloudburst_to_flood is not None
                else zeros
            ),
        }

    # -------------------------------------------------------- descriptions
    def describe(self) -> dict[str, Any]:
        """Model metadata for ``/health`` and prediction audit records."""
        return {
            "model_version": self.model_version,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
            "config": self.config.to_dict(),
        }

    def summary(self) -> str:
        """One-line human readable summary."""
        params = sum(p.numel() for p in self.parameters())
        return (
            f"MultiTaskNowcastNet[{self.model_version}] params={params / 1e3:.1f}k "
            f"encoder={self.config.backbone.encoder_width} "
            f"convlstm={list(self.config.backbone.convlstm_channels)} cha={self.config.use_cha}"
        )
