"""Spatiotemporal Transformer alternative for >6 h lead times (Part 2.5).

Three-dimensional (space + time) ViT-style encoder: the input window is patched
in space and time, embedded with a linear projection plus learned positional
embeddings, passed through standard multi-head self-attention blocks, and decoded
back to the same per-lead-time feature maps as the ConvLSTM backbone so that the
identical multi-task heads can be attached. This makes an apples-to-apples CRPS
comparison possible (see ``training/baselines.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

from app.models.blocks import ConvNormAct, count_parameters
from app.physics import N_CHANNELS

__all__ = ["TransformerConfig", "SpatioTemporalTransformer"]


@dataclass(slots=True)
class TransformerConfig:
    """Hyper-parameters of the spatiotemporal transformer backbone."""

    in_channels: int = N_CHANNELS
    patch_size: int = 4
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.10
    out_channels: int = 16

    def to_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "out_channels": self.out_channels,
        }


class SpatioTemporalTransformer(nn.Module):
    """Patch-based transformer that produces the same interface as the ConvLSTM backbone.

    The module exposes :meth:`forward` returning an object with ``features``
    ``(B, T_out, out_channels, H, W)`` so the multi-task heads are shared between
    the recurrent and attention variants.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        cfg = self.config
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size)
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, cfg.embed_dim))
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.embed_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
                    dropout=cfg.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.to_maps = nn.Sequential(
            nn.ConvTranspose2d(cfg.embed_dim, cfg.out_channels, cfg.patch_size, stride=cfg.patch_size),
            ConvNormAct(cfg.out_channels, cfg.out_channels, 3),
        )
        self.n_parameters = count_parameters(self)

    def forward(self, x: Tensor, terrain: Tensor | None = None) -> Tensor:
        """``(B, T_in, C, H, W) -> (B, T_in, out_channels, H, W)``.

        ``terrain`` is accepted for interface compatibility with the recurrent
        backbone but is not fused here (the flood head still receives it).
        """
        if x.dim() != 5:
            raise ValueError(f"transformer expects (B, T, C, H, W), got {tuple(x.shape)}")
        batch, steps = x.shape[0], x.shape[1]
        flat = x.reshape(batch * steps, *x.shape[2:])
        patch_tokens = self.patch_embed(flat)                       # (B*T, E, h', w')
        _, embed_dim, height, width = patch_tokens.shape
        tokens = patch_tokens.flatten(2).transpose(1, 2).reshape(batch, steps * height * width, embed_dim)
        # Learned temporal positional embedding (cycled when T > table length).
        temporal = self.temporal_embed[:, torch.arange(steps) % self.temporal_embed.shape[1]]
        tokens = tokens.reshape(batch, steps, height * width, embed_dim) + temporal.unsqueeze(2)
        tokens = tokens.reshape(batch, steps * height * width, embed_dim)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        maps = tokens.reshape(batch * steps, height, width, embed_dim).permute(0, 3, 1, 2)
        decoded = self.to_maps(maps)
        if decoded.shape[-2:] != x.shape[-2:]:
            decoded = nn.functional.interpolate(decoded, size=x.shape[-2:], mode="nearest")
        return decoded.reshape(batch, steps, *decoded.shape[1:])

    def export_config(self) -> dict:
        """JSON-serialisable configuration (stored with the checkpoint)."""
        return self.config.to_dict() | {"n_parameters": self.n_parameters}
