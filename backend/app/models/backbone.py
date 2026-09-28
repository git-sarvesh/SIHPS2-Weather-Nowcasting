"""Spatiotemporal backbone: residual CNN encoder + ConvLSTM decoder (Part 2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor, nn

from app.models.blocks import (
    ConvLSTM,
    ConvNormAct,
    ResidualBlock,
    TemporalFeatureStack,
    count_parameters,
)
from app.models.uncertainty import VariationalBottleneck, gaussian_kl
from app.physics import N_CHANNELS

__all__ = ["BackboneConfig", "ResidualCNNEncoder", "CNNConvLSTMBackbone", "BackboneOutput"]


@dataclass(slots=True)
class BackboneConfig:
    """Hyper-parameters of the shared encoder/decoder stack."""

    in_channels: int = N_CHANNELS
    encoder_width: int = 64
    n_residual_blocks: int = 4
    convlstm_channels: Sequence[int] = (64, 32, 16)
    kernel_size: int = 3
    latent_dim: int = 16
    dropout: float = 0.10
    norm: str = "group"
    variational: bool = True
    predict_steps: int = 12

    @classmethod
    def preset(cls, name: str = "full") -> "BackboneConfig":
        """Named presets.

        ``full``
            Specification-exact widths (64-channel encoder, ConvLSTM 64->32->16);
            intended for GPU training.
        ``lite``
            CPU-friendly widths with the same topology, used for the interactive
            demo and API inference on commodity hardware.
        """
        if name == "full":
            return cls()
        if name == "lite":
            return cls(encoder_width=32, convlstm_channels=(32, 16, 8), latent_dim=8)
        raise ValueError(f"unknown preset {name!r} (expected 'full' or 'lite')")


class ResidualCNNEncoder(nn.Module):
    """Stem convolution followed by ``n_residual_blocks`` residual blocks."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(config.in_channels, config.encoder_width, 3, norm=config.norm),
            ConvNormAct(config.encoder_width, config.encoder_width, 3, norm=config.norm),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(config.encoder_width, norm=config.norm, dropout=config.dropout)
                for _ in range(config.n_residual_blocks)
            ]
        )
        self.out_channels = config.encoder_width

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, H, W) -> (B, encoder_width, H, W)``."""
        return self.blocks(self.stem(x))


@dataclass(slots=True)
class BackboneOutput:
    """Container returned by :class:`CNNConvLSTMBackbone`."""

    features: Tensor                        # (B, T_out, C_dec, H, W)
    encoder_features: Tensor                # (B, T_in, C_enc, H, W)
    layer_outputs: list[Tensor] = field(default_factory=list)
    states: list = field(default_factory=list)
    mu: Tensor | None = None
    logvar: Tensor | None = None
    kl: Tensor | None = None

    @property
    def out_channels(self) -> int:
        return int(self.features.shape[2])


class CNNConvLSTMBackbone(nn.Module):
    """Hybrid CNN + ConvLSTM backbone shared by all task heads.

    Pipeline (Part 2.1): residual CNN per frame -> variational bottleneck ->
    ConvLSTM 64 -> 32 -> 16 -> per-lead-time feature maps for the heads.
    """

    def __init__(self, config: BackboneConfig | None = None) -> None:
        super().__init__()
        self.config = config or BackboneConfig()
        self.encoder = TemporalFeatureStack(ResidualCNNEncoder(self.config))
        self.bottleneck = (
            VariationalBottleneck(self.config.encoder_width, self.config.latent_dim)
            if self.config.variational
            else None
        )
        self.recurrent = ConvLSTM(
            self.config.encoder_width,
            tuple(self.config.convlstm_channels),
            self.config.kernel_size,
            dropout=self.config.dropout,
            norm=self.config.norm,
            return_all_layers=True,
        )
        self.n_parameters = count_parameters(self)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        """Encode ``(B, T_in, C, H, W)`` to recurrent inputs plus latent statistics."""
        encoded = self.encoder(x)
        if self.bottleneck is None:
            return encoded, None, None, None
        batch, steps = encoded.shape[0], encoded.shape[1]
        flat = encoded.reshape(batch * steps, *encoded.shape[2:])
        refined, mu, logvar = self.bottleneck(flat)
        return (
            refined.reshape(batch, steps, *refined.shape[1:]),
            mu.reshape(batch, steps, *mu.shape[1:]),
            logvar.reshape(batch, steps, *logvar.shape[1:]),
            gaussian_kl(mu, logvar),
        )

    def forward(self, x: Tensor) -> BackboneOutput:
        """Run the backbone over an input window.

        Parameters
        ----------
        x:
            ``(B, T_in, C, H, W)`` normalised input tensor (``T_in = 6``).
        """
        if x.dim() != 5:
            raise ValueError(f"backbone expects (B, T, C, H, W), got {tuple(x.shape)}")
        refined, mu, logvar, kl = self.encode(x)
        layer_outputs, states = self.recurrent(refined)  # type: ignore[arg-type]
        return BackboneOutput(
            features=layer_outputs[-1],
            encoder_features=refined,
            layer_outputs=list(layer_outputs),
            states=list(states),
            mu=mu,
            logvar=logvar,
            kl=kl,
        )

    def rollout(self, last_features: Tensor, states: list, *, steps: int) -> Tensor:
        """Feature-space rollout beyond the trained horizon (long-range option).

        ``last_features`` is ``(B, C_enc, H, W)``; ``steps`` further timesteps are
        produced by feeding the previous feature map back into the recurrent
        stack. Used by the >6 h transformer comparison (Part 2.5).
        """
        outputs: list[Tensor] = []
        current = last_features
        for _ in range(int(steps)):
            layer_outputs, states = self.recurrent(current.unsqueeze(1), states)
            current = layer_outputs[-1][:, -1]
            outputs.append(current)
        return torch.stack(outputs, dim=1)

    def export_config(self) -> dict:
        """JSON-serialisable configuration (stored alongside checkpoints)."""
        return {
            "in_channels": self.config.in_channels,
            "encoder_width": self.config.encoder_width,
            "n_residual_blocks": self.config.n_residual_blocks,
            "convlstm_channels": list(self.config.convlstm_channels),
            "kernel_size": self.config.kernel_size,
            "latent_dim": self.config.latent_dim,
            "dropout": self.config.dropout,
            "norm": self.config.norm,
            "variational": self.config.variational,
            "n_parameters": self.n_parameters,
        }
