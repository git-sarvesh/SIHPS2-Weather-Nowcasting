"""Neural building blocks: residual CNN stacks and ConvLSTM (Part 2.1).

Shapes follow the specification: input tensors are ``(B, T, C, H, W)`` with
``T = seq_len`` (6 frames = 3 h) and ``C = 12`` channels; ConvLSTM layers are
stacked 64 -> 32 -> 16 by default.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "ResidualBlock",
    "ConvLSTMCell",
    "ConvLSTM",
    "ConvNormAct",
    "TemporalFeatureStack",
    "upsample_like",
    "count_parameters",
]


class TemporalFeatureStack(nn.Module):
    """Apply a CNN encoder frame-wise, keeping the spatial resolution intact.

    This is the standard ``CNN + ConvLSTM`` hybrid encoder (Part 2.1): the CNN
    extracts per-frame spatial features and the recurrent stack models their
    temporal evolution.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, C, H, W) -> (B, T, C_enc, H, W)``."""
        batch, steps = x.shape[0], x.shape[1]
        flat = x.reshape(batch * steps, *x.shape[2:])
        encoded = self.encoder(flat)
        return encoded.reshape(batch, steps, *encoded.shape[1:])


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM producing a sequence of feature maps per layer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int] = (64, 32, 16),
        kernel_size: int = 3,
        *,
        dropout: float = 0.0,
        norm: str = "group",
        return_all_layers: bool = True,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must be non-empty")
        self.return_all_layers = return_all_layers
        self.cells = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev = in_channels
        for hidden in hidden_channels:
            self.cells.append(ConvLSTMCell(prev, hidden, kernel_size, norm=norm))
            self.dropouts.append(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity())
            prev = hidden
        self.out_channels = int(hidden_channels[-1])

    def forward(self, x: Tensor, state: list[tuple[Tensor, Tensor]] | None = None):
        """Run the stack over ``x`` of shape ``(B, T, C, H, W)``.

        Returns
        -------
        layer_outputs:
            One ``(B, T, C_layer, H, W)`` tensor per layer.
        states:
            Final ``(h, c)`` per layer, enabling autoregressive rollout beyond
            the trained horizon.
        """
        if x.dim() != 5:
            raise ValueError(f"ConvLSTM expects (B, T, C, H, W), got {tuple(x.shape)}")
        steps = x.shape[1]
        states = state or [None] * len(self.cells)
        new_states: list[tuple[Tensor, Tensor]] = []
        layer_input = x
        layer_outputs: list[Tensor] = []
        for cell, drop, previous in zip(self.cells, self.dropouts, states):
            outputs = []
            current = previous
            for t in range(steps):
                current = cell(layer_input[:, t], current)
                outputs.append(drop(current[0]))
            layer_input = torch.stack(outputs, dim=1)
            layer_outputs.append(layer_input)
            new_states.append(current)  # type: ignore[arg-type]
        if self.return_all_layers:
            return layer_outputs, new_states
        return [layer_outputs[-1]], new_states


def upsample_like(x: Tensor, reference: Tensor, *, mode: str = "nearest") -> Tensor:
    """Interpolate ``x`` to the spatial size of ``reference``."""
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return nn.functional.interpolate(x, size=reference.shape[-2:], mode=mode)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def iter_named_modules(module: nn.Module, kinds: tuple[type, ...]) -> Iterable[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for modules matching ``kinds``."""
    for name, child in module.named_modules():
        if isinstance(child, kinds):
            yield name, child


def same_padding(kernel_size: int | Sequence[int]) -> int | tuple[int, int]:
    """Padding that preserves spatial size for odd kernel sizes."""
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(k // 2 for k in kernel_size)


def _norm(channels: int, kind: str) -> nn.Module:
    if kind == "group":
        return nn.GroupNorm(num_groups=max(1, min(8, channels // 4 or 1)), num_channels=channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if not kind:
        return nn.Identity()
    raise ValueError(f"unknown norm kind {kind!r}")


class ConvNormAct(nn.Sequential):
    """``Conv2d -> Norm -> SiLU`` convenience block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        norm: str = "group",
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=same_padding(kernel_size),
                bias=False,
            ),
            _norm(out_channels, norm),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """Pre-activation ResNet block with an identity shortcut."""

    def __init__(self, channels: int, *, norm: str = "group", dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, 3, norm=norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _norm(channels, norm)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        out = self.conv1(x)
        out = self.dropout(self.norm2(self.conv2(out)))
        return self.act(out + x)


class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell (Shi et al. 2015): the four gates are 3x3 convolutions."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, *, norm: str = "group"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=same_padding(kernel_size),
            bias=True,
        )
        self.norm = _norm(4 * hidden_channels, norm)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
        """Advance one timestep for ``x`` of shape ``(B, C_in, H, W)``."""
        if state is None:
            shape = (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
            h = x.new_zeros(shape)
            c = x.new_zeros(shape)
        else:
            h, c = state
        gates = self.norm(self.conv(torch.cat([x, h], dim=1)))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next
