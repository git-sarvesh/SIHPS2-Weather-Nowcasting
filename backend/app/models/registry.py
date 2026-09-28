"""Model registry: construction, checkpointing and production export.

Handles backend selection (``torch`` | ``numpy`` | ``auto``), checkpoint save/load
with calibration artefacts, and TorchScript/ONNX export for serving.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from app.logging_conf import get_logger
from app.models.network import MultiTaskNowcastNet, NowcastNetConfig

logger = get_logger("models.registry")

CHECKPOINT_NAME = "nowcast_model.pt"
CONFIG_NAME = "model_config.json"
CALIBRATION_NAME = "calibration.json"


def torch_available() -> bool:
    """Whether PyTorch can be imported in this interpreter."""
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover - environment dependent
        return False
    return True


def resolve_backend(requested: str = "auto") -> str:
    """Resolve the model backend.

    ``auto`` prefers PyTorch and falls back to the NumPy reference implementation
    so the API, risk engine and dashboard always have a working predictor.
    """
    requested = (requested or "auto").lower()
    if requested == "auto":
        return "torch" if torch_available() else "numpy"
    if requested == "torch" and not torch_available():
        raise RuntimeError(
            "SIHPS_MODEL_BACKEND=torch but PyTorch is not importable. "
            "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
    if requested not in {"torch", "numpy"}:
        raise ValueError(f"unknown backend {requested!r}")
    return requested


def build_model(
    *,
    preset: str = "full",
    backend: str = "auto",
    config: NowcastNetConfig | None = None,
    model_version: str | None = None,
):
    """Instantiate a model (randomly initialised unless a checkpoint is loaded)."""
    resolved = resolve_backend(backend)
    if resolved == "numpy":
        from app.models.reference import NumpyReferenceNowcaster

        return NumpyReferenceNowcaster(version=model_version or "sihps-numpy-reference-v0.1.0")
    net_config = config or NowcastNetConfig.preset(preset)
    if model_version:
        net_config.model_version = model_version
    model = MultiTaskNowcastNet(net_config)
    logger.info("model built", extra={"summary": model.summary(), "backend": resolved})
    return model


def save_checkpoint(
    model,
    directory: str | Path,
    *,
    metrics: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights, configuration, metrics and calibration artefacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if isinstance(model, MultiTaskNowcastNet):
        import torch

        torch.save(
            {"state_dict": model.state_dict(), "config": asdict(model.config) | {"backbone_preset": None}},
            directory / CHECKPOINT_NAME,
        )
        config_payload = model.config.to_dict()
    else:  # NumPy reference backend
        np.savez(directory / CHECKPOINT_NAME.replace(".pt", ".npz"), **model.state_dict())
        config_payload = model.describe()

    (directory / CONFIG_NAME).write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")
    if metrics is not None:
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    if calibration is not None:
        (directory / CALIBRATION_NAME).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    if extra is not None:
        (directory / "training_meta.json").write_text(json.dumps(extra, indent=2, default=str), encoding="utf-8")
    logger.info("checkpoint saved", extra={"directory": str(directory)})
    return directory


def load_checkpoint(directory: str | Path, *, backend: str = "auto", map_location: str = "cpu"):
    """Load a model from a checkpoint directory (returns ``None`` when absent)."""
    directory = Path(directory)
    config_path = directory / CONFIG_NAME
    resolved = resolve_backend(backend)
    if resolved == "torch":
        import torch

        checkpoint_path = directory / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            return None
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        config = payload.get("config") if isinstance(payload, dict) else None
        model = MultiTaskNowcastNet(_config_from_payload(config))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        logger.info("torch checkpoint loaded", extra={"path": str(checkpoint_path)})
        return model

    numpy_path = directory / CHECKPOINT_NAME.replace(".pt", ".npz")
    if not numpy_path.exists():
        return None
    from app.models.reference import NumpyReferenceNowcaster

    model = NumpyReferenceNowcaster()
    with np.load(numpy_path, allow_pickle=False) as data:
        model.load_state_dict({key: data[key] for key in data.files})
    logger.info("numpy checkpoint loaded", extra={"path": str(numpy_path)})
    return model


def _config_from_payload(payload: dict[str, Any] | None) -> NowcastNetConfig:
    """Rebuild a :class:`NowcastNetConfig` from a serialised checkpoint payload."""
    from app.models.backbone import BackboneConfig

    if not payload:
        return NowcastNetConfig.preset("full")
    backbone_payload = dict(payload.get("backbone") or {})
    backbone = BackboneConfig(
        in_channels=int(backbone_payload.get("in_channels", 12)),
        encoder_width=int(backbone_payload.get("encoder_width", 64)),
        n_residual_blocks=int(backbone_payload.get("n_residual_blocks", 4)),
        convlstm_channels=tuple(backbone_payload.get("convlstm_channels", (64, 32, 16))),
        kernel_size=int(backbone_payload.get("kernel_size", 3)),
        latent_dim=int(backbone_payload.get("latent_dim", 16)),
        dropout=float(backbone_payload.get("dropout", 0.1)),
        norm=str(backbone_payload.get("norm", "group")),
        variational=bool(backbone_payload.get("variational", True)),
    )
    return NowcastNetConfig(
        backbone=backbone,
        terrain_channels=int(payload.get("terrain_channels", 4)),
        head_hidden=int(payload.get("head_hidden", 16)),
        use_cha=bool(payload.get("use_cha", True)),
        model_version=str(payload.get("model_version", "sihps-convlstm-cha-v0.1.0")),
        rain_classes=tuple(payload.get("rain_classes", ("no_rain", "light", "heavy", "extreme"))),
    )


def load_calibration(directory: str | Path) -> dict[str, Any] | None:
    """Load calibration artefacts (temperature scalers, exposure weights)."""
    path = Path(directory) / CALIBRATION_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def export_torchscript(model, directory: str | Path, *, example_shape: tuple[int, ...] = (1, 6, 12, 64, 80)) -> Path:
    """Export the deterministic inference graph to TorchScript (production serving).

    The exported module returns a tuple of tensors
    ``(thunderstorm, cloudburst, flood, rain_prob)`` for the given input window.
    Only available for the PyTorch backend.
    """
    if not isinstance(model, MultiTaskNowcastNet):
        raise TypeError("TorchScript export requires the PyTorch backend")
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: MultiTaskNowcastNet) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x, terrain):
            heads = self.inner(x, terrain)["heads"]
            return (
                heads.thunderstorm_prob,
                heads.cloudburst_prob,
                heads.flood_prob,
                heads.rain_prob,
            )

    wrapper = _Wrapper(model)
    example_x = torch.zeros(*example_shape)
    example_terrain = torch.zeros(example_shape[0], 4, *example_shape[-2:])
    traced = torch.jit.trace(wrapper, (example_x, example_terrain), strict=False)
    path = directory / "nowcast_torchscript.pt"
    traced.save(str(path))
    logger.info("torchscript exported", extra={"path": str(path)})
    return path


def export_onnx(model, directory: str | Path, *, example_shape: tuple[int, ...] = (1, 6, 12, 64, 80)) -> Path:
    """Export the model to ONNX (requires the ``onnx`` package at export time)."""
    if not isinstance(model, MultiTaskNowcastNet):
        raise TypeError("ONNX export requires the PyTorch backend")
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nowcast_model.onnx"
    torch.onnx.export(
        model,
        (torch.zeros(*example_shape), torch.zeros(example_shape[0], 4, *example_shape[-2:])),
        str(path),
        input_names=["inputs", "terrain"],
        output_names=["thunderstorm_logits", "rain_logits", "flood_logits"],
        dynamic_axes={"inputs": {0: "batch", 3: "height", 4: "width"}},
        opset_version=17,
    )
    logger.info("onnx exported", extra={"path": str(path)})
    return path
