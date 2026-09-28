"""Model package: hybrid CNN-ConvLSTM nowcaster with multi-task heads and CHA."""

from app.models.backbone import BackboneConfig, CNNConvLSTMBackbone
from app.models.cha import CrossHazardAttention
from app.models.multitask import HeadOutputs, MultiTaskHeads
from app.models.network import TASK_NAMES, MultiTaskNowcastNet, NowcastNetConfig
from app.models.registry import (
    build_model,
    export_onnx,
    export_torchscript,
    load_calibration,
    load_checkpoint,
    resolve_backend,
    save_checkpoint,
    torch_available,
)
from app.models.uncertainty import (
    PredictionDistribution,
    TemperatureScaler,
    VariationalBottleneck,
    crps_ensemble,
    enable_mc_dropout,
    summarise_samples,
)

__all__ = [
    "BackboneConfig",
    "CNNConvLSTMBackbone",
    "CrossHazardAttention",
    "HeadOutputs",
    "MultiTaskHeads",
    "MultiTaskNowcastNet",
    "NowcastNetConfig",
    "PredictionDistribution",
    "TASK_NAMES",
    "TemperatureScaler",
    "VariationalBottleneck",
    "build_model",
    "crps_ensemble",
    "enable_mc_dropout",
    "export_onnx",
    "export_torchscript",
    "load_calibration",
    "load_checkpoint",
    "resolve_backend",
    "save_checkpoint",
    "summarise_samples",
    "torch_available",
]
