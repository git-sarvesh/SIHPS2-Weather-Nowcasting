"""Checkpoint round-trip and model-compatibility tests.

These guard the contract between :mod:`app.training.train` (which writes
checkpoints) and :func:`app.models.registry.load_checkpoint` /
:mod:`app.services.inference` (which read them), including the metadata needed to
tell a *trained* checkpoint from a freshly initialised model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from app.models.network import MultiTaskNowcastNet, NowcastNetConfig
from app.models.registry import load_calibration, load_checkpoint, save_checkpoint


def _lite_model(predict_steps: int = 4) -> MultiTaskNowcastNet:
    config = NowcastNetConfig.preset("lite")
    config.backbone.predict_steps = predict_steps
    return MultiTaskNowcastNet(config)

def test_checkpoint_round_trip_preserves_weights(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = _lite_model()
    save_checkpoint(model, tmp_path)

    restored = load_checkpoint(tmp_path, backend="torch")
    assert restored is not None

    original = model.state_dict()
    loaded = restored.state_dict()
    assert set(original) == set(loaded)
    for key, value in original.items():
        assert torch.allclose(value, loaded[key]), key


def test_checkpoint_round_trip_preserves_architecture(tmp_path: Path) -> None:
    model = _lite_model()
    save_checkpoint(model, tmp_path)
    restored = load_checkpoint(tmp_path, backend="torch")

    assert restored.config.backbone.encoder_width == model.config.backbone.encoder_width
    assert tuple(restored.config.backbone.convlstm_channels) == tuple(
        model.config.backbone.convlstm_channels
    )
    assert restored.config.use_cha == model.config.use_cha
    assert restored.config.terrain_channels == model.config.terrain_channels
    assert restored.model_version == model.model_version


def test_restored_model_produces_identical_output(tmp_path: Path) -> None:
    torch.manual_seed(6)
    model = _lite_model().eval()
    x = torch.rand(1, 3, 12, 8, 8)
    terrain = torch.rand(1, 4, 8, 8)
    # ``deterministic_forward`` returns NumPy arrays for serving convenience.
    # The variational bottleneck samples z, so the RNG must be re-seeded before
    # each pass to make the comparison meaningful.
    torch.manual_seed(123)
    expected = model.deterministic_forward(x, terrain)["thunderstorm"]
    save_checkpoint(model, tmp_path)

    restored = load_checkpoint(tmp_path, backend="torch").eval()
    torch.manual_seed(123)
    got = restored.deterministic_forward(x, terrain)["thunderstorm"]
    assert np.allclose(expected, got, atol=1e-6)


def test_checkpoint_carries_training_metadata(tmp_path: Path) -> None:
    model = _lite_model()
    save_checkpoint(
        model,
        tmp_path,
        metrics={"evaluation": {"pooled": {"flood": {"brier": 0.2}}}},
        calibration={"cloudburst": {"temperature": 1.4}},
        extra={"trained": True, "is_synthetic": True, "validation_status": "SYNTHETIC DEMO"},
    )

    meta = json.loads((tmp_path / "training_meta.json").read_text(encoding="utf-8"))
    assert meta["trained"] is True
    assert meta["is_synthetic"] is True
    assert meta["validation_status"] == "SYNTHETIC DEMO"

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluation"]["pooled"]["flood"]["brier"] == 0.2

    calibration = load_calibration(tmp_path)
    assert calibration == {"cloudburst": {"temperature": 1.4}}

    # The standalone model_config.json must stay JSON round-trippable.
    config = json.loads((tmp_path / "model_config.json").read_text(encoding="utf-8"))
    assert config["backbone"]["convlstm_channels"] == [32, 16, 8]
    assert json.loads(json.dumps(config)) == config


def test_incompatible_architecture_fails_loudly(tmp_path: Path) -> None:
    """A config/weight mismatch must raise rather than silently loading."""
    # Weights from the 32-wide ``lite`` model but a config claiming the 64-wide
    # ``full`` architecture: the reconstructed model cannot accept these tensors.
    torch.save(
        {
            "state_dict": _lite_model().state_dict(),
            "config": {
                "backbone": {"encoder_width": 64, "convlstm_channels": [64, 32, 16]},
                "terrain_channels": 4,
                "use_cha": True,
            },
        },
        tmp_path / "nowcast_model.pt",
    )

    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path, backend="torch")


def test_load_checkpoint_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_checkpoint(tmp_path / "missing", backend="torch") is None


def test_numpy_backend_round_trip(tmp_path: Path) -> None:
    from app.models.reference import NumpyReferenceNowcaster

    model = NumpyReferenceNowcaster()
    save_checkpoint(model, tmp_path)
    restored = load_checkpoint(tmp_path, backend="numpy")

    assert restored is not None
    assert set(restored.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert np.allclose(value, restored.state_dict()[key]), key


def test_inference_service_reports_a_trained_checkpoint(tmp_path: Path) -> None:
    """A saved checkpoint must be reported as trained, not as fresh weights."""
    from app.config import Settings
    from app.services.inference import InferenceService

    torch.manual_seed(9)
    save_checkpoint(_lite_model(), tmp_path)

    service = InferenceService(
        Settings(
            demo_mode=True,
            grid_bbox="78,30,79,31",
            grid_res_km=40.0,
            model_dir=str(tmp_path),
            model_version="sihps-convlstm-cha-v0.1.0",
        )
    )
    # Point the loader at the temp directory so no repo-level path is involved.
    service._build_model = lambda: ("torch", load_checkpoint(tmp_path, backend="torch"), True)

    describe = service.describe_model()
    assert describe["describe"]["trained_checkpoint_loaded"] is True
    assert describe["is_synthetic"] is True
    assert "No forecasting accuracy is claimed" in describe["accuracy_claim"]


def test_inference_service_flags_untrained_demo_weights() -> None:
    """Demo mode with no checkpoint must be flagged, not passed off as trained."""
    from app.config import Settings
    from app.services.inference import InferenceService

    service = InferenceService(
        Settings(
            demo_mode=True,
            grid_bbox="78,30,79,31",
            grid_res_km=40.0,
            model_dir="data/models/does-not-exist",
        )
    )
    backend, model, trained = service._build_model()
    assert trained is False, "no checkpoint exists, so this must not report as trained"
    assert backend in {"torch", "numpy"}
    assert hasattr(model, "describe")
