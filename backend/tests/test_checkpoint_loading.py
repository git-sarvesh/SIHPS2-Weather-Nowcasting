"""Tests for active-checkpoint loading, verification and safe rejection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.inference import InferenceService, ModelUnavailableError

from tests.conftest import TEST_BBOX, TEST_RES_KM


def _settings(tmp_path: Path, **overrides):
    from app.config import Settings

    base = {
        "demo_mode": True,
        "grid_bbox": TEST_BBOX,
        "grid_res_km": TEST_RES_KM,
        "model_dir": str(tmp_path / "models"),
    }
    base.update(overrides)
    return Settings(**base)


def _write_training_meta(directory: Path, **overrides) -> Path:
    """Write a training_meta.json describing a synthetic training run."""
    meta = {
        "model_version": "sihps-convlstm-cha-v0.1.0",
        "is_synthetic": True,
        "validation_status": (
            "SYNTHETIC DEMO - agreement with the SIHPS synthetic generator only; no "
            "independent observational validation and no forecasting skill claim."
        ),
    }
    meta.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_meta.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


def test_demo_mode_without_a_checkpoint_uses_untrained_weights(tmp_path: Path) -> None:
    service = InferenceService(_settings(tmp_path))
    info = service.active_checkpoint_info()

    assert info["checkpoint_present"] is False
    assert info["trained_checkpoint_loaded"] is False
    assert info["is_synthetic"] is True
    assert info["observational_validation"] is False
    assert "synthetic demo weights" in info["validation_status"]


def test_checkpoint_info_reports_no_operational_validation(tmp_path: Path) -> None:
    """Even a trained checkpoint must not claim observational validation."""
    service = InferenceService(_settings(tmp_path))
    info = service.active_checkpoint_info()

    assert info["observational_validation"] is False
    assert info["calibration_present"] is False
    assert info["training_provenance"] is None


def test_non_demo_mode_without_a_checkpoint_is_refused(tmp_path: Path) -> None:
    """Production must not silently fall back to random weights."""
    service = InferenceService(_settings(tmp_path, demo_mode=False))
    with pytest.raises(ModelUnavailableError, match="no trained checkpoint"):
        service.ensure_runtime()


def test_training_meta_is_surfaced_when_a_checkpoint_exists(tmp_path: Path) -> None:
    """A trained artefact's own validation status must be reported verbatim."""
    from app.models.registry import CHECKPOINT_NAME
    from app.models.network import MultiTaskNowcastNet

    settings = _settings(tmp_path)
    service = InferenceService(settings)
    runtime = service.ensure_runtime()
    directory = settings.model_dir_path / runtime.model_version
    _write_training_meta(directory)

    # Persist the runtime's own architecture so the load is a genuine match.
    torch = pytest.importorskip("torch")
    from app.models.network import MultiTaskNowcastNet, NowcastNetConfig

    model = runtime.model
    torch.save(
        {
            "config": {
                "model_version": runtime.model_version,
                "terrain_channels": model.config.terrain_channels,
                "head_hidden": model.config.head_hidden,
                "use_cha": model.config.use_cha,
                "backbone": {
                    "in_channels": model.config.backbone.in_channels,
                    "encoder_width": model.config.backbone.encoder_width,
                    "n_residual_blocks": model.config.backbone.n_residual_blocks,
                    "convlstm_channels": list(model.config.backbone.convlstm_channels),
                    "kernel_size": model.config.backbone.kernel_size,
                    "latent_dim": model.config.backbone.latent_dim,
                    "dropout": model.config.backbone.dropout,
                    "norm": model.config.backbone.norm,
                    "variational": model.config.backbone.variational,
                },
            },
            "state_dict": model.state_dict(),
            "model_version": runtime.model_version,
        },
        directory / CHECKPOINT_NAME,
    )
    assert isinstance(MultiTaskNowcastNet(NowcastNetConfig.preset("lite")), MultiTaskNowcastNet)

    info = InferenceService(settings).active_checkpoint_info()
    assert info["checkpoint_present"] is True
    assert info["trained_checkpoint_loaded"] is True
    assert info["training_is_synthetic"] is True
    assert "SYNTHETIC DEMO" in info["training_validation_status"]
    assert info["observational_validation"] is False
    assert len(info["checkpoint_sha256"]) == 64


def test_corrupt_checkpoint_is_rejected_not_loaded(tmp_path: Path) -> None:
    """A damaged file must not be loaded, and must not be reported as trained."""
    from app.models.registry import CHECKPOINT_NAME

    settings = _settings(tmp_path)
    service = InferenceService(settings)
    directory = settings.model_dir_path / service.ensure_runtime().model_version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_NAME).write_bytes(b"this is not a torch checkpoint")

    fresh = InferenceService(settings)
    info = fresh.active_checkpoint_info()
    # The file exists, so the digest is still reported...
    assert info["checkpoint_present"] is True
    # ...but it must not be presented as a usable, trained model.
    assert info["trained_checkpoint_loaded"] is False
    assert len(info["checkpoint_sha256"]) == 64


def test_corrupt_checkpoint_blocks_production_mode(tmp_path: Path) -> None:
    """In production a corrupt checkpoint is an error, not a fallback."""
    from app.models.registry import CHECKPOINT_NAME

    settings = _settings(tmp_path, demo_mode=False)
    service = InferenceService(settings)
    directory = settings.model_dir_path / service.ensure_runtime.__self__._settings.model_version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_NAME).write_bytes(b"garbage")

    with pytest.raises(ModelUnavailableError):
        InferenceService(settings).ensure_runtime()


def test_unreadable_metadata_is_tolerated(tmp_path: Path) -> None:
    """Bad metadata must not crash the status endpoint."""
    settings = _settings(tmp_path)
    service = InferenceService(settings)
    directory = settings.model_dir_path / service.ensure_runtime().model_version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nowcast_model.pt").write_bytes(b"x")
    (directory / "training_meta.json").write_text("{not json", encoding="utf-8")

    info = InferenceService(settings).active_checkpoint_info()
    assert info["training_provenance"] is None
    assert info["observational_validation"] is False
