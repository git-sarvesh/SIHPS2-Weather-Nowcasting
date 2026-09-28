"""Regression tests for model metadata and slot-dataclass serialization."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from app.models.network import MultiTaskNowcastNet, NowcastNetConfig


def test_describe_serializes_slot_dataclass_config() -> None:
    """Model metadata must remain JSON serializable for API/audit responses."""
    model = MultiTaskNowcastNet(NowcastNetConfig.preset("lite"))

    description = model.describe()

    assert description["model_version"] == model.model_version
    assert description["n_parameters"] > 0
    assert description["config"]["backbone"]["convlstm_channels"] == [32, 16, 8]
    assert json.loads(json.dumps(description)) == description
