"""Shared pytest fixtures for the backend test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:  # allow `pytest` from the repository root
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.services.inference import InferenceService, reset_inference_service  # noqa: E402

#: A deliberately coarse AOI so the demo terrain + model build fast in tests.
TEST_BBOX = "78.0,30.0,79.0,31.0"
TEST_RES_KM = 12.0


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Demo-mode settings on a coarse grid (still the real code paths)."""
    return Settings(
        demo_mode=True,
        grid_bbox=TEST_BBOX,
        grid_res_km=TEST_RES_KM,
        sequence_length=6,
        forecast_steps=6,
        mc_samples=4,
    )


@pytest.fixture(scope="session")
def service(test_settings: Settings) -> InferenceService:
    """A single built :class:`InferenceService` shared by the API tests."""
    return InferenceService(test_settings)


@pytest.fixture
def client(service: InferenceService):
    """FastAPI test client bound to the coarse-grid service."""
    from fastapi.testclient import TestClient

    from app.api.v1.deps import get_inference_service
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_inference_service()
