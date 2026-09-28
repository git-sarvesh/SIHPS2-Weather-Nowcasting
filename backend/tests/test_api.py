"""API tests for the v1 FastAPI layer.

These exercise the *real* integration (synthetic terrain + untrained demo model +
risk engine + explainability). They assert contracts and labelling, never
forecasting accuracy: with untrained weights the numbers are arbitrary.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.services.inference import InferenceService, ModelUnavailableError

# --------------------------------------------------------------------------- #
# health / model metadata
# --------------------------------------------------------------------------- #
def test_health_reports_loaded_components(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["runtime_loaded"] is True
    for component in ("grid", "terrain", "model", "risk_engine", "explainability"):
        assert body["components"][component]["status"] == "ok", component
    assert body["components"]["model"]["resolved_backend"] in {"torch", "numpy"}


def test_health_labels_synthetic_demo_output(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()

    assert body["demo_mode"] is True
    assert body["is_synthetic"] is True
    assert "synthetic" in body["data_source"].lower()
    assert "not an official" in body["disclaimer"].lower()
    assert "no forecasting accuracy is claimed" in body["accuracy_claim"].lower()
    # The demo event is a synthetic catalogue entry, not an observation.
    assert body["demo_event"]["documented_source"].startswith("Synthetic catalogue entry")


def test_model_describe_returns_serialisable_metadata(client: TestClient) -> None:
    response = client.get("/api/v1/model/describe")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"]
    assert body["backend"] in {"torch", "numpy"}
    assert len(body["input_shape"]) == 5
    assert body["input_shape"][0] == 1  # batch
    assert len(body["terrain_shape"]) == 4
    assert body["n_forecast_steps"] == len(body["lead_times_h"])
    assert body["hazards"] == ["thunderstorm", "cloudburst", "flood"]
    # Metadata must stay JSON round-trippable (guards the describe() regression).
    assert body["describe"]["config"]["backbone"]["convlstm_channels"]
    assert body["is_synthetic"] is True


# --------------------------------------------------------------------------- #
# forecast
# --------------------------------------------------------------------------- #
def test_forecast_runs_in_synthetic_demo_mode(client: TestClient) -> None:
    response = client.post("/api/v1/forecast", json={})

    assert response.status_code == 200
    body = response.json()
    assert set(body["fields"]) == {"thunderstorm", "cloudburst", "flood"}
    assert len(body["per_step"]) == len(body["lead_times_h"])
    assert body["selected"]["step"] == len(body["lead_times_h"]) - 1
    assert body["risk"]["summary"]["max_overall_risk"] <= 1.0
    assert body["init_time"]
    assert body["uncertainty"] is None
    assert body["is_synthetic"] is True


def test_forecast_probabilities_are_bounded(client: TestClient) -> None:
    body = client.post("/api/v1/forecast", json={}).json()

    for step in body["per_step"]:
        for hazard in ("thunderstorm", "cloudburst", "flood"):
            assert 0.0 <= step[hazard]["max"] <= 1.0, (hazard, step)


def test_forecast_rejects_unknown_field(client: TestClient) -> None:
    assert client.post("/api/v1/forecast", json={"not_a_field": 1}).status_code == 422


def test_forecast_rejects_out_of_range_lead_time(client: TestClient) -> None:
    assert client.post("/api/v1/forecast", json={"lead_hours": -1}).status_code == 422
    assert client.post("/api/v1/forecast", json={"lead_hours": 99}).status_code == 422


def test_forecast_rejects_lead_time_off_the_model_grid(client: TestClient) -> None:
    """A schema-valid but non-existent lead time is a 400, not a 500."""
    response = client.post("/api/v1/forecast", json={"lead_hours": 0.37})

    assert response.status_code == 400
    assert "lead_hours" in response.json()["detail"]


def test_forecast_unknown_event_is_404(client: TestClient) -> None:
    assert client.post("/api/v1/forecast", json={"event_id": "evt-nope"}).status_code == 404


def test_forecast_with_uncertainty_reports_ensemble_spread(client: TestClient) -> None:
    response = client.post(
        "/api/v1/forecast", json={"include_uncertainty": True, "mc_samples": 3}
    )

    assert response.status_code == 200
    uncertainty = response.json()["uncertainty"]
    assert uncertainty["n_samples"] == 3
    for hazard in ("thunderstorm", "cloudburst", "flood"):
        assert uncertainty["spread_at_selected_step"][hazard]["mean_std"] >= 0.0
        interval = uncertainty["interval_90_at_selected_step"][hazard]
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


# --------------------------------------------------------------------------- #
# risk
# --------------------------------------------------------------------------- #
def test_point_risk_schema_and_bounds(client: TestClient) -> None:
    body = client.post("/api/v1/risk/point", json={"lat": 30.5, "lon": 78.5}).json()

    assert body["row"] is not None and body["col"] is not None
    assert set(body["hazards"]) == {"thunderstorm", "cloudburst", "flood_probability"}
    for value in body["hazards"].values():
        assert 0.0 <= value <= 1.0
    assert 0.0 <= body["terrain_exposure"] <= 1.0
    assert 0.0 <= body["overall_risk"] <= 1.0
    assert body["risk_category"] in {"LOW", "MODERATE", "HIGH", "EXTREME"}
    assert body["is_synthetic"] is True


def test_point_risk_outside_aoi_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/risk/point", json={"lat": 10.0, "lon": 10.0})

    assert response.status_code == 400
    assert "outside the model AOI" in response.json()["detail"]


def test_point_risk_validates_coordinates(client: TestClient) -> None:
    assert client.post("/api/v1/risk/point", json={"lat": 200.0, "lon": 78.5}).status_code == 422
    assert client.post("/api/v1/risk/point", json={"lat": 30.5, "lon": 500.0}).status_code == 422
    assert client.post("/api/v1/risk/point", json={"lat": 30.5}).status_code == 422


def test_risk_geojson_feature_collection(client: TestClient) -> None:
    body = client.post("/api/v1/risk/geojson", json={"min_category": 1}).json()

    assert body["type"] == "FeatureCollection"
    assert isinstance(body["features"], list)
    for feature in body["features"][:5]:
        assert feature["geometry"]["type"] == "Polygon"
        ring = feature["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1]  # closed ring
        assert len(ring) == 5  # axis-aligned rectangle
        props = feature["properties"]
        assert props["risk_category"] in {"MODERATE", "HIGH", "EXTREME"}
        assert props["area_km2"] > 0.0
        assert props["is_synthetic"] is True
    assert body["metadata"]["n_features"] == len(body["features"])


def test_risk_geojson_filters_by_category(client: TestClient) -> None:
    low = client.post("/api/v1/risk/geojson", json={"min_category": 0}).json()
    high = client.post("/api/v1/risk/geojson", json={"min_category": 3}).json()

    assert high["metadata"]["n_features"] <= low["metadata"]["n_features"]


def test_risk_geojson_rejects_invalid_options(client: TestClient) -> None:
    assert client.post("/api/v1/risk/geojson", json={"risk_field": "banana"}).status_code == 422
    assert client.post("/api/v1/risk/geojson", json={"min_category": 9}).status_code == 422


# --------------------------------------------------------------------------- #
# explainability
# --------------------------------------------------------------------------- #
def test_explain_returns_attribution_and_consistency(client: TestClient) -> None:
    body = client.post("/api/v1/explain", json={"hazard": "cloudburst"}).json()

    assert body["hazard"] == "cloudburst"
    result = body["attribution_result"]
    assert result["channel_attributions"]
    assert abs(sum(result["channel_attributions"].values()) - 1.0) < 1e-3
    assert result["top_channels"]
    assert body["gradcam_map"]["max"] <= 1.0 + 1e-6
    consistency = body["physical_consistency"]
    assert 0.0 <= consistency["score"] <= 1.0
    assert isinstance(consistency["violations"], list)
    assert body["what_if"] is None
    assert body["is_synthetic"] is True


def test_explain_optional_what_if_block(client: TestClient) -> None:
    body = client.post(
        "/api/v1/explain",
        json={"hazard": "flood", "include_consistency": False, "include_what_if": True},
    ).json()

    assert body["physical_consistency"] is None
    assert set(body["what_if"]["delta"]) == {"thunderstorm", "cloudburst", "flood"}


def test_explain_rejects_unknown_hazard(client: TestClient) -> None:
    assert client.post("/api/v1/explain", json={"hazard": "snow"}).status_code == 422


def test_explain_rejects_out_of_range_step(client: TestClient) -> None:
    response = client.post("/api/v1/explain", json={"hazard": "cloudburst", "step": 9999})

    assert response.status_code == 400
    assert "step must be" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# unavailable model
# --------------------------------------------------------------------------- #
def test_non_demo_mode_without_checkpoint_returns_503(test_settings: Settings) -> None:
    """Without demo mode and without a trained checkpoint the API must not guess."""
    from app.api.v1.deps import get_inference_service
    from app.main import create_app

    production = InferenceService(test_settings.model_copy(update={"demo_mode": False}))
    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: production
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        forecast = client.post("/api/v1/forecast", json={})
    app.dependency_overrides.clear()

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["components"]["model"]["status"] == "unavailable"
    assert forecast.status_code == 503
    assert "no trained checkpoint" in forecast.json()["detail"]


def test_broken_runtime_is_reported_not_raised(test_settings: Settings) -> None:
    """A component failure surfaces as ``degraded`` health, not a 500."""
    from app.api.v1.deps import get_inference_service
    from app.main import create_app

    broken = InferenceService(test_settings)

    def _boom(event_id: str | None = None):
        raise ModelUnavailableError("synthetic failure for test")

    broken.ensure_runtime = _boom  # type: ignore[method-assign]
    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: broken
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        describe = client.get("/api/v1/model/describe")
    app.dependency_overrides.clear()

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert describe.status_code == 503
    assert "synthetic failure for test" in describe.json()["detail"]


# --------------------------------------------------------------------------- #
# application shell
# --------------------------------------------------------------------------- #
def test_root_and_openapi_expose_the_api(client: TestClient) -> None:
    root = client.get("/")

    assert root.status_code == 200
    assert root.json()["api"] == "/api/v1"
    assert root.json()["demo_mode"] is True
    assert client.get("/openapi.json").status_code == 200


def test_cors_allows_the_configured_dev_frontend(client: TestClient) -> None:
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
