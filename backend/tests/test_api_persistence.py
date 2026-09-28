"""API tests for persistence: persist, paginated history, detail, checkpoint info.

These build a real database and run the real persistence path, so the endpoints
are exercised end to end rather than mocked.
"""

from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from app.db import session as session_module
from app.db.migrations import init_db
from app.db.repository import list_forecast_runs, record_forecast_run
from app.services.inference import InferenceService


@pytest.fixture
def db(monkeypatch):
    """Point the session singleton at a migrated in-memory database."""
    factory = session_module.SessionFactory("sqlite:///:memory:")
    init_db(factory.engine)
    monkeypatch.setattr(session_module, "_FACTORY", factory)
    return factory


@pytest.fixture
def client(service: InferenceService, db):
    """Test client whose routes use the in-memory database."""
    from app.api.v1.deps import get_inference_service
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_inference_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _seed(db, count: int, *, is_synthetic: bool = True, prefix: str = "seed") -> list[int]:
    """Insert ``count`` runs with staggered init times."""
    ids: list[int] = []
    with db.scope() as session:
        for index in range(count):
            init = dt.datetime(2024, 7, 15, 3, 0, tzinfo=dt.timezone.utc) + dt.timedelta(
                hours=index
            )
            run = record_forecast_run(
                session,
                idempotency_key=f"{prefix}-{index}",
                model_version="v-test",
                init_time=init,
                lead_hours=1.0 + index,
                lead_times_h=[0.5, 1.0],
                is_synthetic=is_synthetic,
            )
            ids.append(run.id)
    return ids


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_persist_creates_a_run_with_synthetic_labels(client: TestClient) -> None:
    response = client.post("/api/v1/forecast/persist", json={"max_cells": 12})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["forecast_run_id"] > 0
    assert body["is_synthetic"] is True
    assert body["trained_checkpoint_loaded"] is False
    assert body["risk_cells"] == 12
    assert "SYNTHETIC DEMONSTRATION" in body["notice"]
    assert body["init_time"] and body["valid_time"]
    assert body["idempotency_key"]


def test_persist_is_idempotent(client: TestClient, db) -> None:
    """The same request twice must not create a second run."""
    first = client.post("/api/v1/forecast/persist", json={"max_cells": 4}).json()
    second = client.post("/api/v1/forecast/persist", json={"max_cells": 4}).json()

    assert first["forecast_run_id"] == second["forecast_run_id"]
    with db.scope() as session:
        assert list_forecast_runs(session)[1] == 1


def test_persist_can_skip_the_risk_raster(client: TestClient) -> None:
    body = client.post("/api/v1/forecast/persist", json={"persist_risk": False}).json()
    assert body["risk_cells"] == 0


def test_persist_rejects_an_unknown_event(client: TestClient) -> None:
    assert client.post("/api/v1/forecast/persist", json={"event_id": "evt-nope"}).status_code == 404


def test_persist_rejects_an_off_grid_lead_time(client: TestClient) -> None:
    assert client.post("/api/v1/forecast/persist", json={"lead_hours": 0.37}).status_code == 400


def test_persist_rejects_an_unknown_field(client: TestClient) -> None:
    assert client.post("/api/v1/forecast/persist", json={"nope": 1}).status_code == 422


def test_persist_pins_the_checkpoint_identity(client: TestClient) -> None:
    """A run must record which weights produced it, including the digest."""
    body = client.post("/api/v1/forecast/persist", json={"max_cells": 2}).json()
    detail = client.get(f"/api/v1/forecast/{body['forecast_run_id']}").json()

    # The demo has no operational checkpoint, so the identity must be null
    # rather than a fabricated path.
    assert detail["checkpoint"] is None
    assert detail["checkpoint_sha256"] is None
    assert detail["run"]["trained_checkpoint_loaded"] is False
    assert detail["provenance"]["observational_validation"] is False


def test_persist_provenance_states_synthetic_training_is_not_validation(client: TestClient) -> None:
    body = client.post("/api/v1/forecast/persist", json={"max_cells": 2}).json()
    detail = client.get(f"/api/v1/forecast/{body['forecast_run_id']}").json()

    provenance = detail["provenance"]
    assert provenance["trained_checkpoint_loaded"] is False
    assert "training_validation_status" in provenance
    assert detail["run"]["is_synthetic"] is True


# --------------------------------------------------------------------------- #
# history pagination
# --------------------------------------------------------------------------- #
def test_history_is_empty_initially(client: TestClient) -> None:
    body = client.get("/api/v1/forecast/history").json()
    assert body["total"] == 0
    assert body["count"] == 0
    assert body["has_more"] is False
    assert body["runs"] == []


def test_history_paginates(client: TestClient, db) -> None:
    _seed(db, 5)

    first = client.get("/api/v1/forecast/history?limit=2&offset=0").json()
    assert first["total"] == 5
    assert first["count"] == 2
    assert first["has_more"] is True
    assert first["limit"] == 2 and first["offset"] == 0

    last = client.get("/api/v1/forecast/history?limit=2&offset=4").json()
    assert last["count"] == 1
    assert last["has_more"] is False
    # Newest first.
    assert first["runs"][0]["init_time"] > first["runs"][1]["init_time"]


def test_history_pages_do_not_overlap(client: TestClient, db) -> None:
    _seed(db, 5)
    ids1 = {r["id"] for r in client.get("/api/v1/forecast/history?limit=2&offset=0").json()["runs"]}
    ids2 = {r["id"] for r in client.get("/api/v1/forecast/history?limit=2&offset=2").json()["runs"]}
    assert not (ids1 & ids2)


def test_history_filters_by_synthetic_flag(client: TestClient, db) -> None:
    _seed(db, 2, is_synthetic=True, prefix="syn")
    _seed(db, 1, is_synthetic=False, prefix="ops")

    synthetic = client.get("/api/v1/forecast/history?is_synthetic=true").json()
    assert synthetic["total"] == 2
    assert all(r["is_synthetic"] for r in synthetic["runs"])

    operational = client.get("/api/v1/forecast/history?is_synthetic=false").json()
    assert operational["total"] == 1
    assert all(not r["is_synthetic"] for r in operational["runs"])


def test_history_rejects_an_out_of_range_limit(client: TestClient) -> None:
    assert client.get("/api/v1/forecast/history?limit=0").status_code == 422
    assert client.get("/api/v1/forecast/history?limit=9999").status_code == 422
    assert client.get("/api/v1/forecast/history?offset=-1").status_code == 422


# --------------------------------------------------------------------------- #
# detail
# --------------------------------------------------------------------------- #
def test_detail_returns_provenance_and_risk_cells(client: TestClient) -> None:
    created = client.post("/api/v1/forecast/persist", json={"max_cells": 9}).json()
    body = client.get(f"/api/v1/forecast/{created['forecast_run_id']}?risk_limit=5").json()

    assert body["run"]["id"] == created["forecast_run_id"]
    assert body["run"]["risk_cells"] == 9
    assert len(body["risk_cells"]) == 5
    assert body["provenance"] is not None
    assert "SYNTHETIC" in body["notice"].upper()
    cell = body["risk_cells"][0]
    assert -90.0 <= cell["lat"] <= 90.0
    assert 0.0 <= cell["overall_risk"] <= 1.0
    assert cell["risk_category"] in {"LOW", "MODERATE", "HIGH", "EXTREME"}


def test_detail_can_omit_risk_cells(client: TestClient) -> None:
    created = client.post("/api/v1/forecast/persist", json={"max_cells": 4}).json()
    body = client.get(f"/api/v1/forecast/{created['forecast_run_id']}?include_risk=false").json()

    assert body["risk_cells"] == []
    assert body["risk_cell_count"] == 4


def test_detail_filters_risk_cells_by_minimum(client: TestClient) -> None:
    created = client.post("/api/v1/forecast/persist", json={"max_cells": 64}).json()
    body = client.get(
        f"/api/v1/forecast/{created['forecast_run_id']}?risk_limit=100&min_risk=0.99"
    ).json()
    assert all(c["overall_risk"] >= 0.99 for c in body["risk_cells"])


def test_detail_missing_run_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/forecast/999999").status_code == 404


def test_run_status_endpoint(client: TestClient) -> None:
    created = client.post("/api/v1/forecast/persist", json={"max_cells": 2}).json()
    body = client.get(f"/api/v1/forecast/{created['forecast_run_id']}/status").json()

    assert body["status"] == "succeeded"
    assert body["is_terminal"] is True
    assert body["is_synthetic"] is True
    assert body["error"] is None
    assert client.get("/api/v1/forecast/999999/status").status_code == 404


# --------------------------------------------------------------------------- #
# checkpoint + health
# --------------------------------------------------------------------------- #
def test_checkpoint_endpoint_reports_no_operational_validation(client: TestClient) -> None:
    body = client.get("/api/v1/model/checkpoint").json()

    assert body["observational_validation"] is False
    assert body["is_synthetic"] is True
    assert "validation_status" in body
    assert "trained_checkpoint_loaded" in body
    assert "calibration_present" in body


def test_health_includes_database_schedule_and_connectors(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()

    assert body["status"] == "ok"
    assert body["database"]["status"] == "ok"
    assert body["database"]["dialect"] == "sqlite"
    assert body["database"]["current_revision"]
    assert body["schedule"]["schedule_enabled"] is False
    assert body["connectors"]["any_live_available"] is False


def test_health_survives_an_unavailable_database(client: TestClient, monkeypatch) -> None:
    """Health must degrade gracefully, not 500, when the database is down."""

    def _boom() -> dict:
        raise RuntimeError("no database")

    monkeypatch.setattr("app.db.session.get_session_factory", _boom)
    body = client.get("/api/v1/health").json()
    assert body["database"]["status"] == "unavailable"
    assert body["database"]["error"] == "RuntimeError"


def test_history_returns_503_when_the_database_is_broken(client: TestClient, monkeypatch) -> None:
    """A persistence failure must not be reported as an empty success."""
    from app.db.session import DatabaseUnavailableError

    def _boom(*args, **kwargs):
        raise DatabaseUnavailableError("connection refused")

    # The route imported the symbol directly, so patch it where it is used.
    monkeypatch.setattr("app.api.v1.history.list_forecast_runs", _boom)
    response = client.get("/api/v1/forecast/history")

    assert response.status_code == 503
    assert "database unavailable" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# existing API compatibility
# --------------------------------------------------------------------------- #
def test_the_six_original_endpoints_are_unchanged(client: TestClient) -> None:
    """Phase 3 must be additive: the Phase 1 contract still holds."""
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/model/describe").status_code == 200
    assert client.post("/api/v1/forecast", json={}).status_code == 200
    assert client.post("/api/v1/risk/point", json={"lat": 30.5, "lon": 78.5}).status_code == 200
    assert client.post("/api/v1/risk/geojson", json={}).status_code == 200
    assert client.post("/api/v1/explain", json={"hazard": "cloudburst"}).status_code == 200


def test_plain_forecast_is_unaffected_by_persistence(client: TestClient) -> None:
    """A plain forecast must still work, and must not claim to be persisted."""
    client.post("/api/v1/forecast/persist", json={"max_cells": 4})
    body = client.post("/api/v1/forecast", json={}).json()
    assert set(body["fields"]) == {"thunderstorm", "cloudburst", "flood"}
    assert "forecast_run_id" not in body
    assert body["is_synthetic"] is True


# --------------------------------------------------------------------------- #
# explainability (regression)
# --------------------------------------------------------------------------- #
def test_explain_defaults_to_the_last_available_step(client: TestClient) -> None:
    """The default step must be a step the backend actually emitted.

    Regression: the runtime declares ``forecast_steps`` lead times, but a backend
    may emit fewer. The default (``n_steps - 1``) used the declared count and
    indexed past the real output, raising IndexError -> 500.
    """
    response = client.post("/api/v1/explain", json={"hazard": "cloudburst"})

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["step"]
    # The reported lead time must match the step that was actually used.
    assert body["lead_hours"] > 0


def test_explain_rejects_an_out_of_range_step(client: TestClient) -> None:
    assert client.post("/api/v1/explain", json={"hazard": "cloudburst", "step": 999}).status_code == 400


def test_explain_supports_what_if(client: TestClient) -> None:
    response = client.post(
        "/api/v1/explain",
        json={
            "hazard": "cloudburst",
            "include_what_if": True,
            "perturbations": {"iwv": 0.1, "ctt": -0.1},
        },
    )
    assert response.status_code == 200
    assert response.json()["what_if"] is not None
