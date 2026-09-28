"""Tests for the Celery layer: schedule safety, disabled connectors, idempotency."""

from __future__ import annotations

import pytest

pytest.importorskip("celery")
pytest.importorskip("sqlalchemy")

from app.config import Settings
from app.db.models import ExecutionStatus
from app.tasks.celery_app import create_celery_app, describe_schedule, get_celery_app
from app.tasks.connectors import describe_connectors, live_connectors, synthetic_connectors
from app.tasks.jobs import ingest_observations, run_batch_inference, task_status


def _settings(**overrides) -> Settings:
    base = {
        "demo_mode": True,
        "database_url": "sqlite:///:memory:",
        "grid_bbox": "78,30,79,31",
        "grid_res_km": "40",
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# application + schedule
# --------------------------------------------------------------------------- #
def test_celery_app_uses_configured_broker_and_backend() -> None:
    settings = _settings(
        celery_broker_url="redis://broker:6379/1",
        celery_result_backend="redis://broker:6379/2",
    )
    app = create_celery_app(settings)
    assert app.conf.broker_url == "redis://broker:6379/1"
    assert app.conf.result_backend == "redis://broker:6379/2"
    assert app.conf.timezone == "UTC"
    # Bounded execution so a stuck task cannot occupy a worker forever.
    assert app.conf.task_time_limit and app.conf.task_soft_time_limit
    assert app.conf.task_acks_late is True


def test_schedule_is_disabled_by_default() -> None:
    """Synthetic predictions must never be scheduled unattended."""
    report = describe_schedule(_settings())
    assert report["schedule_enabled"] is False
    assert report["beat_schedule"] == []
    assert "DISABLED" in report["note"]
    assert create_celery_app(_settings()).conf.beat_schedule == {}


def test_schedule_can_be_enabled_explicitly() -> None:
    settings = _settings(schedule_enabled=True, demo_mode=False)
    report = describe_schedule(settings)
    assert report["schedule_enabled"] is True
    assert "sihps-periodic-ingest" in report["beat_schedule"]
    assert "sihps-periodic-batch-inference" in report["beat_schedule"]


def test_schedule_requested_but_blocked_in_demo_mode() -> None:
    """Asking for scheduling in demo mode must still produce an empty schedule."""
    settings = _settings(schedule_enabled=True, demo_mode=True)
    report = describe_schedule(settings)
    assert report["schedule_requested"] is True
    assert report["schedule_enabled"] is False
    assert report["beat_schedule"] == []
    assert create_celery_app(settings).conf.beat_schedule == {}


def test_schedule_safe_requires_non_demo_mode() -> None:
    """Even with the switch on, demo mode must block scheduling."""
    assert _settings(schedule_enabled=True, demo_mode=True).schedule_safe is False
    assert _settings(schedule_enabled=True, demo_mode=False).schedule_safe is True
    assert _settings(demo_mode=False).schedule_safe is False


def test_intervals_come_from_settings() -> None:
    settings = _settings(
        schedule_enabled=True,
        demo_mode=False,
        ingest_interval_minutes=15,
        batch_inference_interval_minutes=45,
    )
    report = describe_schedule(settings)
    assert report["ingest_interval_minutes"] == 15
    assert report["batch_inference_interval_minutes"] == 45
    app = create_celery_app(settings)
    ingest = app.conf.beat_schedule["sihps-periodic-ingest"]["schedule"]
    inference = app.conf.beat_schedule["sihps-periodic-batch-inference"]["schedule"]
    assert float(ingest.seconds) == pytest.approx(15 * 60)
    assert float(inference.seconds) == pytest.approx(45 * 60)


def test_get_celery_app_is_a_singleton() -> None:
    assert get_celery_app() is get_celery_app()


def test_app_import_does_not_need_a_broker() -> None:
    """Importing the app must not attempt a broker connection."""
    import importlib

    module = importlib.import_module("app.tasks.celery_app")
    assert module.celery_app.conf.broker_url is not None


# --------------------------------------------------------------------------- #
# connectors
# --------------------------------------------------------------------------- #
def test_no_live_connector_is_reported_available() -> None:
    """With no credentials and no staged files, no real source may claim to work.

    Phase 5 added real adapters, so the honest assertion is no longer "the
    connector does not exist" but "the connector reports its concrete blocker".
    """
    live = live_connectors(_settings())
    assert live, "live connectors must be described"
    assert all(not c.available for c in live)
    assert all(not c.is_synthetic for c in live)
    # Each reason must name a concrete requirement, not a generic placeholder.
    assert all(len(c.reason) > 20 for c in live)
    assert any("account" in c.reason or "files" in c.reason for c in live)
    assert describe_connectors(_settings())["any_live_available"] is False


def test_live_connector_reasons_never_leak_credentials(monkeypatch) -> None:
    """A configured credential must not appear in any reported reason."""
    monkeypatch.setenv("SIHPS_MOSDAC_USER", "alice")
    monkeypatch.setenv("SIHPS_MOSDAC_PASSWORD", "hunter2")
    reasons = " ".join(c.reason for c in live_connectors(_settings()))
    assert "hunter2" not in reasons
    assert "alice" not in reasons


def test_synthetic_connectors_are_available_in_demo_mode() -> None:
    synthetic = synthetic_connectors(_settings(demo_mode=True))
    assert all(c.available for c in synthetic)
    assert all(c.is_synthetic for c in synthetic)


def test_synthetic_connectors_are_disabled_outside_demo_mode() -> None:
    synthetic = synthetic_connectors(_settings(demo_mode=False))
    assert all(not c.available for c in synthetic)
    assert all(c.is_synthetic for c in synthetic), "still synthetic, just not usable"


# --------------------------------------------------------------------------- #
# task behaviour (eager, no broker)
# --------------------------------------------------------------------------- #
@pytest.fixture
def eager():
    """Run Celery tasks inline so no broker is required."""
    app = get_celery_app()
    app.conf.task_always_eager = True
    yield app
    app.conf.task_always_eager = False


@pytest.fixture
def db(monkeypatch):
    """Point the session singleton at a fresh migrated in-memory database."""
    from app.db import session as session_module
    from app.db.migrations import init_db

    factory = session_module.SessionFactory("sqlite:///:memory:")
    init_db(factory.engine)
    monkeypatch.setattr(session_module, "_FACTORY", factory)
    return factory


def test_ingestion_is_disabled_without_a_live_connector(eager, db) -> None:
    """The task must report `disabled`, never fabricate observations."""
    result = ingest_observations()

    assert result["status"] == ExecutionStatus.DISABLED.value
    assert result["ingested"] == 0
    assert result["is_synthetic"] is True
    # The reason must name a concrete blocker (credentials / staged files),
    # not a generic placeholder.
    assert len(result["reason"]) > 20
    assert any(
        keyword in result["reason"].lower()
        for keyword in ("account", "files", "download", "credential")
    )
    assert result["connectors"]["any_live_available"] is False


def test_ingestion_audit_records_the_disabled_outcome(eager, db) -> None:
    from app.db.repository import list_audit_records

    ingest_observations()

    with db.scope() as session:
        rows, total = list_audit_records(session, operation="ingest")
        assert total == 1
        assert rows[0].status == ExecutionStatus.DISABLED.value
        assert rows[0].is_synthetic is True


def test_ingestion_force_does_not_invent_data(eager, db) -> None:
    """``force=True`` must not substitute synthetic data for a live feed."""
    result = ingest_observations(force=True)

    assert result["status"] == ExecutionStatus.FAILED.value
    assert result["ingested"] == 0
    assert result["is_synthetic"] is False
    assert "not implemented" in result["reason"]


def test_batch_inference_is_skipped_in_demo_mode_by_default(eager, db, monkeypatch) -> None:
    """A scheduled synthetic forecast must not be produced unattended."""
    from app.db.repository import list_forecast_runs

    monkeypatch.setattr("app.tasks.jobs.get_settings", lambda: _settings(demo_mode=True))

    result = run_batch_inference()

    assert result["status"] == ExecutionStatus.SKIPPED.value
    assert result["is_synthetic"] is True
    assert "allow_synthetic" in result["reason"]
    with db.scope() as session:
        assert list_forecast_runs(session)[1] == 0, "no run may be stored"


def test_task_status_reports_the_schedule(eager, db) -> None:
    result = task_status()
    assert result["found"] is False
    assert result["run"] is None
    assert "schedule" in result
    assert task_status(999)["found"] is False


# APPEND_MARKER_TASKS
