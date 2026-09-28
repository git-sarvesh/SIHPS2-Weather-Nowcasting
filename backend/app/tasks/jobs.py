"""Task bodies: ingestion, batch inference and forecast persistence.

All three share a shape:

1. open a short database scope (never holding one for the whole task);
2. write a ``running`` audit record;
3. do the work, reusing :class:`~app.services.inference.InferenceService` and the
   existing synthetic ingestion connectors;
4. write a terminal audit record and return a JSON-serialisable summary.

Safety rules baked in
---------------------
* ``ingest_observations`` reports ``disabled`` when no live connector exists. It
  never fabricates observations.
* ``run_batch_inference`` refuses to run unattended in demo mode unless
  ``allow_synthetic=True`` is passed explicitly by an operator, and the result
  is always flagged ``is_synthetic``.
* Idempotency keys make a redelivered task update the existing run instead of
  inserting a duplicate.
"""

from __future__ import annotations

import time
from typing import Any

from app.config import get_settings
from app.db.models import ExecutionStatus, RunKind
from app.db.migrations import init_db
from app.db.repository import (
    build_idempotency_key,
    create_audit_record,
    record_forecast_run,
    replace_risk_cells,
)
from app.db.session import DatabaseUnavailableError, session_scope
from app.logging_conf import get_logger
from app.tasks.celery_app import OP_BATCH_INFERENCE, OP_FORECAST_PERSIST, OP_INGEST, celery_app
from app.tasks.connectors import describe_connectors, live_connectors

logger = get_logger("tasks.jobs")

__all__ = [
    "persist_forecast",
    "run_batch_inference",
    "ingest_observations",
    "task_status",
]


def _audit(session, **kwargs: Any) -> None:
    """Write an audit row when auditing is enabled; never fail the task over it."""
    if not get_settings().audit_enabled:
        return
    try:
        create_audit_record(session, **kwargs)
    except Exception as exc:  # noqa: BLE001 - auditing must not break the pipeline
        logger.warning("audit write failed", extra={"error": type(exc).__name__})


@celery_app.task(
    bind=True,
    name="sihps.tasks.ingest_observations",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(),
)
def ingest_observations(self, *, force: bool = False) -> dict[str, Any]:
    """Ingest the newest observations for the configured AOI.

    Returns ``status="disabled"`` when no live connector is available, which is
    the current state of this repository. The synthetic generator is never
    presented as ingested observation data.
    """
    settings = get_settings()
    started = time.time()
    connectors = describe_connectors(settings)
    live_available = any(c["available"] for c in connectors["live"])

    if not live_available and not force:
        reason = connectors["live"][0]["reason"]
        logger.info("ingestion disabled: no live connector", extra={"reason": reason})
        with session_scope() as session:
            _audit(
                session,
                operation=OP_INGEST,
                status=ExecutionStatus.DISABLED.value,
                is_synthetic=settings.demo_mode,
                error=None,
                details={"reason": reason, "connectors": connectors},
                duration_seconds=round(time.time() - started, 3),
            )
        return {
            "status": ExecutionStatus.DISABLED.value,
            "reason": reason,
            "connectors": connectors,
            "ingested": 0,
            "is_synthetic": True,
        }

    # A live connector is only reachable once it exists; until then this branch
    # cannot run, and it refuses rather than substituting synthetic data.
    with session_scope() as session:
        _audit(
            session,
            operation=OP_INGEST,
            status=ExecutionStatus.FAILED.value,
            is_synthetic=False,
            error="live ingestion is not implemented; refusing to substitute synthetic data",
            details={"connectors": connectors},
            duration_seconds=round(time.time() - started, 3),
        )
    return {
        "status": ExecutionStatus.FAILED.value,
        "reason": "live ingestion is not implemented in this repository",
        "connectors": connectors,
        "ingested": 0,
        "is_synthetic": False,
    }


@celery_app.task(
    bind=True,
    name="sihps.tasks.run_batch_inference",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(),
)
def run_batch_inference(
    self, *, allow_synthetic: bool = False, persist_risk: bool = True
) -> dict[str, Any]:
    """Run one batch inference and persist it as a forecast run.

    Guard: in demo mode this task refuses to run unattended. The periodic
    schedule is disabled by default, and a manual invocation must pass
    ``allow_synthetic=True`` to accept a synthetic forecast - the result is still
    recorded with ``is_synthetic=True``.
    """
    settings = get_settings()
    started = time.time()
    if settings.demo_mode and not allow_synthetic:
        reason = (
            "batch inference is disabled in demo mode; pass allow_synthetic=True to run it "
            "explicitly. Synthetic predictions are never produced unattended."
        )
        logger.info("batch inference skipped", extra={"reason": reason})
        with session_scope() as session:
            _audit(
                session,
                operation=OP_BATCH_INFERENCE,
                status=ExecutionStatus.SKIPPED.value,
                is_synthetic=True,
                details={"reason": reason},
                duration_seconds=round(time.time() - started, 3),
            )
        return {"status": ExecutionStatus.SKIPPED.value, "reason": reason, "is_synthetic": True}

    from app.services.inference import InferenceService

    try:
        service = InferenceService(settings)
        result = persist_forecast(
            event_id=None,
            lead_hours=None,
            service=service,
            kind=RunKind.BATCH.value,
            persist_risk=persist_risk,
        )
    except DatabaseUnavailableError as exc:
        logger.error("batch inference could not persist", extra={"error": str(exc)})
        return {
            "status": ExecutionStatus.FAILED.value,
            "reason": "database unavailable",
            "error": str(exc),
            "is_synthetic": settings.demo_mode,
        }
    except Exception as exc:  # noqa: BLE001 - report, do not crash the worker
        logger.exception("batch inference failed")
        with session_scope() as session:
            _audit(
                session,
                operation=OP_BATCH_INFERENCE,
                status=ExecutionStatus.FAILED.value,
                is_synthetic=settings.demo_mode,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=round(time.time() - started, 3),
            )
        return {
            "status": ExecutionStatus.FAILED.value,
            "reason": f"{type(exc).__name__}: {exc}",
            "is_synthetic": settings.demo_mode,
        }

    result["duration_seconds"] = round(time.time() - started, 3)
    return result


def _ensure_schema() -> None:
    """Bring the configured database up to date before writing.

    Migrations are idempotent, so calling this on every task is cheap and
    removes the "forgot to migrate" failure mode.
    """
    from app.db.session import get_session_factory

    factory = get_session_factory()
    init_db(factory.engine)


@celery_app.task(name="sihps.tasks.persist_forecast", max_retries=2, default_retry_delay=30)
def persist_forecast(
    event_id: str | None = None,
    lead_hours: float | None = None,
    *,
    service: Any = None,
    kind: str = RunKind.API.value,
    persist_risk: bool = True,
    max_cells: int = 20000,
) -> dict[str, Any]:
    """Run inference and store the run (and optionally its risk raster).

    Idempotent: the key is derived from (kind, event, model version, init time,
    lead), so a redelivered task refreshes the same row instead of inserting a
    duplicate.
    """
    from datetime import timedelta

    from app.services.inference import InferenceService

    settings = get_settings()
    service = service or InferenceService(settings)
    started = time.time()
    _ensure_schema()

    forecast = service.forecast(event_id=event_id, lead_hours=lead_hours)
    runtime = service.ensure_runtime(event_id)
    grid = runtime.grid

    prediction = service.deterministic_fields(runtime)
    step = service.resolve_lead(runtime, lead_hours)
    risk = runtime.risk_engine.compute(
        prediction,
        init_time=runtime.init_time,
        lead_hours=runtime.lead_hours,
        step=step,
        model_version=runtime.model_version,
    )

    init_time = runtime.init_time
    valid_time = init_time + timedelta(hours=runtime.lead_hours[step])
    key = build_idempotency_key(
        model_version=runtime.model_version,
        init_time=init_time,
        lead_hours=runtime.lead_hours[step],
        event_id=forecast.get("event_id"),
        kind=kind,
    )
    is_synthetic = bool(forecast.get("is_synthetic", True))
    # Pin the exact checkpoint that produced this run, so a later reader can tell
    # which weights were in play even after the artefact is replaced.
    checkpoint_info = service.active_checkpoint_info()
    trained = bool(checkpoint_info.get("trained_checkpoint_loaded"))

    with session_scope() as session:
        run = record_forecast_run(
            session,
            idempotency_key=key,
            model_version=runtime.model_version,
            init_time=init_time,
            lead_hours=runtime.lead_hours[step],
            lead_times_h=runtime.lead_hours,
            valid_time=valid_time,
            status=ExecutionStatus.SUCCEEDED.value,
            kind=kind,
            is_synthetic=is_synthetic,
            model_backend=runtime.backend,
            checkpoint=checkpoint_info.get("checkpoint_path"),
            checkpoint_sha256=checkpoint_info.get("checkpoint_sha256"),
            trained_checkpoint_loaded=trained,
            data_source=forecast.get("data_source"),
            grid=forecast.get("grid"),
            provenance={
                "attribution": forecast.get("attribution"),
                "accuracy_claim": forecast.get("accuracy_claim"),
                "trained_checkpoint_loaded": trained,
                "training_is_synthetic": checkpoint_info.get("training_is_synthetic"),
                "training_validation_status": checkpoint_info.get("training_validation_status"),
                "calibration_present": checkpoint_info.get("calibration_present"),
                "observational_validation": False,
            },
            summary=forecast.get("risk", {}).get("summary"),
            duration_seconds=round(time.time() - started, 3),
        )
        cells_written = 0
        if persist_risk:
            cells_written = replace_risk_cells(
                session,
                run,
                thunderstorm=risk.thunderstorm,
                cloudburst=risk.cloudburst,
                flood_risk=risk.flood_risk,
                flood_probability=risk.flood_probability,
                exposure=risk.exposure,
                compound=risk.compound_storm_cloudburst,
                overall=risk.overall,
                grid=grid,
                thresholds=runtime.risk_engine.thresholds,
                uncertainty=risk.uncertainty or None,
                max_cells=max_cells,
            )
        _audit(
            session,
            operation=OP_FORECAST_PERSIST,
            status=ExecutionStatus.SUCCEEDED.value,
            forecast_run_id=run.id,
            is_synthetic=is_synthetic,
            model_version=runtime.model_version,
            data_source=forecast.get("data_source"),
            duration_seconds=round(time.time() - started, 3),
            details={"kind": kind, "risk_cells": cells_written, "idempotency_key": key},
        )

    return {
        "status": ExecutionStatus.SUCCEEDED.value,
        "forecast_run_id": run.id,
        "idempotency_key": key,
        "event_id": forecast.get("event_id"),
        "init_time": init_time.isoformat(),
        "valid_time": valid_time.isoformat(),
        "lead_hours": runtime.lead_hours[step],
        "model_version": runtime.model_version,
        "trained_checkpoint_loaded": trained,
        "is_synthetic": is_synthetic,
        "risk_cells": cells_written,
    }


@celery_app.task(name="sihps.tasks.task_status", bind=False)
def task_status(forecast_run_id: int | None = None) -> dict[str, Any]:
    """Report the schedule configuration and, optionally, one run's status."""
    from app.db.repository import get_forecast_run
    from app.tasks.celery_app import describe_schedule

    schedule_info = describe_schedule()
    if forecast_run_id is None:
        return {"schedule": schedule_info, "run": None, "found": False}
    with session_scope() as session:
        run = get_forecast_run(session, int(forecast_run_id))
        if run is None:
            return {"schedule": schedule_info, "run": None, "found": False}
        return {
            "schedule": schedule_info,
            "found": True,
            "run": {
                "id": run.id,
                "status": run.status,
                "kind": run.kind,
                "is_synthetic": run.is_synthetic,
                "model_version": run.model_version,
                "trained_checkpoint_loaded": run.trained_checkpoint_loaded,
                "init_time": run.init_time.isoformat() if run.init_time else None,
                "valid_time": run.valid_time.isoformat() if run.valid_time else None,
                "error": run.error,
            },
        }

