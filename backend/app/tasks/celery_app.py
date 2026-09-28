"""Celery application with schedules driven by the existing settings.

Startup safety
--------------
The app is constructed with ``task_always_eager`` left at its default, but
:func:`create_celery_app` never contacts the broker at import time. Importing
:mod:`app.tasks.celery_app` is therefore safe with no Redis/Postgres running -
which is what keeps ``uvicorn app.main:app`` working in a bare checkout.

Scheduling safety
-----------------
Beat only runs the periodic tasks when ``SIHPS_SCHEDULE_ENABLED`` is true *and*
the data source is not synthetic-by-default. A synthetic pipeline is never
scheduled into "production" mode; the tasks themselves re-check and mark
themselves ``disabled``/``skipped`` rather than emitting synthetic forecasts
unattended.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from celery import Celery
from celery.schedules import schedule

from app import __version__
from app.config import Settings, get_settings
from app.logging_conf import get_logger

logger = get_logger("tasks.celery")

__all__ = [
    "celery_app",
    "create_celery_app",
    "describe_schedule",
    "get_celery_app",
    "reset_celery_app",
]

#: Operation names used in audit records, so API and task activity are comparable.
OP_INGEST = "ingest"
OP_BATCH_INFERENCE = "batch_inference"
OP_FORECAST_PERSIST = "forecast_persist"


def _ingest_schedule(settings: Settings) -> schedule:
    """Periodic ingestion, every ``ingest_interval_minutes`` from app start."""
    return schedule(timedelta(minutes=max(1, settings.ingest_interval_minutes)))


def _inference_schedule(settings: Settings) -> schedule:
    """Periodic batch inference, every ``batch_inference_interval_minutes``."""
    return schedule(timedelta(minutes=max(1, settings.batch_inference_interval_minutes)))


def create_celery_app(settings: Settings | None = None) -> Celery:
    """Build the Celery application from settings (no broker contact)."""
    settings = settings or get_settings()
    app = Celery(
        "sihps",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["app.tasks.jobs"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Do not let a stuck task hold a worker slot forever.
        task_time_limit=1800,
        task_soft_time_limit=1500,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_default_queue="sihps",
        result_expires=86400,
        broker_connection_retry_on_startup=True,
        # Serialise datetimes rather than repr-ing them.
        task_track_started=True,
    )
    if not getattr(settings, "schedule_safe", False):
        # Leave beat unscheduled unless the switch is on *and* demo mode is off.
        # A synthetic pipeline is never scheduled unattended.
        app.conf.beat_schedule = {}
        app.conf.sihps_schedule = {}
    else:
        app.conf.beat_schedule = {
            "sihps-periodic-ingest": {
                "task": "sihps.tasks.ingest_observations",
                "schedule": _ingest_schedule(settings),
                "options": {"queue": "sihps"},
            },
            "sihps-periodic-batch-inference": {
                "task": "sihps.tasks.run_batch_inference",
                "schedule": _inference_schedule(settings),
                "options": {"queue": "sihps"},
            },
        }
    return app


def describe_schedule(settings: Settings | None = None) -> dict[str, Any]:
    """Machine-readable description of the schedule (for ``/health`` and tests)."""
    settings = settings or get_settings()
    enabled = bool(getattr(settings, "schedule_safe", False))
    return {
        "broker_configured": bool(settings.celery_broker_url),
        "result_backend_configured": bool(settings.celery_result_backend),
        "schedule_enabled": enabled,
        "schedule_requested": bool(getattr(settings, "schedule_enabled", False)),
        "demo_mode": bool(settings.demo_mode),
        "ingest_interval_minutes": settings.ingest_interval_minutes,
        "batch_inference_interval_minutes": settings.batch_inference_interval_minutes,
        "beat_schedule": sorted((create_celery_app(settings).conf.beat_schedule or {}).keys()),
        "note": (
            "Periodic tasks are DISABLED. Synthetic predictions are never scheduled "
            "automatically; enable with SIHPS_SCHEDULE_ENABLED only alongside a real "
            "operational data source (and with SIHPS_DEMO_MODE=false)."
            if not enabled
            else "Periodic tasks are enabled."
        ),
    }


_CELERY_APP: Celery | None = None


def get_celery_app(settings: Settings | None = None) -> Celery:
    """Process-wide Celery application singleton."""
    global _CELERY_APP
    if _CELERY_APP is None:
        _CELERY_APP = create_celery_app(settings)
        logger.info("celery app configured", extra={"version": __version__})
    return _CELERY_APP


def reset_celery_app() -> None:
    """Drop the singleton (tests)."""
    global _CELERY_APP
    _CELERY_APP = None


#: Module-level app for ``celery -A app.tasks worker``.
celery_app = get_celery_app()
