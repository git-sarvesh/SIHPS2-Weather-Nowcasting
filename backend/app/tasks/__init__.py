"""Celery tasks: ingestion orchestration, batch inference and forecast persistence.

See :mod:`app.tasks.celery_app` for the application and schedule, and
:mod:`app.tasks.jobs` for the task bodies.

Commands (three separate processes)::

    # API
    uvicorn app.main:app --port 8000
    # worker
    celery -A app.tasks.celery_app worker --loglevel=INFO --queues=sihps
    # scheduler (only meaningful when SIHPS_SCHEDULE_ENABLED=true)
    celery -A app.tasks.celery_app beat --loglevel=INFO
"""

from app.tasks.celery_app import celery_app, create_celery_app, describe_schedule

__all__ = ["celery_app", "create_celery_app", "describe_schedule"]
