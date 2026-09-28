"""SIHPS FastAPI application.

Run locally (Windows, from ``D:\\SIHPS2\\backend``)::

    $env:PYTHONPATH = "."
    python -m uvicorn app.main:app --reload --port 8000

Then open http://localhost:8000/docs .

Scope and limitations of this phase
-----------------------------------
* The API is fully functional **in synthetic demo mode**: terrain, observations
  and the model are all synthetic or untrained, and every response is labelled
  accordingly. Nothing returned here is an official IMD warning.
* No forecasting accuracy is claimed. Independent validation against IMD/MOSDAC
  observations has not been implemented yet.
* Not yet wired: SQLAlchemy persistence/audit records, Celery ingestion and
  batch-inference tasks, the alert dispatcher, live MOSDAC/IMDAA connectors,
  the training/evaluation CLI entry points, and the React dashboard.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1 import checkpoint, explain, forecast, health, history, model, risk
from app.config import get_settings
from app.logging_conf import get_logger, setup_logging

logger = get_logger("api.main")

DESCRIPTION = """
Hyper-local (0-6 h) nowcasting API for thunderstorm, cloudburst and flash-flood
risk over the Uttarakhand AOI.

**Demo mode disclaimer.** In the default configuration every input is synthetic
and the model weights are untrained, so responses are a *pipeline* demonstration
only. They must not be used as weather warnings; always refer to IMD.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "starting API",
        extra={"env": settings.env, "demo_mode": settings.demo_mode, "version": __version__},
    )
    # Apply any pending schema migrations. This is idempotent and additive, and
    # it means the API does not depend on a Celery worker having run first
    # before the new tables (e.g. `dataset_provenance`) exist. A database that
    # cannot be reached is logged, not fatal: `/health` reports it and the API
    # still serves the read-only demo paths.
    try:
        from app.db.migrations import init_db
        from app.db.session import get_session_factory

        report = init_db(get_session_factory().engine)
        logger.info("schema up to date", extra={"revision": report.get("current_revision")})
    except Exception as exc:  # noqa: BLE001 - startup must not fail on the DB
        logger.warning("schema migration skipped", extra={"error": type(exc).__name__})
    yield
    logger.info("shutting down API")


def create_app() -> FastAPI:
    """Application factory (used by ``uvicorn app.main:app`` and by tests)."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_v1_prefix.rstrip("/")
    app.include_router(health.router, prefix=prefix)
    app.include_router(model.router, prefix=prefix)
    app.include_router(checkpoint.router, prefix=prefix)
    app.include_router(forecast.router, prefix=prefix)
    app.include_router(risk.router, prefix=prefix)
    app.include_router(explain.router, prefix=prefix)
    # Registered after /forecast so the literal /forecast/history route is not
    # shadowed by the /forecast/{run_id} pattern.
    app.include_router(history.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "app": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "api": prefix,
            "demo_mode": settings.demo_mode,
            "disclaimer": (
                "Synthetic demo outputs only; not an official IMD warning and no "
                "forecasting accuracy is claimed."
            ),
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]
