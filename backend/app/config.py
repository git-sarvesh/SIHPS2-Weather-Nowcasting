"""Application settings.

All configuration is environment driven (12-factor) with safe demo defaults so
``uvicorn app.main:app`` works out of the box with no credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.grid import GridSpec

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files searched for settings, in order of precedence (later wins only if unset).
_ENV_FILES = (
    REPO_ROOT / ".env",
    REPO_ROOT / "backend" / ".env",
    Path.cwd() / ".env",
)


class Settings(BaseSettings):
    """Runtime settings for the SIHPS backend."""

    model_config = SettingsConfigDict(
        env_prefix="SIHPS_",
        env_file=tuple(str(p) for p in _ENV_FILES),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # -- application ---------------------------------------------------------
    env: str = "development"
    app_name: str = "SIHPS Hyper-Local Nowcasting API"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    demo_mode: bool = True

    #: Comma separated origin list; ``*`` allows everything (dev only).
    cors_origins: str = "http://localhost:5173,http://localhost:4173,http://localhost"

    # -- persistence ---------------------------------------------------------
    db_backend: str = "sqlite"
    database_url: str = "sqlite:///./data/sihps.db"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # -- scheduling ----------------------------------------------------------
    ingest_interval_minutes: int = 30
    batch_inference_interval_minutes: int = 60
    batch_inference_districts: int = 12
    #: Master switch for Celery beat. Off by default so a synthetic pipeline is
    #: never scheduled unattended; enable only with a real data source.
    schedule_enabled: bool = False
    #: Persist Celery task outcomes to the audit table.
    audit_enabled: bool = True
    #: Retry budget for transient task failures.
    task_max_retries: int = 3
    task_retry_backoff_seconds: float = 30.0

    # -- grid / model --------------------------------------------------------
    grid_bbox: str = "77.5,29.0,80.5,31.5"
    grid_res_km: float = 2.0
    model_backend: str = "auto"  # auto | torch | numpy
    model_version: str = "sihps-convlstm-cha-v0.1.0"
    model_dir: str = "data/models"
    mc_samples: int = 20
    dropout_p: float = 0.10
    latent_dim: int = 16
    sequence_length: int = 6       # 6 x 30 min = 3 h of history
    forecast_steps: int = 12       # 12 x 30 min = 6 h of lead time
    lead_times_h: str = "0.5,1,2,3,4,5,6"
    device: str = "cpu"

    # -- risk engine ---------------------------------------------------------
    risk_weights: str = "0.3,0.4,0.3"       # thunderstorm, cloudburst, flood
    risk_thresholds: str = "0.3,0.6,0.85"   # LOW/MODERATE/HIGH/EXTREME cuts
    exposure_weights: str = "0.35,0.25,0.30,0.10"  # elev, slope, flow, landuse

    # -- alerts --------------------------------------------------------------
    alert_min_probability: float = 0.6
    alert_validity_hours: int = 6
    alert_webhook_timeout_s: float = 5.0
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "nowcast-alerts@example.org"

    # -- data source credentials (unused in demo mode) -----------------------
    mosdac_user: str = Field(default="", validation_alias=AliasChoices("MOSDAC_USER", "SIHPS_MOSDAC_USER"))
    mosdac_password: str = Field(
        default="", validation_alias=AliasChoices("MOSDAC_PASSWORD", "SIHPS_MOSDAC_PASSWORD")
    )
    mosdac_base_url: str = Field(
        default="https://www.mosdac.gov.in", validation_alias=AliasChoices("MOSDAC_BASE_URL")
    )
    imdaa_base_url: str = Field(
        default="https://data.rimes.int", validation_alias=AliasChoices("IMDAA_BASE_URL")
    )

    # -- NCMRWF RDS (IMDAA / MERA) credentials -------------------------------
    # Registration: https://rds.ncmrwf.gov.in/  (email + password).
    # Verified 2026-09-26: /datasets/* is public, but /jobs, /jobs/my and
    # /auth/me all return 401 without a bearer token, so bulk data is gated.
    rds_email: str = Field(default="", validation_alias=AliasChoices("SIHPS_RDS_EMAIL"))
    rds_password: str = Field(
        default="", validation_alias=AliasChoices("SIHPS_RDS_PASSWORD")
    )
    #: Optional explicit API base; defaults to the public portal.
    rds_api_url: str = Field(
        default="https://rds.ncmrwf.gov.in/api",
        validation_alias=AliasChoices("SIHPS_RDS_API_URL"),
    )

    # -- real-data staging directories (Phase 5) ----------------------------
    #: Directories an operator stages downloaded source files into. Empty by
    #: default, which is why every real connector reports "needs manual
    #: download" out of the box and the project stays fully offline-capable.
    mosdac_dir: str = Field(default="", validation_alias=AliasChoices("SIHPS_MOSDAC_DIR"))
    imdaa_dir: str = Field(default="", validation_alias=AliasChoices("SIHPS_IMDAA_DIR"))
    imd_dir: str = Field(default="", validation_alias=AliasChoices("SIHPS_IMD_DIR"))

    # -- validators ----------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("grid_res_km")
    @classmethod
    def _positive_res(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("grid_res_km must be > 0")
        return v

    @field_validator("forecast_steps", "sequence_length", "mc_samples")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    # -- derived properties --------------------------------------------------
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(min_lon, min_lat, max_lon, max_lat)`` for the model AOI."""
        parts = [float(p) for p in self.grid_bbox.split(",")]
        if len(parts) != 4:
            raise ValueError(f"SIHPS_GRID_BBOX needs 4 comma separated floats, got {self.grid_bbox!r}")
        return parts[0], parts[1], parts[2], parts[3]

    @property
    def grid(self) -> GridSpec:
        """Common spatio-temporal model grid (1-3 km, 30 min cadence)."""
        return GridSpec.from_bbox(self.bbox, res_km=self.grid_res_km)

    @property
    def lead_times(self) -> tuple[float, ...]:
        """Forecast lead times in hours (must fit within ``forecast_steps``)."""
        return tuple(float(p) for p in self.lead_times_h.split(",") if p.strip())

    @property
    def risk_weight_tuple(self) -> tuple[float, float, float]:
        """Normalised ``(thunderstorm, cloudburst, flood)`` fusion weights."""
        parts = [float(p) for p in self.risk_weights.split(",")]
        if len(parts) != 3:
            raise ValueError("SIHPS_RISK_WEIGHTS must contain 3 values (thunderstorm, cloudburst, flood)")
        total = sum(parts) or 1.0
        return parts[0] / total, parts[1] / total, parts[2] / total

    @property
    def threshold_tuple(self) -> tuple[float, float, float]:
        """Risk category cut points: LOW < t0 <= MODERATE < t1 <= HIGH < t2 <= EXTREME."""
        parts = [float(p) for p in self.risk_thresholds.split(",")]
        if len(parts) != 3:
            raise ValueError("SIHPS_RISK_THRESHOLDS must contain 3 values (low, moderate, high)")
        return parts[0], parts[1], parts[2]

    @property
    def exposure_weight_tuple(self) -> tuple[float, float, float, float]:
        """Normalised exposure weights ``(elevation, slope, flow_accumulation, land_use)``."""
        parts = [float(p) for p in self.exposure_weights.split(",")]
        if len(parts) != 4:
            raise ValueError("SIHPS_EXPOSURE_WEIGHTS must contain 4 values")
        total = sum(parts) or 1.0
        return parts[0] / total, parts[1] / total, parts[2] / total, parts[3] / total

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_dir(self) -> Path:
        """Directory holding demo/reference data (created on demand)."""
        path = Path(os.getenv("SIHPS_DATA_DIR", str(REPO_ROOT / "data")))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_dir_path(self) -> Path:
        """Absolute directory for trained weights / calibration artefacts."""
        path = Path(self.model_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}

    @property
    def is_sqlite(self) -> bool:
        """Whether the configured database is SQLite (dev/tests)."""
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        """Whether the configured database is PostgreSQL (production)."""
        return self.database_url.startswith(("postgresql", "postgres://"))

    @property
    def schedule_safe(self) -> bool:
        """Whether automatic scheduling is allowed right now.

        Scheduling requires the master switch *and* a non-demo configuration.
        A demo-mode deployment is a demonstration, so its predictions are never
        produced unattended, however the intervals are configured.
        """
        return bool(self.schedule_enabled) and not self.demo_mode

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to expose through ``/health`` (never secrets)."""
        return {
            "env": self.env,
            "demo_mode": self.demo_mode,
            "model_backend": self.model_backend,
            "model_version": self.model_version,
            "grid": self.grid.to_dict(),
            "sequence_length": self.sequence_length,
            "forecast_steps": self.forecast_steps,
            "lead_times_h": list(self.lead_times),
            "mc_samples": self.mc_samples,
            "risk_weights": self.risk_weight_tuple,
            "risk_thresholds": self.threshold_tuple,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache (used by tests and by long-lived Celery workers)."""
    get_settings.cache_clear()
    return get_settings()
