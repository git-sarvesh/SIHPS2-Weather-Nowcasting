"""Connector availability: what can actually be ingested right now.

This module is the single place that answers "is there live data?". It probes
the existing ingestion connectors and reports their availability honestly:

* ``Synthetic*`` connectors are always available and always marked synthetic.
* Real connectors (:mod:`app.ingestion.realtime`) report their own access
  state. Phase 5 added the MOSDAC / IMDAA / IMD adapters; each returns an
  :class:`~app.ingestion.realtime.access.AccessStatus` naming the exact
  blocker (missing credentials, missing local files) rather than a generic
  "not implemented".
* With no staged files and no credentials, :func:`live_connectors` reports
  every real source as unavailable, and the ingestion task marks itself
  ``disabled`` rather than fabricating observations.

No function here ever invents live weather data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.logging_conf import get_logger

logger = get_logger("tasks.connectors")

__all__ = [
    "ConnectorStatus",
    "describe_connectors",
    "live_connectors",
    "synthetic_connectors",
]


@dataclass(frozen=True, slots=True)
class ConnectorStatus:
    """Availability of one data source."""

    name: str
    available: bool
    is_synthetic: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "is_synthetic": self.is_synthetic,
            "reason": self.reason,
        }


def synthetic_connectors(settings: Settings | None = None) -> list[ConnectorStatus]:
    """The demo generators that always exist."""
    settings = settings or get_settings()
    reason = "SYNTHETIC generator; available in demo mode, never an observation"
    if not settings.demo_mode:
        return [
            ConnectorStatus(name, available=False, is_synthetic=True, reason=reason)
            for name in ("INSAT-3D/3DR (synthetic)", "IMDAA (synthetic)", "IMD (synthetic)")
        ]
    return [
        ConnectorStatus(name, available=True, is_synthetic=True, reason=reason)
        for name in ("INSAT-3D/3DR (synthetic)", "IMDAA (synthetic)", "IMD (synthetic)")
    ]


def live_connectors(
    settings: Settings | None = None, *, grid: Any | None = None, probe_network: bool = False
) -> list[ConnectorStatus]:
    """Real MOSDAC / IMDAA / IMD readers and their current access state.

    Each entry reports what is actually blocking use. With no credentials and no
    staged files every source comes back unavailable, which is what stops the
    ingestion task from pretending it has an observation feed.
    """
    _settings = settings or get_settings()
    from app.ingestion.realtime.sources import real_source_statuses

    target_grid = grid or _settings.grid
    try:
        statuses = real_source_statuses(
            target_grid,
            mosdac_dir=_settings.mosdac_dir or None,
            imdaa_dir=_settings.imdaa_dir or None,
            imd_dir=_settings.imd_dir or None,
            probe_network=probe_network,
        )
    except Exception as exc:  # noqa: BLE001 - health must not raise
        logger.warning("real connector probe failed", extra={"error": type(exc).__name__})
        return [
            ConnectorStatus(
                name=name,
                available=False,
                is_synthetic=False,
                reason=f"probe failed: {type(exc).__name__}",
            )
            for name in (
                "MOSDAC INSAT-3D/3DR (live)",
                "IMDAA reanalysis (live)",
                "IMD stations (live)",
            )
        ]
    return [
        ConnectorStatus(
            name=status.source,
            available=status.available,
            is_synthetic=status.is_synthetic,
            # The reason names the blocker; the manual step is appended so an
            # operator sees exactly what to do from /health alone.
            reason=(
                status.reason
                + (f" | {status.manual_instructions}" if status.manual_instructions else "")
            ),
        )
        for status in statuses
    ]


def describe_connectors(
    settings: Settings | None = None, *, grid: Any | None = None
) -> dict[str, Any]:
    """Full availability report for ``/health`` and the ingestion task."""
    settings = settings or get_settings()
    live = live_connectors(settings, grid=grid)
    synthetic = synthetic_connectors(settings)
    return {
        "demo_mode": settings.demo_mode,
        "any_live_available": any(c.available for c in live),
        "live": [c.to_dict() for c in live],
        "synthetic": [c.to_dict() for c in synthetic],
    }
