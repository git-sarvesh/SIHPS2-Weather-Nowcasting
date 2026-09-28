"""Aggregate the access state of every real source, for ``/health``.

:func:`describe_real_sources` is the single place that answers "can this
deployment actually read real weather data right now?". It never probes
network endpoints by default (that would make ``/health`` slow and fragile) -
it reports each connector's locally-determinable state, and callers that want a
live probe pass ``probe_network=True``.
"""

from __future__ import annotations

from typing import Any

from app.ingestion.realtime.access import AccessStatus, Availability
from app.ingestion.realtime.imd import IMDStationConnector
from app.ingestion.realtime.imd_ogc import IMDOGCConnector
from app.ingestion.realtime.imdaa import IMDAAConnector, assess_split_feasibility
from app.ingestion.realtime.mosdac import MOSDACConnector

__all__ = ["describe_real_sources", "real_source_statuses"]


def real_source_statuses(
    grid,
    *,
    mosdac_dir: str | None = None,
    imdaa_dir: str | None = None,
    imd_dir: str | None = None,
    probe_network: bool = False,
    network_timeout: float = 10.0,
) -> list[AccessStatus]:
    """Access status for each real source, without fabricating availability.

    ``probe_network=False`` (the default) keeps ``/health`` fast and offline:
    only locally-determinable state is reported. ``probe_network=True`` makes
    each connector actually contact its endpoint, which is how the OGC source
    reports genuine connectivity rather than a stored assumption.
    """
    mosdac = MOSDACConnector(grid, data_dir=mosdac_dir)
    imdaa = IMDAAConnector(grid, data_dir=imdaa_dir)
    imd = IMDStationConnector(grid, data_dir=imd_dir)
    # Phase 6: a genuinely keyless, working source.
    ogc = IMDOGCConnector(grid, bbox=(grid.min_lon, grid.min_lat, grid.max_lon, grid.max_lat))
    # ``timeout=0`` tells the IMDAA connector to skip the optional catalog read.
    timeout = network_timeout if probe_network else 0.0
    if probe_network:
        return [
            mosdac.availability(),
            imdaa.availability(timeout=timeout),
            imd.availability(),
            ogc.availability(timeout=network_timeout),
        ]
    # Offline: the OGC source reports as a known keyless endpoint without
    # asserting a connection it has not just verified.
    return [
        mosdac.availability(),
        imdaa.availability(timeout=timeout),
        imd.availability(),
        AccessStatus(
            source=ogc.source_name,
            availability=Availability.NOT_PROBED,
            reason=(
                "IMD GeoServer OGC is a keyless public service and needs no "
                "credentials; connectivity was not probed in this response "
                "(pass probe_network=True to verify it live)"
            ),
            details={"endpoint": ogc.health()["endpoint"], "data_class": "observation"},
            is_synthetic=False,
        ),
    ]


def describe_real_sources(grid=None, **kwargs: Any) -> dict[str, Any]:
    """Full report: per-source status plus the year-split feasibility verdict.

    ``grid`` defaults to the configured canonical grid, so ``/health`` can call
    this without threading a grid through.
    """
    if grid is None:
        from app.config import get_settings

        grid = get_settings().grid
    statuses = real_source_statuses(grid, **kwargs)
    feasibility = assess_split_feasibility()
    return {
        "any_real_available": any(s.available for s in statuses),
        "sources": [s.to_dict() for s in statuses],
        "split_feasibility": feasibility.to_dict(),
        "notes": [
            "The IMD GeoServer OGC source is keyless and has been used to "
            "acquire a real station-observation dataset; it is reported as "
            "'not probed' unless probe_network=True verifies it live.",
            "IMDAA and MERA bulk downloads are implemented over the NCMRWF RDS "
            "API (authenticated, credentials from SIHPS_RDS_EMAIL / "
            "SIHPS_RDS_PASSWORD) but have not been executed: no account is "
            "configured, so the portal returns 401 for every data endpoint.",
            "MOSDAC INSAT downloads still require a separate approved account.",
            "The synthetic demo pipeline remains the only source of trained data.",
        ],
    }
