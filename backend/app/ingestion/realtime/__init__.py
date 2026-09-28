"""Real-world weather and satellite data sources (Phase 5).

Scope and honesty
-----------------
This package implements the **adapter interface and the canonical processing
pipeline** for real observations and reanalysis. What it deliberately does *not*
do is claim an automated feed it does not have:

============  =====================================================  =================
Source        Verified access state                                  Class
============  =====================================================  =================
MOSDAC        Download needs an **approved account**; search is       observation
INSAT-3D/3DR  anonymous. Staged local L1B/L2 files are read.
IMDAA         Public catalog API is open; bulk download is            reanalysis
              account-gated. Coverage **1979-2020**, 3-hourly, ~12 km.
MERA          Same portal, open catalog. Hourly rainfall, 4 km,       observation
              **2020-2025**.
IMD stations  No documented credential-free endpoint found;           observation
              staged CSV/NetCDF only.
============  =====================================================  =================

Two facts drive much of the design and are worth stating plainly:

* **The required 2015-2022 / 2023 / 2024-2025 split cannot be built.** IMDAA
  ends in 2020; MERA starts in 2020 and holds only rainfall. See
  :func:`~.imdaa.assess_split_feasibility`, which reports the shortfall and
  proposes a documented alternative instead of manufacturing observations.
* **The canonical grid is 2 km / 30 min, but no accessible source is natively
  1 km / 30 min.** IMDAA is ~12 km and 3-hourly; MERA is 4 km and hourly;
  INSAT-3D imager is 4-8 km and 15-min. Every alignment is therefore an
  upsample/averaging step, recorded as a
  :class:`~.provenance.ResamplingRecord`.

Nothing here writes a synthetic value into a real-data field, and no
credential is ever logged or persisted.
"""

from app.ingestion.realtime.access import (
    AccessStatus,
    Availability,
    NOT_IMPLEMENTED_REASON,
    CredentialMissing,
    Credentials,
    SourceAccessError,
    credential_status,
    redact,
)
from app.ingestion.realtime.imd import (
    IMD_ATTRIBUTION,
    IMDObservation,
    IMDStationConnector,
    load_station_csv,
)
from app.ingestion.realtime.imdaa import (
    IMDAA_ATTRIBUTION,
    MERA_ATTRIBUTION,
    CoverageResult,
    IMDAAConnector,
    IMDAAProduct,
    SplitFeasibility,
    assess_split_feasibility,
    catalog_entry,
    list_datasets,
)
from app.ingestion.realtime.mosdac import (
    INSAT_ATTRIBUTION,
    INSAT_CHANNELS,
    INSATChannel,
    MOSDACConnector,
    decode_brightness_temperature,
)
from app.ingestion.realtime.pipeline import (
    AlignedField,
    CanonicalPipeline,
    PipelineResult,
    build_windows,
    composite_temporal,
    normalise_units,
    validate_dataset,
)
from app.ingestion.realtime.provenance import (
    DataClass,
    DatasetProvenance,
    QualityReport,
    ResamplingRecord,
    file_sha256,
    ingest_key,
)
from app.ingestion.realtime.sources import describe_real_sources

__all__ = [
    "AccessStatus",
    "AlignedField",
    "Availability",
    "CanonicalPipeline",
    "CoverageResult",
    "CredentialMissing",
    "Credentials",
    "DataClass",
    "DatasetProvenance",
    "IMDAAConnector",
    "IMDAAProduct",
    "IMDStationConnector",
    "INSAT_CHANNELS",
    "INSATChannel",
    "MOSDACConnector",
    "NOT_IMPLEMENTED_REASON",
    "PipelineResult",
    "QualityReport",
    "ResamplingRecord",
    "SourceAccessError",
    "SplitFeasibility",
    "assess_split_feasibility",
    "build_windows",
    "catalog_entry",
    "composite_temporal",
    "credential_status",
    "decode_brightness_temperature",
    "describe_real_sources",
    "file_sha256",
    "ingest_key",
    "list_datasets",
    "load_station_csv",
    "normalise_units",
    "redact",
    "validate_dataset",
]
