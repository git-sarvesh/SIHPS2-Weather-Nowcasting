"""IMD ground-observation and gridded-rainfall adapter.

Verified access facts
---------------------
Checked while writing this module:

* ``https://mausam.imd.gov.in/`` resolves and serves content (HTTP 200).
* The IMD legacy data portals ``eclims.imd.gov.in`` and ``rts.imd.gov.in`` did
  **not** resolve (DNS failure), and ``imd.gov.in`` timed out from this host.
* No documented, credential-free, machine-readable endpoint for station or
  gridded rainfall was located that an automated pipeline could rely on.

Therefore this adapter is intentionally **staged-file only**: it reads operator
supplied CSV/NetCDF station and gridded files, and reports the honest
access state rather than pretending to a feed. Nothing here labels the
synthetic station generator's output as IMD data -
``app.ingestion.synthetic.SyntheticIMDConnector`` remains the only source of
"IMD" records in demo mode, and it is flagged ``is_synthetic=True``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import GridSpec
from app.ingestion.base import DataConnector
from app.ingestion.realtime.access import AccessStatus, Availability

__all__ = [
    "IMD_ATTRIBUTION",
    "IMDObservation",
    "IMDStationConnector",
    "load_station_csv",
]

IMD_ATTRIBUTION = (
    "IMD rainfall observations: India Meteorological Department. Retain station "
    "identifiers and observation times where IMD permits; do not redistribute "
    "restricted station metadata."
)

#: Column names accepted by :func:`load_station_csv`, lower-cased.
STATION_RAIN_COLUMNS = {
    "station_id",
    "lat",
    "lon",
    "time",
    "rain_mm",
}


@dataclass(frozen=True, slots=True)
class IMDObservation:
    """One station rainfall record, with units and provenance preserved."""

    station_id: str
    lat: float
    lon: float
    time: datetime
    #: Accumulated rainfall over the record's period [mm]. May be NaN.
    rain_mm: float
    #: Optional quality flag from the source file (None when not supplied).
    quality_flag: str | None = None

    @property
    def is_missing(self) -> bool:
        """A record with no usable value is missing, *not* zero rain."""
        return not np.isfinite(self.rain_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "lat": self.lat,
            "lon": self.lon,
            "time": self.time.isoformat(),
            "rain_mm": None if self.is_missing else self.rain_mm,
            "quality_flag": self.quality_flag,
            "is_missing": self.is_missing,
        }


def _normalise_header(name: str) -> str:
    """Lowercase a CSV header and collapse separators, so ``Station ID`` and
    ``station_id`` resolve to the same alias."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())


def _parse_time(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, falling back to IMD's ``dd-mm-YYYY HH:MM``."""
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.strptime(text, "%d-%m-%Y %H:%M")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_station_csv(path: str | Path) -> list[IMDObservation]:
    """Parse an operator-supplied IMD-style station rainfall CSV.

    Accepts the short header (``station_id,lat,lon,time,rain_mm``) and the
    verbose IMD-style header with ``latitude``/``longitude``/``date``/
    ``rainfall`` plus an optional ``quality_flag`` column.

    Missing rainfall values are preserved as ``nan`` and flagged via
    :attr:`IMDObservation.is_missing`. They are **never** coerced to 0.0, which
    would be indistinguishable from a genuine dry spell.
    """
    rows: list[IMDObservation] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header row")
        # IMD exports use both "station_id" and "Station ID"; normalising to
        # lowercase with non-alphanumerics collapsed makes one alias set work.
        columns: dict[str, str] = {}
        for name in reader.fieldnames:
            key = _normalise_header(name)
            columns.setdefault(key, name)

        def pick(*candidates: str) -> str | None:
            for candidate in candidates:
                if candidate in columns:
                    return columns[candidate]
            return None

        id_col = pick("station_id", "stationid", "id", "station")
        lat_col = pick("lat", "latitude")
        lon_col = pick("lon", "long", "longitude")
        time_col = pick("time", "date", "datetime", "utctime")
        rain_col = pick("rain_mm", "rainfall", "rf", "rain", "value")
        flag_col = pick("quality_flag", "qc", "flag", "status")
        missing = [
            name
            for name, col in (
                ("station_id", id_col),
                ("lat", lat_col),
                ("lon", lon_col),
                ("time", time_col),
                ("rain_mm", rain_col),
            )
            if col is None
        ]
        if missing:
            raise ValueError(
                f"{path}: required column(s) {missing} not found in header "
                f"{list(reader.fieldnames)}"
            )

        for row in reader:
            raw_rain = (row.get(rain_col) or "").strip()
            # Empty / non-numeric rainfall means "no value", not "0 mm".
            rain_value = (
                np.nan
                if raw_rain in {"", "NA", "N/A", "null", "NaN", "-", "None"}
                else float(raw_rain)
            )
            rows.append(
                IMDObservation(
                    station_id=str(row[id_col]).strip(),
                    lat=float(row[lat_col]),
                    lon=float(row[lon_col]),
                    time=_parse_time(row[time_col]),
                    rain_mm=float(rain_value),
                    quality_flag=(
                        ((row.get(flag_col) or "").strip() or None) if flag_col else None
                    ),
                )
            )
    return rows



class IMDStationConnector(DataConnector):
    """Reads staged IMD station-rainfall files; no automated retrieval."""

    source_name = "IMD observations (staged)"
    is_synthetic = False
    data_class = "observation"

    def __init__(
        self,
        grid: GridSpec,
        *,
        data_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(grid, demo_mode=False, cache_dir=cache_dir)
        self.data_dir = Path(data_dir) if data_dir else None

    def local_files(self) -> list[Path]:
        if self.data_dir is None or not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("*.csv")) + sorted(self.data_dir.glob("*.nc"))

    def availability(self) -> AccessStatus:
        files = self.local_files()
        details = {
            "local_dir": str(self.data_dir) if self.data_dir else None,
            "n_local_files": len(files),
            "portal_reachable": "https://mausam.imd.gov.in/ (HTTP 200 when probed)",
            "legacy_portals": "eclims.imd.gov.in / rts.imd.gov.in did not resolve",
        }
        if files:
            return AccessStatus(
                source=self.source_name,
                availability=Availability.AVAILABLE,
                reason=f"{len(files)} staged IMD file(s) present",
                details=details,
                is_synthetic=False,
            )
        return AccessStatus(
            source=self.source_name,
            availability=Availability.NEEDS_MANUAL_DOWNLOAD,
            reason=(
                "no documented credential-free automated IMD endpoint was found; "
                "station/gridded files must be supplied by the operator"
            ),
            details=details,
            manual_instructions=(
                "Obtain station and/or gridded rainfall from the IMD data portal "
                "(https://mausam.imd.gov.in/ data section). Gridded rainfall for the "
                "AOI is also available as the NCMRWF MERA product at "
                "https://rds.ncmrwf.gov.in/ (hourly, 4 km, 2020-2025). Stage files "
                "in the configured directory as CSV "
                "(station_id,lat,lon,time,rain_mm) or NetCDF."
            ),
            is_synthetic=False,
        )

    def fetch(self, *args, **kwargs):
        raise NotImplementedError(
            "IMD station -> gridded target assembly is not implemented. See "
            "app.ingestion.realtime.imdaa.MERA for the gridded rainfall route."
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["attribution"] = IMD_ATTRIBUTION
        payload["data_class"] = self.data_class
        payload["availability"] = self.availability().to_dict()
        return payload

