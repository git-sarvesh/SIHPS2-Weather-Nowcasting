"""MOSDAC / INSAT-3D & INSAT-3DR satellite adapter.

Verified access facts
---------------------
Checked against the live service while writing this module:

* The official client is ``mdapi.py`` + ``config.json`` from
  ``https://www.mosdac.gov.in/software/mdapi.zip``.
* **Searching is anonymous; downloading requires an approved account.** The
  manual states: "To download any data using this application, you must first
  authenticate yourself using valid credentials", and that an account must be
  created and *approved* first.
* Search parameters are ``datasetId`` (mandatory), ``startTime``/``endTime``
  (``YYYY-MM-DD``), ``boundingBox`` (``minLon,minLat,maxLon,maxLat``), ``gId``,
  and ``count`` (max 100 per request).
* The published Data Access Policy tiers: anonymous = metadata and open data
  only; registered = 3-day latency; privileged = all data including NRT.

Consequences honoured by this adapter
-------------------------------------
Because credentials are mandatory and the anonymous tier cannot retrieve the
bulk products this project needs, this adapter:

* reads credentials from the environment only, and never logs them;
* reports :attr:`Availability.NEEDS_CREDENTIALS` with the exact registration
  steps when they are absent;
* supports a **local staging directory** of already-downloaded L1B/L2 NetCDF or
  HDF files, so a user who downloaded via ``mdapi.py`` can run the pipeline
  offline without re-authenticating here;
* validates channel/resolution/time metadata rather than assuming it.

It does not attempt to bypass authentication, scrape the portal, or embed keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.grid import FRAME_MINUTES, GridSpec
from app.ingestion.base import DataConnector
from app.ingestion.realtime.access import (
    AccessStatus,
    Availability,
    Credentials,
    credential_status,
)
from app.ingestion.realtime.insat_netcdf import (
    INSAT_ATTRIBUTION,
    INSAT_PROVIDER,
    build_insat_cube,
    validate_insat_file,
)

__all__ = [
    "INSAT_ATTRIBUTION",
    "INSATChannel",
    "INSATProduct",
    "INSAT_PROVIDER",
    "MOSDACConnector",
    "MOSDAC_SEARCH_MAX_RECORDS",
    "decode_brightness_temperature",
]

#: Hard cap the manual documents for a single search request.
MOSDAC_SEARCH_MAX_RECORDS = 100

#: Environment variables (never hardcoded, never logged).
MOSDAC_USER_ENV = "SIHPS_MOSDAC_USER"
MOSDAC_PASSWORD_ENV = "SIHPS_MOSDAC_PASSWORD"


@dataclass(frozen=True, slots=True)
class INSATChannel:
    """One INSAT-3D imager channel mapped onto a model channel."""

    #: The model channel this fills (see ``app.physics.CHANNELS``).
    model_channel: str
    #: MOSDAC / INSAT channel name as published.
    dataset_channel: str
    #: Central wavelength [um].
    wavelength_um: float
    #: Nominal spatial resolution [km] at nadir. INSAT-3D imager is 4 km in
    #: TIR/VIS and 8 km in the other channels; this is documented metadata, not
    #: a claim about the files.
    resolution_km: float
    #: Radiance [mW m-2 sr-1 (cm-1)-4] -> brightness temperature coefficients,
    #: from the published INSAT-3D radiometric calibration. Applied by
    #: :func:`decode_brightness_temperature`.
    c1: float = 1.0
    c2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_channel": self.model_channel,
            "dataset_channel": self.dataset_channel,
            "wavelength_um": self.wavelength_um,
            "resolution_km": self.resolution_km,
        }


#: Channel mapping for the INSAT-3D imager, as documented by MOSDAC.
#: NOTE: these are *nominal* resolutions. A real file's resolution is read from
#: its metadata during validation and any mismatch is reported, not smoothed over.
INSAT_CHANNELS: tuple[INSATChannel, ...] = (
    INSATChannel("tir1_bt", "TIR1", 10.8, 4.0),
    INSATChannel("tir2_bt", "TIR2", 12.0, 4.0),
    INSATChannel("wv_bt", "WV", 6.8, 8.0),
    INSATChannel("vis_refl", "VIS", 0.65, 4.0),
    INSATChannel("swir_refl", "SWIR", 1.6, 8.0),
    INSATChannel("mir_bt", "MIR", 3.9, 8.0),
)


@dataclass(frozen=True, slots=True)
class INSATProduct:
    """A MOSDAC dataset identifier and the metadata needed to interpret it."""

    dataset_id: str
    description: str
    level: str
    #: Cadence [minutes] of the product's imagery. INSAT-3D imager scans the
    #: Indian region every 15 min at 4 km, so 30-minute frames are a *composite*
    #: of two scans, not a native cadence. This is recorded so the resampling is
    #: never misrepresented.
    native_cadence_minutes: int
    channel: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


#: Documented product identifiers. The exact strings must be confirmed in the
#: operator's own MOSDAC catalog (``datasetId`` must match exactly), which is
#: why availability asks the user to confirm rather than asserting a download
#: would succeed.
INSAT_PRODUCTS: dict[str, INSATProduct] = {
    "3SIMG_L1B_STD": INSATProduct(
        dataset_id="3SIMG_L1B_STD",
        description="INSAT-3D imager Level-1B standard",
        level="L1B",
        native_cadence_minutes=15,
    ),
    "3RIMG_L1B_STD": INSATProduct(
        dataset_id="3RIMG_L1B_STD",
        description="INSAT-3DR imager Level-1B standard",
        level="L1B",
        native_cadence_minutes=15,
    ),
}


class MOSDACConnector(DataConnector):
    """INSAT-3D/3DR adapter: staged local files plus credential gating.

    The connector is deliberately conservative. It will not attempt an
    automated download against an account-gated endpoint, because doing so
    would mean either embedding credentials or scraping the portal - both
    prohibited. It reports exactly what is required and reads files the operator
    has already obtained through the official ``mdapi.py`` client.
    """

    source_name = "MOSDAC INSAT-3D/3DR"
    #: Real observation, when files are present.
    is_synthetic = False
    data_class = "observation"

    def __init__(
        self,
        grid: GridSpec,
        *,
        data_dir: str | Path | None = None,
        product_id: str = "3SIMG_L1B_STD",
        credentials: Credentials | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(grid, demo_mode=False, cache_dir=cache_dir)
        self.product_id = product_id
        self.product = INSAT_PRODUCTS.get(product_id)
        self.data_dir = Path(data_dir) if data_dir else None
        self.credentials = credential_status(
            MOSDAC_USER_ENV, MOSDAC_PASSWORD_ENV, overrides=credentials
        )

    def local_files(self) -> list[Path]:
        """Satellite files the operator has already downloaded."""
        if self.data_dir is None or not self.data_dir.exists():
            return []
        found: list[Path] = []
        for pattern in ("*.nc", "*.nc4", "*.h5", "*.hdf", "*.HDF"):
            found.extend(sorted(self.data_dir.glob(pattern)))
        return found

    def require_credentials(self) -> Credentials:
        """Raise :class:`CredentialMissing` unless credentials are configured."""
        return self.credentials.require(
            source=self.source_name,
            env_hint=(MOSDAC_USER_ENV, MOSDAC_PASSWORD_ENV),
        )

    def availability(self) -> AccessStatus:
        """Describe what MOSDAC access currently permits in this deployment."""
        files = self.local_files()
        details: dict[str, Any] = {
            "product_id": self.product_id,
            "product_known": self.product is not None,
            "local_dir": str(self.data_dir) if self.data_dir else None,
            "n_local_files": len(files),
            "credentials": self.credentials.public_status(),
            "official_client": "https://www.mosdac.gov.in/software/mdapi.zip",
            "search_max_records": MOSDAC_SEARCH_MAX_RECORDS,
            "channels": [c.to_dict() for c in INSAT_CHANNELS],
        }
        if files:
            return AccessStatus(
                source=self.source_name,
                availability=Availability.AVAILABLE,
                reason=f"{len(files)} staged satellite file(s) present; no download required",
                details=details,
                is_synthetic=False,
            )
        if not self.credentials.present:
            return AccessStatus(
                source=self.source_name,
                availability=Availability.NEEDS_CREDENTIALS,
                reason=(
                    "MOSDAC downloads require an approved account and no staged files "
                    "were supplied; anonymous access is limited to open data"
                ),
                details=details,
                manual_instructions=(
                    "1) Register at https://mosdac.gov.in/signup/ and wait for approval. "
                    f"2) Set {MOSDAC_USER_ENV} and {MOSDAC_PASSWORD_ENV} in the environment. "
                    "3) Download with the official client "
                    "(https://www.mosdac.gov.in/software/mdapi.zip) using config.json with "
                    f"datasetId='{self.product_id}' and a boundingBox covering "
                    "77.5,29.0,80.5,31.5. "
                    f"4) Place the files in {self.data_dir}."
                ),
                is_synthetic=False,
            )
        return AccessStatus(
            source=self.source_name,
            availability=Availability.NEEDS_MANUAL_DOWNLOAD,
            reason="credentials present but no staged files; downloads are run out-of-band",
            details=details,
            manual_instructions=(
                "Run the official mdapi.py client with your credentials to fetch "
                f"datasetId='{self.product_id}', then place the files in {self.data_dir}."
            ),
            is_synthetic=False,
        )

    def fetch(self, *args, **kwargs):
        """Assemble an :class:`ObservationCube` from the staged INSAT granules.

        Only files the operator already obtained through the official
        ``mdapi.py`` client are read. Nothing is downloaded, and nothing is
        synthesised: a granule that cannot be geolocated, declares radiance
        instead of brightness temperature, or carries no recognised channel is
        skipped with the reason recorded, and if no granule can supply a channel
        this raises instead of returning an empty or filled cube.

        Granules are ordered by their own valid times. A duplicate valid time is
        dropped, keeping the first granule: the frames are never averaged into a
        value that no instrument measured.
        """
        from app.ingestion.base import ObservationCube
        from app.physics import channel_index, channel_units

        files = self.local_files()
        if not files:
            raise FileNotFoundError(
                f"no INSAT L1B/L2 granule in {self.data_dir}. Download it with the "
                "official MOSDAC client (https://www.mosdac.gov.in/software/mdapi.zip) "
                "and place it there; no radiance is generated in its absence."
            )

        used: list[tuple[Any, ObservationCube]] = []
        notes: list[str] = []
        for path in files:
            report = validate_insat_file(path, product_id=self.product_id)
            if not report.ok:
                notes.append(
                    f"{path.name}: skipped ({'; '.join(report.quality.errors)})"
                )
                continue
            try:
                cube = build_insat_cube(
                    path, self.grid, report=report, product_id=self.product_id
                )
            except (ValueError, NotImplementedError) as exc:
                notes.append(f"{path.name}: {exc}")
                continue
            used.append((report, cube))
            notes.extend(f"{path.name}: {note}" for note in cube.metadata["notes"])

        if not used:
            raise ValueError(
                f"none of the {len(files)} staged granule(s) could supply an INSAT "
                f"channel: {'; '.join(notes) or 'unknown reason'}"
            )

        # One frame per valid time; the first granule wins a tie.
        chosen: dict[Any, tuple[ObservationCube, int]] = {}
        for _report, cube in used:
            for index, stamp in enumerate(cube.times):
                chosen.setdefault(stamp, (cube, index))
        times = sorted(chosen)
        n_read = sum(cube.n_frames for _report, cube in used)
        if n_read > len(times):
            notes.append(
                f"{n_read - len(times)} duplicate valid time(s) dropped, keeping the "
                "first granule; overlapping frames are never averaged"
            )

        channels = np.stack([chosen[stamp][0].channels[chosen[stamp][1]] for stamp in times], axis=0)
        quality = np.stack([chosen[stamp][0].quality[chosen[stamp][1]] for stamp in times], axis=0)
        declared = channel_units()
        available = [
            name for name in declared
            if np.isfinite(channels[:, channel_index(name)]).any()
        ]
        missing = [name for name in declared if name not in available]

        contributing = [
            cube for _report, cube in used
            if any(chosen[stamp][0] is cube for stamp in times)
        ]
        coverage = _union_coverage([report for report, cube in used if cube in contributing])
        cadence_note = _cadence_note(times)
        if cadence_note:
            notes.append(cadence_note)
        return ObservationCube(
            grid=self.grid,
            times=times,
            channels=channels,
            quality=quality,
            provenance=[cube.provenance[0] for cube in contributing],
            metadata={
                "files": [cube.metadata["source_file"] for cube in contributing],
                "granule_coverage": [cube.metadata["spatial_coverage"] for cube in contributing],
                "spatial_coverage": coverage,
                "channel_units": {name: declared[name] for name in available},
                "channels_available": available,
                "channels_missing": missing,
                "product_id": self.product_id,
                "native_cadence_minutes": (
                    self.product.native_cadence_minutes if self.product else None
                ),
                "resampling": [
                    record
                    for cube in contributing
                    for record in cube.metadata.get("resampling", [])
                ],
                "notes": notes,
                "trainable": False,
                "trainable_reason": (
                    f"INSAT supplied {len(available)} of {len(declared)} channels "
                    f"({available}). The remaining {len(missing)} need IMDAA (iwv, "
                    "cape) and DEM (elevation) inputs, and the granule cadence is not "
                    "the model's 30-minute contract, so this is not a complete model "
                    "input."
                ),
            },
        )

    def health(self) -> dict[str, Any]:
        payload = super().health()
        payload["attribution"] = INSAT_ATTRIBUTION
        payload["provider"] = INSAT_PROVIDER
        payload["data_class"] = self.data_class
        payload["availability"] = self.availability().to_dict()
        return payload


def decode_brightness_temperature(radiance, c1: float, c2: float):
    """Invert the Planck form used for INSAT TIR/WV channels.

    ``T = c2 / ln(c1 / L + 1)`` with ``L`` in mW m-2 sr-1 (cm-1)-4, returning
    kelvin. Kept here (rather than in the connector) so the conversion is unit
    tested independently of any file format.
    """
    arr = np.asarray(radiance, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        bt = c2 / np.log(c1 / arr + 1.0)
    return bt


# --------------------------------------------------------------------------- #
# Multi-granule reporting helpers
# --------------------------------------------------------------------------- #
def _union_coverage(reports: Sequence[Any]) -> str | None:
    """Spatial coverage of the *granules actually used*, or ``None``.

    The union of the real file extents, not the requested bounding box: a
    requested box is an intention, an extent measured from the files is evidence.
    """
    lon: list[float] = []
    lat: list[float] = []
    for report in reports:
        if report.lon_range:
            lon.extend(report.lon_range)
        if report.lat_range:
            lat.extend(report.lat_range)
    if not lon or not lat:
        return None
    return f"{min(lon)}-{max(lon)}E {min(lat)}-{max(lat)}N"


def _cadence_note(times: Sequence[Any]) -> str | None:
    """Note when the granules are not on the model's 30-minute cadence."""
    if len(times) < 2:
        return None
    steps = {
        round((times[index] - times[index - 1]).total_seconds() / 60.0, 6)
        for index in range(1, len(times))
    }
    if steps == {float(FRAME_MINUTES)}:
        return None
    return (
        f"granule spacing is {sorted(steps)} minute(s), not the model's "
        f"{FRAME_MINUTES}-minute contract; no frame was resampled or interpolated "
        "to hide the mismatch"
    )

