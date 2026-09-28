"""Authenticated client for the NCMRWF RDS portal (IMDAA / MERA).

Verified access boundary (probed 2026-09-26; reproduced in
``docs/phase7_imdaa_acquisition.md``)
--------------------------------------------------------------------
======================  =========  =========================================
Endpoint                 Auth?      Verified response (no credentials)
======================  =========  =========================================
``GET /``                no         200 "Backend IMDAA is running"
``GET /datasets/catalog`` no         200, 9 datasets
``GET /datasets/{slug}`` no         200, full metadata
``POST /auth/login``     n/a        422 on a syntactically invalid email
``GET /auth/me``         **yes**    401 "Not authenticated"
``GET /jobs/my``         **yes**    401
``GET /jobs/id/{id}``    **yes**    401
``POST /jobs``           **yes**    401
======================  =========  =========================================

The **metadata is open; every endpoint that moves data is gated** behind a
bearer token from ``POST /auth/login``.

Workflow implemented here
-------------------------
1. ``POST /auth/login`` with ``{email, password}`` -> ``{access_token, token_type}``
2. ``POST /jobs`` with ``Authorization: Bearer <token>`` and a
   ``JobCreateRequest`` -> a job id
3. ``GET /jobs/id/{job_id}`` polled until ``COMPLETED``/``FAILED``
4. fetch the returned ``file_url`` and validate it before use

Credential handling
-------------------
* Read from ``SIHPS_RDS_EMAIL`` / ``SIHPS_RDS_PASSWORD`` only.
* The token lives in memory for the process lifetime. It is **never** logged,
  written to disk, or included in a provenance record;
  :meth:`RDSClient.public_status` reports only its length.
* No credential is hardcoded, and no endpoint is accessed by bypassing auth.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.realtime.access import AccessStatus, Availability, SourceAccessError
from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.rds")

__all__ = [
    "IMDAA_STANDARD_LEVELS_HPA",
    "RDS_API_URL",
    "RDSClient",
    "RDSCredentials",
    "RDSDownloadError",
    "RDSJob",
    "build_sample_request",
]

RDS_API_URL = "https://rds.ncmrwf.gov.in/api"

#: Environment variables, documented in ``.env.example``.
RDS_EMAIL_ENV = "SIHPS_RDS_EMAIL"
RDS_PASSWORD_ENV = "SIHPS_RDS_PASSWORD"

#: Terminal and successful job states reported by the portal.
JOB_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "ERROR", "CANCELLED"})
JOB_SUCCESS_STATES = frozenset({"COMPLETED"})

#: Standard IMDAA pressure levels published for the pressure-level product.
#: Used only to *document* the expected axis; the real levels are always read
#: from the file and never assumed.
IMDAA_STANDARD_LEVELS_HPA: tuple[int, ...] = (
    1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100,
)


class RDSDownloadError(SourceAccessError):
    """The RDS portal could not be used as requested.

    The message is safe to surface: it names the requirement that failed, never
    a credential or token value.
    """


def _redact(value: str) -> str:
    """Mask a secret for display, keeping only a length hint."""
    if not value:
        return "<unset>"
    if "@" in value:
        # An address is identifying but not secret; do not echo it.
        return "<set:email>"
    return f"<set:{len(value)} chars>"


def _json_or_raise(response: Any, what: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RDSDownloadError(f"RDS {what} returned a non-JSON body") from exc
    if not isinstance(payload, dict):
        raise RDSDownloadError(
            f"RDS {what} returned {type(payload).__name__}, not an object"
        )
    return payload


def _looks_textual(content_type: str) -> bool:
    lowered = content_type.lower()
    return "json" in lowered or "html" in lowered or "xml" in lowered or not lowered


@dataclass(frozen=True, slots=True)
class RDSCredentials:
    """RDS portal credentials, sourced from the environment only."""

    email: str = ""
    password: str = ""
    email_env: str = RDS_EMAIL_ENV
    password_env: str = RDS_PASSWORD_ENV

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"RDSCredentials(email={_redact(self.email)}, "
            f"password={_redact(self.password)}, "
            f"email_env={self.email_env!r}, password_env={self.password_env!r})"
        )

    __str__ = __repr__

    @property
    def present(self) -> bool:
        return bool(self.email.strip() and self.password.strip())

    def require(self) -> RDSCredentials:
        """Return self, or raise naming the exact external action required."""
        if self.present:
            return self
        raise RDSDownloadError(
            "NCMRWF RDS credentials are not configured. Set "
            f"{RDS_EMAIL_ENV} and {RDS_PASSWORD_ENV} in the environment. "
            "Register for an account at https://rds.ncmrwf.gov.in/ first; "
            "IMDAA/MERA bulk downloads return 401 without a bearer token."
        )

    @classmethod
    def from_env(cls) -> RDSCredentials:
        return cls(
            email=os.getenv(RDS_EMAIL_ENV, "").strip(),
            password=os.getenv(RDS_PASSWORD_ENV, ""),
        )

    def public_status(self) -> dict[str, Any]:
        """Credential state safe for ``/health`` and provenance records."""
        return {
            "email_env": self.email_env,
            "password_env": self.password_env,
            "email": _redact(self.email),
            "password": _redact(self.password),
            "present": self.present,
        }


@dataclass(slots=True)
class RDSJob:
    """One submitted download job and its polled state."""

    job_id: str
    status: str
    file_url: str | None = None
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    history: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.status.upper() in JOB_TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        return self.status.upper() in JOB_SUCCESS_STATES

    def to_dict(self) -> dict[str, Any]:
        """Serialisable state. Never contains the bearer token."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "file_url": self.file_url,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status_history": list(self.history),
        }



def build_sample_request(
    *,
    dataset_type: str = "imdaa-daily",
    year: str = "2019",
    month: Sequence[str] = ("07",),
    day: Sequence[str] = ("01",),
    time: Sequence[str] = ("00",),
    variables: Sequence[str] = ("2t", "dpt", "r", "u", "v", "msl", "tcc"),
    min_lon: float = 77.5,
    min_lat: float = 29.0,
    max_lon: float = 80.5,
    max_lat: float = 31.5,
    frequency: str | None = None,
    pressure_level: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a small, explicit ``JobCreateRequest`` for the Uttarakhand AOI.

    Field names come from the portal's published ``JobPayloadSchema``, not from
    guesswork; ``area`` is the documented key for the geographic box and its
    sub-values are strings, per ``GeoArea``.

    Defaults describe one day of IMDAA single-level data. Variable codes are
    the short names IMDAA publishes; they are still validated against the
    downloaded file rather than assumed correct.
    """
    payload: dict[str, Any] = {
        "dataset_type": dataset_type,
        "year": str(year),
        "month": [str(m) for m in month],
        "day": [str(d) for d in day],
        "time": [str(t) for t in time],
        "variables": [str(v) for v in variables],
        "area": {
            "type": "rectangle",
            "north": str(max_lat),
            "south": str(min_lat),
            "east": str(max_lon),
            "west": str(min_lon),
        },
    }
    if frequency:
        payload["frequency"] = frequency
    if pressure_level:
        payload["pressure_level"] = [str(p) for p in pressure_level]
    return {"request_payload": payload}



class RDSClient:
    """Minimal, honest client for the authenticated RDS workflow.

    Nothing here fabricates a download. With no credentials,
    :meth:`availability` reports the blocker and every mutating call raises
    :class:`RDSDownloadError` naming the missing environment variables.
    """

    def __init__(
        self,
        *,
        base_url: str = RDS_API_URL,
        credentials: RDSCredentials | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or RDSCredentials.from_env()
        self.timeout = timeout
        #: Bearer token, held in memory only. Never logged or persisted.
        self._token: str | None = None

    # ------------------------------------------------------------------ auth
    def _require_token(self) -> str:
        if self._token:
            return self._token
        creds = self.credentials.require()
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url}/auth/login",
                json={"email": creds.email, "password": creds.password},
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - transport failure
            raise RDSDownloadError(
                f"could not reach the RDS login endpoint ({type(exc).__name__})"
            ) from exc
        if response.status_code in (401, 403):
            raise RDSDownloadError(
                "RDS rejected the configured credentials (HTTP "
                f"{response.status_code}). Verify {RDS_EMAIL_ENV}/{RDS_PASSWORD_ENV} "
                "or re-register at https://rds.ncmrwf.gov.in/."
            )
        if response.status_code >= 400:
            raise RDSDownloadError(f"RDS login failed with HTTP {response.status_code}")
        payload = _json_or_raise(response, "login")
        token = payload.get("access_token")
        if not token:
            raise RDSDownloadError(
                "RDS login response contained no access_token; the portal's "
                "auth contract may have changed"
            )
        self._token = str(token)
        # Logged without the token value.
        logger.info("RDS authenticated", extra={"token_chars": len(self._token)})
        return self._token

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._require_token()}"}

    # ----------------------------------------------------------------- jobs
    def submit_job(self, request: dict[str, Any]) -> RDSJob:
        """Submit a ``JobCreateRequest`` and return the created job."""
        try:
            import httpx

            response = httpx.post(
                f"{self.base_url}/jobs",
                json=request,
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise RDSDownloadError(
                f"could not submit the RDS job ({type(exc).__name__})"
            ) from exc
        if response.status_code in (401, 403):
            raise RDSDownloadError(
                "RDS refused the job submission as unauthenticated (HTTP "
                f"{response.status_code}). The token may have expired; retry."
            )
        if response.status_code >= 400:
            raise RDSDownloadError(
                f"RDS rejected the job request (HTTP {response.status_code}): "
                f"{response.text[:200]}"
            )
        payload = _json_or_raise(response, "job submission")
        job_id = payload.get("id") or payload.get("job_id")
        if not job_id:
            raise RDSDownloadError(f"RDS job response had no id: {payload!r}")
        return RDSJob(
            job_id=str(job_id),
            status=str(payload.get("status", "SUBMITTED")),
            file_url=payload.get("file_url"),
            error=payload.get("last_error") or payload.get("error"),
            created_at=payload.get("created_at"),
            completed_at=payload.get("completed_at"),
            history=[str(payload.get("status", "SUBMITTED"))],
        )

    def poll_job(self, job_id: str) -> RDSJob:
        """Fetch the current state of one job."""
        try:
            import httpx

            response = httpx.get(
                f"{self.base_url}/jobs/id/{job_id}",
                headers=self._auth_headers(),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise RDSDownloadError(
                f"could not poll RDS job {job_id} ({type(exc).__name__})"
            ) from exc
        if response.status_code >= 400:
            raise RDSDownloadError(
                f"RDS job poll failed (HTTP {response.status_code}) for {job_id}"
            )
        payload = _json_or_raise(response, "job poll")
        return RDSJob(
            job_id=str(payload.get("id", job_id)),
            status=str(payload.get("status", "UNKNOWN")),
            file_url=payload.get("file_url"),
            error=payload.get("last_error"),
            created_at=payload.get("created_at"),
            completed_at=payload.get("completed_at"),
        )


    def wait_for_job(
        self, job: RDSJob, *, poll_seconds: float = 5.0, max_polls: int = 60
    ) -> RDSJob:
        """Poll until terminal, or raise.

        Raising rather than returning a half-finished job is deliberate: a
        caller must never be able to treat "still running" as "data acquired".
        """
        current = job
        for _attempt in range(1, max(1, max_polls) + 1):
            if current.finished:
                break
            time.sleep(max(0.0, poll_seconds))
            current = self.poll_job(current.job_id)
            current.history.append(current.status)
        if not current.finished:
            raise RDSDownloadError(
                f"RDS job {job.job_id} did not finish within {max_polls} polls "
                f"(last status {current.status!r}); it may still be running"
            )
        if not current.succeeded:
            raise RDSDownloadError(
                f"RDS job {job.job_id} finished with status {current.status!r}"
                + (f": {current.error}" if current.error else "")
            )
        if not current.file_url:
            raise RDSDownloadError(
                f"RDS job {job.job_id} reported COMPLETED but returned no file_url"
            )
        return current

    # -------------------------------------------------------------- download
    def download(self, file_url: str, dest_dir: str | Path) -> Path:
        """Download a completed job's artefact, validating it is real data.

        Rejects an HTML body served with HTTP 200 (the same failure mode seen on
        the OGC service) and writes a SHA-256 sidecar next to the file.
        """
        import hashlib

        try:
            import httpx

            response = httpx.get(file_url, timeout=self.timeout, follow_redirects=True)
        except Exception as exc:  # noqa: BLE001
            raise RDSDownloadError(
                f"could not download the RDS artefact ({type(exc).__name__})"
            ) from exc
        if response.status_code >= 400:
            raise RDSDownloadError(
                f"RDS artefact download failed (HTTP {response.status_code})"
            )
        content_type = response.headers.get("content-type", "")
        if _looks_textual(content_type):
            head = response.text[:400].lstrip().lower()
            if head.startswith("<!doctype html") or head.startswith("<html"):
                raise RDSDownloadError(
                    "RDS returned an HTML page instead of a data file; the link "
                    "is probably a login redirect"
                )
        if not response.content:
            raise RDSDownloadError("RDS returned an empty file")

        directory = Path(dest_dir)
        directory.mkdir(parents=True, exist_ok=True)
        name = file_url.rstrip("/").rsplit("/", 1)[-1] or "download.nc"
        if not name.lower().endswith((".nc", ".nc4", ".grib", ".grib2", ".zip")):
            name = f"{name}.bin"
        path = directory / name
        path.write_bytes(response.content)
        digest = hashlib.sha256(response.content).hexdigest()
        (directory / f"{name}.sha256").write_text(
            f"{digest}  {name}\n", encoding="utf-8"
        )
        logger.info(
            "RDS artefact downloaded",
            extra={"path": str(path), "bytes": len(response.content), "sha256": digest},
        )
        return path

    def run_sample(
        self, request: dict[str, Any], dest_dir: str | Path, **wait_kwargs: Any
    ) -> tuple[RDSJob, Path]:
        """Submit, wait for and download one sample in a single call."""
        job = self.wait_for_job(self.submit_job(request), **wait_kwargs)
        path = self.download(job.file_url or "", dest_dir)
        return job, path


    # --------------------------------------------------------------- health
    def availability(self, *, probe: bool = False) -> AccessStatus:
        """Report what this client can do right now.

        With ``probe=True`` and credentials present this actually attempts a
        login to prove the credentials work. Without credentials it reports the
        blocker without contacting anything that needs auth.
        """
        details = {
            "base_url": self.base_url,
            "credentials": self.credentials.public_status(),
            "public_metadata": "GET /datasets/catalog requires no authentication",
            "gated_endpoints": ["POST /jobs", "GET /jobs/my", "GET /jobs/id/{id}"],
            "registration": "https://rds.ncmrwf.gov.in/",
        }
        if not self.credentials.present:
            return AccessStatus(
                source="NCMRWF RDS (IMDAA/MERA)",
                availability=Availability.NEEDS_CREDENTIALS,
                reason=(
                    "IMDAA/MERA bulk downloads require a registered RDS account; "
                    f"set {RDS_EMAIL_ENV} and {RDS_PASSWORD_ENV}"
                ),
                details=details,
                manual_instructions=(
                    "1) Register at https://rds.ncmrwf.gov.in/ (email + password). "
                    f"2) Set {RDS_EMAIL_ENV} and {RDS_PASSWORD_ENV} in the "
                    "environment (never in source control). "
                    "3) Re-run python -m app.ingestion.realtime.imdaa_acquire."
                ),
                is_synthetic=False,
            )
        if not probe:
            return AccessStatus(
                source="NCMRWF RDS (IMDAA/MERA)",
                availability=Availability.NOT_PROBED,
                reason=(
                    "credentials are configured but the RDS login was not "
                    "attempted in this response"
                ),
                details=details,
                is_synthetic=False,
            )
        try:
            self._require_token()
        except RDSDownloadError as exc:
            return AccessStatus(
                source="NCMRWF RDS (IMDAA/MERA)",
                availability=Availability.NEEDS_CREDENTIALS,
                reason=str(exc),
                details=details,
                is_synthetic=False,
            )
        return AccessStatus(
            source="NCMRWF RDS (IMDAA/MERA)",
            availability=Availability.AVAILABLE,
            reason="RDS login succeeded; IMDAA/MERA job submission is authorised",
            details={**details, "token_chars": len(self._token or "")},
            is_synthetic=False,
        )

    def public_status(self) -> dict[str, Any]:
        """State safe to log or persist. Never includes the token value."""
        return {
            "base_url": self.base_url,
            "authenticated": self._token is not None,
            "token_chars": len(self._token) if self._token else 0,
            "credentials": self.credentials.public_status(),
        }

