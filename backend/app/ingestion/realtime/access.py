"""Credential handling and availability probing for real-world data sources.

Design rules enforced here
--------------------------
* **No credential ever reaches a log, an exception message, or a provenance
  record.** :func:`redact` is the single place secrets are masked, and
  :class:`Credentials.__repr__` is overridden so an accidental f-string cannot
  leak one.
* **Missing credentials are an explicit, actionable error**, never a silent
  fallback to synthetic data.
* Availability is *probed*, not assumed. A connector reports an
  :class:`AccessStatus` describing what it needs and what it actually got, so
  ``/health`` can tell the truth about live feeds.

The access facts encoded here were verified against the live services; see
``docs/phase5_data_sources.md`` for the evidence and dates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.logging_conf import get_logger

logger = get_logger("ingestion.realtime.access")

#: Shared reason for connectors that are intentionally absent in this build.
NOT_IMPLEMENTED_REASON = (
    "adapter interface defined; automated retrieval not verified in this build"
)

__all__ = [
    "AccessStatus",
    "Availability",
    "NOT_IMPLEMENTED_REASON",
    "CredentialMissing",
    "Credentials",
    "SourceAccessError",
    "credential_status",
    "redact",
]


class Availability(str, Enum):
    """Outcome of an availability probe.

    ``AVAILABLE`` is only ever set after a real request succeeded. Everything
    else is a named, reportable state.
    """

    AVAILABLE = "available"
    #: Reachable, but this deployment lacks the credentials to use it.
    NEEDS_CREDENTIALS = "needs_credentials"
    #: Reachable, but the operator has not supplied the local files yet.
    NEEDS_MANUAL_DOWNLOAD = "needs_manual_download"
    #: Network/DNS/TLS failure or an unexpected response.
    UNREACHABLE = "unreachable"
    #: Public metadata read succeeded, bulk data does not.
    METADATA_ONLY = "metadata_only"
    #: Deliberately not attempted in this build.
    NOT_IMPLEMENTED = "not_implemented"
    #: A real, keyless source that this report has not just contacted. Distinct
    #: from "needs manual download": nothing is required from the operator.
    NOT_PROBED = "not_probed"


@dataclass(frozen=True, slots=True)
class Credentials:
    """A username/password pair sourced from environment variables.

    The values are held in memory only for the duration of a request. The
    ``__repr__`` override guarantees an interpolated ``Credentials`` prints
    ``<redacted>`` rather than the secret.
    """

    username: str = ""
    password: str = ""
    #: Names of the environment variables the values came from (safe to record).
    username_env: str = ""
    password_env: str = ""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Credentials(username={redact(self.username)}, "
            f"password={redact(self.password)}, "
            f"username_env={self.username_env!r}, password_env={self.password_env!r})"
        )

    __str__ = __repr__

    @property
    def present(self) -> bool:
        return bool(self.username.strip() and self.password.strip())

    def require(self, *, source: str, env_hint: tuple[str, str]) -> "Credentials":
        """Return self, or raise :class:`CredentialMissing` with next steps."""
        if self.present:
            return self
        user_env, pass_env = env_hint
        raise CredentialMissing(
            f"{source} requires an approved account. Set {user_env} and {pass_env} "
            f"in the environment (never in source control), then re-run ingestion. "
            f"Registration: https://mosdac.gov.in/signup/ - accounts must be approved "
            f"before downloads are permitted."
        )

    def public_status(self) -> dict[str, Any]:
        """Credential state safe for ``/health`` and provenance records."""
        return {
            "username_env": self.username_env,
            "password_env": self.password_env,
            "username": redact(self.username),
            "password": redact(self.password),
            "present": self.present,
        }


def credential_status(
    username_env: str, password_env: str, *, overrides: Credentials | None = None
) -> Credentials:
    """Read credentials from the environment (or accept an explicit override).

    Used by tests and by callers that already hold a credentials object; the
    environment remains the only supported production path.
    """
    if overrides is not None:
        return overrides
    return Credentials(
        username=os.getenv(username_env, "").strip(),
        password=os.getenv(password_env, ""),
        username_env=username_env,
        password_env=password_env,
    )


@dataclass(slots=True)
class AccessStatus:
    """What a connector needs, what it has, and what it obtained."""

    source: str
    availability: Availability
    reason: str
    #: Free-form, non-sensitive detail (endpoints probed, HTTP codes, ...).
    details: dict[str, Any] = field(default_factory=dict)
    #: The concrete manual step a human must take, when there is one.
    manual_instructions: str | None = None
    is_synthetic: bool = False

    @property
    def available(self) -> bool:
        return self.availability is Availability.AVAILABLE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "availability": self.availability.value,
            "available": self.available,
            "reason": self.reason,
            "is_synthetic": self.is_synthetic,
        }
        if self.details:
            payload["details"] = self.details
        if self.manual_instructions:
            payload["manual_instructions"] = self.manual_instructions
        return payload

    METADATA_ONLY = "metadata_only"
    #: Deliberately not attempted in this build.
    NOT_IMPLEMENTED = "not_implemented"
    #: A real, keyless source that this report has not just contacted. Distinct
    #: from "needs manual download": nothing is required from the operator.
    NOT_PROBED = "not_probed"


class SourceAccessError(RuntimeError):
    """A real source could not be used, with an actionable reason.

    The message is safe to surface through the API: it names the *missing
    requirement*, never a credential value.
    """


class CredentialMissing(SourceAccessError):
    """Required credentials were not configured in the environment."""


def redact(value: str | None) -> str:
    """Mask a secret for display, keeping only a length hint."""
    if not value:
        return "<unset>"
    return f"<set:{len(value)} chars>"
