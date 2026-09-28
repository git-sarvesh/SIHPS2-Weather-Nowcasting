"""Engine / session management, plus a FastAPI session dependency.

Key behaviours
--------------
* ``SessionFactory`` owns one engine and one sessionmaker. Tests get their own
  factory over an in-memory or temp-file SQLite database.
* ``scope()`` is a context manager with **real** transaction semantics: commit on
  success, rollback on any exception. Callers never commit manually.
* Database problems surface as :class:`DatabaseUnavailableError` so routes can
  return ``503`` instead of a misleading success.
* SQLite gets ``PRAGMA foreign_keys=ON`` (off by default) so ``ON DELETE CASCADE``
  and the check constraints are actually enforced.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.logging_conf import get_logger

logger = get_logger("db.session")

__all__ = [
    "DatabaseUnavailableError",
    "SessionFactory",
    "get_session",
    "get_session_factory",
    "reset_session_factory",
    "session_scope",
]


class DatabaseUnavailableError(RuntimeError):
    """The database could not be reached, or a write violated a constraint."""


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """SQLite ignores foreign keys unless the pragma is set per connection."""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def redact_url(url: str) -> str:
    """Strip credentials from a database URL before it is logged or returned."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    _credentials, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}"


def _ensure_sqlite_directory(url: str) -> None:
    """Create the parent directory of a file-backed SQLite database.

    SQLite does not create intermediate directories, so a default URL like
    ``sqlite:///./data/sihps.db`` fails on a fresh checkout.
    """
    from pathlib import Path

    path_part = url.split("///", 1)[-1] if "///" in url else url.split("://", 1)[-1]
    if not path_part or path_part == ":memory:":
        return
    parent = Path(path_part).expanduser().parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


class SessionFactory:
    """Owns an engine and hands out sessions.

    Parameters
    ----------
    url:
        SQLAlchemy URL. Defaults to ``SIHPS_DATABASE_URL``.
    echo:
        Log every statement (development aid).
    engine:
        Use a pre-built engine (tests share one).
    """

    def __init__(
        self, url: str | None = None, *, echo: bool = False, engine: Engine | None = None
    ) -> None:
        settings = get_settings()
        self.url = url or settings.database_url
        if engine is not None:
            self.engine = engine
        else:
            kwargs: dict[str, Any] = {"echo": echo, "future": True}
            if self.url.startswith("sqlite"):
                # The API and a Celery worker may both hold connections.
                kwargs["connect_args"] = {"check_same_thread": False}
                if ":memory:" in self.url or self.url.endswith("sqlite://"):
                    # An in-memory database lives inside one connection, so the
                    # pool must hand out that same connection every time.
                    kwargs["poolclass"] = StaticPool
                else:
                    # A relative sqlite path needs its parent directory to exist,
                    # otherwise the first connect fails with "unable to open".
                    _ensure_sqlite_directory(self.url)
            else:
                # Postgres: recycle connections so a restart cannot use stale ones.
                kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)
            self.engine = create_engine(self.url, **kwargs)
        if self.engine.dialect.name == "sqlite":
            _enable_sqlite_foreign_keys(self.engine)
        self.sessionmaker = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False, future=True
        )

    # ------------------------------------------------------------------ util
    def health(self) -> dict[str, Any]:
        """Connectivity probe used by ``/health``; never raises."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "dialect": self.engine.dialect.name,
                "url": redact_url(self.url),
            }
        except SQLAlchemyError as exc:
            logger.warning("database health check failed", extra={"error": str(exc)})
            return {
                "status": "unavailable",
                "dialect": self.engine.dialect.name,
                "url": redact_url(self.url),
                "error": type(exc).__name__,
            }

    def create_all(self) -> None:
        """Create every table declared on :data:`app.db.base.Base`."""
        from app.db import models  # noqa: F401 - ensure models are registered
        from app.db.base import Base

        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """Drop every table (tests only)."""
        from app.db import models  # noqa: F401
        from app.db.base import Base

        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        """A new session. The caller is responsible for closing it."""
        return self.sessionmaker()

    @contextmanager
    def scope(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback on error."""
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


_FACTORY: SessionFactory | None = None


def get_session_factory(settings: Settings | None = None) -> SessionFactory:
    """Process-wide :class:`SessionFactory` singleton."""
    global _FACTORY
    if _FACTORY is None:
        _FACTORY = SessionFactory((settings or get_settings()).database_url)
    return _FACTORY


def reset_session_factory() -> None:
    """Drop the singleton (tests, and settings reloads)."""
    global _FACTORY
    if _FACTORY is not None:
        _FACTORY.dispose()
    _FACTORY = None


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Module-level transactional scope over the singleton factory."""
    with get_session_factory(settings).scope() as session:
        yield session


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a transactional session.

    Commits when the route returns normally and rolls back on any exception, so
    a handler that raises never leaves a half-written run behind. A driver-level
    failure becomes :class:`DatabaseUnavailableError` for the route to map to 503.
    """
    session = get_session_factory().session()
    try:
        yield session
        session.commit()
    except (OperationalError, IntegrityError) as exc:
        session.rollback()
        raise DatabaseUnavailableError(describe_error(exc)) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def describe_error(exc: SQLAlchemyError) -> str:
    """Short human message for a database error, without leaking the URL."""
    message = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
    return f"{type(exc).__name__}: {message.splitlines()[0][:300]}"
