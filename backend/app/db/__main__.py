"""``python -m app.db`` - apply or inspect schema migrations.

Added in Phase 4 so the dashboard can be started against a fresh database
without reaching into application internals::

    python -m app.db upgrade      # apply pending migrations
    python -m app.db current      # show the applied revision
    python -m app.db init         # create tables directly (no version record)
    python -m app.db downgrade --to 0001_initial_schema
"""

from __future__ import annotations

import argparse
import sys

from app.db.migrations import MIGRATIONS, applied_revisions, current_revision, downgrade, upgrade
from app.db.session import get_session_factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.db", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("upgrade", help="apply pending migrations")
    sub.add_parser("current", help="print the applied revision(s)")
    sub.add_parser("init", help="create tables without recording a revision")
    back = sub.add_parser("downgrade", help="revert to a target revision")
    back.add_argument("--to", required=True, help="target revision id")

    args = parser.parse_args(argv)
    engine = get_session_factory().engine

    if args.command == "upgrade":
        applied = upgrade(engine)
        print(f"applied: {applied or 'nothing (already up to date)'}")
    elif args.command == "current":
        revisions = applied_revisions(engine)
        print(f"current: {current_revision(engine) or 'none'}")
        for revision in revisions:
            print(f"  - {revision}")
    elif args.command == "init":
        get_session_factory().create_all()
        print("tables created")
    elif args.command == "downgrade":
        print(f"reverted: {downgrade(engine, args.to) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
