"""Persistence: SQLAlchemy models, sessions, mappers, and repositories.

The domain never imports from here. Dependencies point inward: repositories
depend on :mod:`multicam_tracker.models`, not the reverse, so the schema can
change without touching the vocabulary the rest of the system reasons in.
"""

from __future__ import annotations

from multicam_tracker.db.folding import fold_plate
from multicam_tracker.db.session import (
    dispose_engines,
    get_engine,
    get_session,
    get_session_factory,
    session_scope,
)

__all__ = [
    "dispose_engines",
    "fold_plate",
    "get_engine",
    "get_session",
    "get_session_factory",
    "session_scope",
]
