"""Unit tests for :mod:`multicam_tracker.db.session`.

These use a recording double rather than a database. The behaviour under test is
the transaction discipline of the context manager -- when it commits, when it
rolls back, and which exception survives -- and none of that needs Postgres to
verify.
"""

from __future__ import annotations

from typing import Any

import pytest

from multicam_tracker.db.session import session_scope

pytestmark = pytest.mark.unit


class RecordingSession:
    """A stand-in for a SQLAlchemy session that records what was called.

    Args:
        fail_on: Method names that should raise when called, simulating a
            connection lost during cleanup.
    """

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on or set()

    def _record(self, name: str) -> None:
        """Record a call and raise if this method is configured to fail.

        Args:
            name: The method being called.

        Raises:
            RuntimeError: If ``name`` is in the configured failure set.
        """
        self.calls.append(name)
        if name in self._fail_on:
            msg = f"{name} failed"
            raise RuntimeError(msg)

    def commit(self) -> None:
        """Record a commit."""
        self._record("commit")

    def rollback(self) -> None:
        """Record a rollback."""
        self._record("rollback")

    def close(self) -> None:
        """Record a close."""
        self._record("close")


def _factory(session: Any) -> Any:
    """Return a session factory yielding ``session``.

    Args:
        session: The session to hand out.

    Returns:
        A zero-argument callable returning ``session``.
    """
    return lambda: session


def test_session_scope__clean_exit__commits_then_closes() -> None:
    """The success path commits exactly once and always closes."""
    session = RecordingSession()

    with session_scope(_factory(session)) as scoped:
        assert scoped is session

    assert session.calls == ["commit", "close"]


def test_session_scope__body_raises__rolls_back_and_does_not_commit() -> None:
    """A failed unit of work must leave nothing behind."""
    session = RecordingSession()

    with pytest.raises(ValueError, match="boom"), session_scope(_factory(session)):
        raise ValueError("boom")

    assert session.calls == ["rollback", "close"]
    assert "commit" not in session.calls


def test_session_scope__body_raises__propagates_the_original_exception() -> None:
    """The caller sees their own error, not a wrapped or replaced one."""
    session = RecordingSession()

    with pytest.raises(KeyError) as excinfo, session_scope(_factory(session)):
        raise KeyError("missing_camera")

    assert "missing_camera" in str(excinfo.value)


def test_session_scope__rollback_itself_fails__original_exception_still_wins() -> None:
    """A dropped connection during cleanup must not hijack the diagnosis.

    Replacing a UniqueViolation with an InterfaceError from the rollback would
    send whoever reads the log chasing the wrong problem entirely.
    """
    session = RecordingSession(fail_on={"rollback"})

    with pytest.raises(ValueError, match="the real problem"), session_scope(_factory(session)):
        raise ValueError("the real problem")

    assert session.calls == ["rollback", "close"]


def test_session_scope__close_fails_while_unwinding__original_exception_still_wins() -> None:
    """Same rule for the close in the finally block."""
    session = RecordingSession(fail_on={"rollback", "close"})

    with pytest.raises(ValueError, match="the real problem"), session_scope(_factory(session)):
        raise ValueError("the real problem")


def test_session_scope__close_fails_on_the_success_path__surfaces_the_error() -> None:
    """With no original error to protect, a failing close is the news."""
    session = RecordingSession(fail_on={"close"})

    with pytest.raises(RuntimeError, match="close failed"), session_scope(_factory(session)):
        pass


def test_session_scope__commit_fails__rolls_back_and_closes() -> None:
    """A commit that fails still has to release the connection."""
    session = RecordingSession(fail_on={"commit"})

    with pytest.raises(RuntimeError, match="commit failed"), session_scope(_factory(session)):
        pass

    assert session.calls == ["commit", "rollback", "close"]


def test_session_scope__closes_on_both_paths() -> None:
    """Stated explicitly because leaking a pooled connection is silent until exhaustion."""
    success, failure = RecordingSession(), RecordingSession()

    with session_scope(_factory(success)):
        pass
    with pytest.raises(ValueError, match="x"), session_scope(_factory(failure)):
        raise ValueError("x")

    assert "close" in success.calls
    assert "close" in failure.calls


def test_session_scope__generator_exit__still_rolls_back_and_closes() -> None:
    """An abandoned scope must not leave a transaction open."""
    session = RecordingSession()

    scope = session_scope(_factory(session))
    scope.__enter__()
    scope.__exit__(GeneratorExit, GeneratorExit(), None)

    assert "close" in session.calls
