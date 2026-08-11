"""Unit tests for :mod:`multicam_tracker.exceptions`.

The hierarchy's whole value is that one ``except MulticamTrackerError`` catches
everything the project raises. A subclass that accidentally inherits from
``Exception`` directly would slip through every such handler in the codebase, so
that property is asserted mechanically over the module's exports rather than
by hand.
"""

from __future__ import annotations

import pytest

from multicam_tracker import exceptions
from multicam_tracker.exceptions import (
    AuthorizationError,
    ConfigurationError,
    MulticamTrackerError,
    StorageError,
    TopologyError,
)

pytestmark = pytest.mark.unit

ALL_ERRORS = [getattr(exceptions, name) for name in exceptions.__all__]
SUBCLASSES = [error for error in ALL_ERRORS if error is not MulticamTrackerError]


@pytest.mark.parametrize("error_type", SUBCLASSES, ids=lambda cls: cls.__name__)
def test_exceptions__every_exported_error__subclasses_the_root(
    error_type: type[Exception],
) -> None:
    """Nothing in the hierarchy escapes a root-level except clause."""
    assert issubclass(error_type, MulticamTrackerError)


def test_exceptions__root__subclasses_builtin_exception() -> None:
    """The root is a normal exception, catchable by generic handlers."""
    assert issubclass(MulticamTrackerError, Exception)


def test_exceptions__all_is_complete() -> None:
    """__all__ lists every error class defined in the module."""
    defined = {
        name
        for name, value in vars(exceptions).items()
        if isinstance(value, type) and issubclass(value, MulticamTrackerError)
    }

    assert defined == set(exceptions.__all__)


def test_error__with_context__renders_context_in_str() -> None:
    """Context detail is visible in the message a human reads."""
    error = TopologyError("Unknown camera", {"camera_id": "cam_99"})

    rendered = str(error)

    assert "Unknown camera" in rendered
    assert "camera_id='cam_99'" in rendered


def test_error__with_multiple_context_keys__renders_them_in_sorted_order() -> None:
    """Ordering is deterministic so log assertions and diffs stay stable."""
    error = StorageError("Insert failed", {"table": "sightings", "attempt": 2})

    assert str(error) == "Insert failed (attempt=2, table='sightings')"


def test_error__without_context__renders_message_only() -> None:
    """No context means no trailing separator and no empty parentheses."""
    error = ConfigurationError("Something went wrong")

    rendered = str(error)

    assert rendered == "Something went wrong"
    assert not rendered.endswith(("(", ")", ":", " "))


def test_error__with_empty_context_dict__renders_message_only() -> None:
    """An explicitly empty mapping behaves the same as omitting it."""
    assert str(ConfigurationError("Boom", {})) == "Boom"


def test_error__context_defaults_to_empty_dict__never_none() -> None:
    """Callers can read .context unconditionally without a None check."""
    error = AuthorizationError("Denied")

    assert error.context == {}


def test_error__context_is_copied__mutating_the_source_does_not_leak() -> None:
    """The error keeps its own snapshot of the context it was raised with."""
    source = {"actor": "analyst_1"}
    error = AuthorizationError("Denied", source)
    source["actor"] = "someone_else"

    assert error.context == {"actor": "analyst_1"}


def test_error__repr__includes_message_and_context() -> None:
    """The repr is unambiguous for debugger and test-failure output."""
    error = StorageError("Insert failed", {"table": "sightings"})

    assert repr(error) == "StorageError(message='Insert failed', context={'table': 'sightings'})"


def test_error__raised_and_caught_as_root__preserves_the_subclass() -> None:
    """Catching broadly does not lose the specific type."""
    with pytest.raises(MulticamTrackerError) as excinfo:
        raise ConfigurationError("Bad config", {"field": "database.port"})

    assert isinstance(excinfo.value, ConfigurationError)
    assert excinfo.value.context["field"] == "database.port"


def test_error__message_attribute__is_the_bare_message() -> None:
    """.message stays clean for structured logging; str() carries the context."""
    error = TopologyError("Unknown camera", {"camera_id": "cam_99"})

    assert error.message == "Unknown camera"
    assert error.args == ("Unknown camera",)
