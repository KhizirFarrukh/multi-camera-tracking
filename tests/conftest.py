"""Shared pytest fixtures for every stage.

Fixtures live here (and in ``tests/fixtures/``) rather than being redefined per
test module, per the global contract's ``testing_standards.fixtures``.

The autouse fixtures below exist to make the suite hermetic. Configuration reads
from the process environment, the current working directory's ``.env``, and a
YAML file on disk -- three channels through which a developer's machine could
silently change a test result. Each is neutralised before every test.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import structlog

from multicam_tracker.clock import FixedClock
from multicam_tracker.config import Settings, reset_settings_cache

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root, independent of the working directory."""

REPO_THRESHOLDS_FILE = REPO_ROOT / "config" / "thresholds.yaml"
"""The real, committed thresholds file. Tests assert against its actual values."""

FROZEN_INSTANT = datetime(2026, 8, 10, 14, 22, 11, 500000, tzinfo=UTC)
"""A fixed reference instant. Matches the example sighting in the project plan."""


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Neutralise every host-specific configuration channel.

    Clears all ``MCT_`` environment variables, moves the working directory to a
    scratch path so a developer's ``.env`` is not discovered, and clears the
    settings cache on both sides of the test.

    Yields:
        ``None``; the isolation is the effect.
    """
    for name in [key for key in os.environ if key.startswith("MCT_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Restore root logger handlers and structlog context around each test.

    ``configure_logging`` replaces the root handlers by design. Without this,
    the first test to configure logging would leak its handler into every
    subsequent test's output.

    Yields:
        ``None``; the restoration is the effect.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


@pytest.fixture
def thresholds_file() -> Path:
    """Return the path to the committed ``config/thresholds.yaml``.

    Returns:
        Absolute path to the real thresholds file.
    """
    return REPO_THRESHOLDS_FILE


@pytest.fixture
def custom_thresholds_file(tmp_path: Path) -> Path:
    """Write a thresholds YAML with non-default values and return its path.

    Values differ from the committed defaults so a test can prove the file was
    actually read rather than coincidentally matching.

    Returns:
        Path to the temporary YAML file.
    """
    path = tmp_path / "custom_thresholds.yaml"
    path.write_text(
        "\n".join(
            [
                "plate_auto_accept_min_confidence: 0.70",
                "plate_fuzzy_max_edit_distance: 1",
                "embedding_auto_accept_min_similarity: 0.88",
                "embedding_review_min_similarity: 0.60",
                "hop_implausible_penalty: 0.25",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def build_settings(thresholds_file: Path) -> Callable[..., Settings]:
    """Return a factory that builds :class:`Settings` in a hermetic way.

    The factory pins the thresholds file to the committed one and disables
    ``.env`` discovery, so a test only sees the overrides it passes explicitly.

    Returns:
        A callable accepting keyword overrides and returning a ``Settings``.
    """

    def _build(**overrides: Any) -> Settings:
        overrides.setdefault("thresholds_file", thresholds_file)
        return Settings(_env_file=None, **overrides)

    return _build


@pytest.fixture
def fixed_clock() -> FixedClock:
    """Return a clock pinned to :data:`FROZEN_INSTANT`.

    Returns:
        A :class:`~multicam_tracker.clock.FixedClock`.
    """
    return FixedClock(FROZEN_INSTANT)


@pytest.fixture
def log_stream() -> io.StringIO:
    """Return an in-memory stream for capturing configured log output.

    Returns:
        An empty :class:`io.StringIO` to pass to ``configure_logging``.
    """
    return io.StringIO()
