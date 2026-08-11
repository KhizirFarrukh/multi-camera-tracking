"""Unit tests for :mod:`multicam_tracker.logging_config`.

Two properties matter enough to pin down here. First, JSON mode must emit
records a log aggregator can parse -- a single stray non-JSON line breaks
ingestion for the whole stream. Second, the correlation id must be isolated per
async context: stage 15 runs several camera streams concurrently, and an id that
leaked between tasks would attribute one camera's sightings to another's run.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import pytest
import structlog

from multicam_tracker.logging_config import (
    CORRELATION_ID_KEY,
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    correlation_scope,
    get_correlation_id,
    get_logger,
)

pytestmark = pytest.mark.unit


def _records(stream: io.StringIO) -> list[dict[str, Any]]:
    """Parse every line written to ``stream`` as a JSON log record.

    Args:
        stream: The stream ``configure_logging`` was pointed at.

    Returns:
        One parsed mapping per emitted line.
    """
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# Logger construction and rendering
# ---------------------------------------------------------------------------


def test_get_logger__returns_a_bound_logger(log_stream: io.StringIO) -> None:
    """The returned object binds to a concrete stdlib BoundLogger on use."""
    configure_logging(json_output=True, stream=log_stream)

    logger = get_logger("multicam_tracker.test")

    assert isinstance(logger.bind(), structlog.stdlib.BoundLogger)
    assert logger.bind(camera_id="cam_03").info("bound") is None


def test_get_logger__assigned_before_configuration__uses_the_later_config(
    log_stream: io.StringIO,
) -> None:
    """Module-level loggers are created at import time, before configuration.

    The lazy proxy is what makes that safe: a logger materialized eagerly would
    have captured the unconfigured chain and never emitted into this stream.
    """
    logger = get_logger("multicam_tracker.test")

    configure_logging(json_output=True, stream=log_stream, level="DEBUG")
    logger.info("configured_afterwards")

    assert _records(log_stream)[0]["event"] == "configured_afterwards"


def test_get_logger__bound_values__appear_on_every_record(log_stream: io.StringIO) -> None:
    """Values bound to a logger decorate each of its records."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    logger = get_logger("multicam_tracker.test").bind(source_id="clip_a")
    logger.info("first")
    logger.info("second")

    assert [record["source_id"] for record in _records(log_stream)] == ["clip_a", "clip_a"]


def test_configure_logging__json_mode__emits_parseable_json(log_stream: io.StringIO) -> None:
    """Every emitted line is a standalone JSON object."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    get_logger("multicam_tracker.test").info("sighting_recorded", camera_id="cam_03", count=2)

    records = _records(log_stream)
    assert len(records) == 1
    assert records[0]["event"] == "sighting_recorded"
    assert records[0]["camera_id"] == "cam_03"
    assert records[0]["count"] == 2


def test_configure_logging__json_mode__stamps_level_logger_and_timestamp(
    log_stream: io.StringIO,
) -> None:
    """The shared processor chain adds the fields an aggregator indexes on."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    get_logger("multicam_tracker.test").warning("threshold_exceeded")

    record = _records(log_stream)[0]
    assert record["level"] == "warning"
    assert record["logger"] == "multicam_tracker.test"
    assert record["timestamp"].endswith("Z")


def test_configure_logging__console_mode__emits_human_readable_text(
    log_stream: io.StringIO,
) -> None:
    """Dev mode trades machine-parseability for readability."""
    configure_logging(json_output=False, stream=log_stream, level="DEBUG", colors=False)

    get_logger("multicam_tracker.test").info("pipeline_started", source_id="clip_a")

    output = log_stream.getvalue()
    assert "pipeline_started" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.splitlines()[0])


def test_configure_logging__level_filtering__drops_records_below_the_threshold(
    log_stream: io.StringIO,
) -> None:
    """Boundary: the configured level is inclusive, the level below it is not."""
    configure_logging(json_output=True, stream=log_stream, level="WARNING")

    logger = get_logger("multicam_tracker.test")
    logger.info("suppressed")
    logger.warning("kept")

    records = _records(log_stream)
    assert [record["event"] for record in records] == ["kept"]


def test_configure_logging__called_twice__does_not_duplicate_output(
    log_stream: io.StringIO,
) -> None:
    """Reconfiguration replaces the handler instead of stacking another one."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    get_logger("multicam_tracker.test").info("once")

    assert len(_records(log_stream)) == 1


def test_configure_logging__third_party_stdlib_logger__is_routed_through_structlog(
    log_stream: io.StringIO,
) -> None:
    """Library logs land in the same stream and the same shape as ours.

    The logger name is deliberately one no real dependency owns. Naming an
    actual library here would couple the test to that library's own logging
    setup -- SQLAlchemy, for one, pins its logger to WARNING on import, which
    silently swallows an INFO record and fails this test only when the import
    happens to have occurred.
    """
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    logging.getLogger("some_third_party.client").info("connected")

    record = _records(log_stream)[0]
    assert record["event"] == "connected"
    assert record["logger"] == "some_third_party.client"


def test_configure_logging__exception_info__is_rendered_into_the_record(
    log_stream: io.StringIO,
) -> None:
    """A traceback must survive into the structured record, not vanish."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("multicam_tracker.test").exception("ingest_failed")

    record = _records(log_stream)[0]
    assert record["event"] == "ingest_failed"
    assert "ValueError: boom" in record["exception"]


# ---------------------------------------------------------------------------
# Correlation id
# ---------------------------------------------------------------------------


def test_bind_correlation_id__then_log__id_appears_in_subsequent_records(
    log_stream: io.StringIO,
) -> None:
    """One binding decorates every later record from the same context."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")
    bind_correlation_id("run-abc")

    logger = get_logger("multicam_tracker.test")
    logger.info("first")
    logger.info("second")

    records = _records(log_stream)
    assert [record[CORRELATION_ID_KEY] for record in records] == ["run-abc", "run-abc"]


def test_bind_correlation_id__without_argument__generates_one() -> None:
    """Callers that do not care about the value still get traceability."""
    generated = bind_correlation_id()

    assert generated
    assert get_correlation_id() == generated


def test_get_correlation_id__nothing_bound__returns_none() -> None:
    """Absence is reported as None rather than an empty string or a KeyError."""
    assert get_correlation_id() is None


def test_clear_correlation_id__after_binding__removes_it(log_stream: io.StringIO) -> None:
    """Clearing stops the id from decorating later records."""
    configure_logging(json_output=True, stream=log_stream, level="DEBUG")
    bind_correlation_id("run-abc")
    clear_correlation_id()

    get_logger("multicam_tracker.test").info("after_clear")

    assert CORRELATION_ID_KEY not in _records(log_stream)[0]


def test_clear_correlation_id__nothing_bound__is_a_no_op() -> None:
    """Idempotent teardown: clearing twice must not raise."""
    clear_correlation_id()
    clear_correlation_id()

    assert get_correlation_id() is None


def test_correlation_scope__on_exit__restores_the_previous_id() -> None:
    """A nested scope must not clobber the enclosing request's id."""
    bind_correlation_id("outer")

    with correlation_scope("inner") as scoped:
        assert scoped == "inner"
        assert get_correlation_id() == "inner"

    assert get_correlation_id() == "outer"


def test_correlation_scope__with_no_previous_id__clears_on_exit() -> None:
    """Exiting an outermost scope leaves no residue behind."""
    with correlation_scope("only"):
        assert get_correlation_id() == "only"

    assert get_correlation_id() is None


def test_correlation_scope__raising_inside__still_restores() -> None:
    """Restoration happens on the error path too."""
    bind_correlation_id("outer")

    with pytest.raises(RuntimeError), correlation_scope("inner"):
        raise RuntimeError("boom")

    assert get_correlation_id() == "outer"


async def test_bind_correlation_id__concurrent_tasks__ids_are_isolated() -> None:
    """Stage 15 runs cameras concurrently; their ids must not cross-contaminate."""
    observed: dict[str, str | None] = {}

    async def worker(name: str, correlation_id: str) -> None:
        bind_correlation_id(correlation_id)
        await asyncio.sleep(0)
        observed[name] = get_correlation_id()

    await asyncio.gather(worker("a", "run-a"), worker("b", "run-b"))

    assert observed == {"a": "run-a", "b": "run-b"}
    # The bindings made inside the tasks do not escape into the parent context.
    assert get_correlation_id() is None


async def test_bind_correlation_id__parent_binding__is_inherited_by_child_tasks() -> None:
    """A run-level id set before fan-out still tags every task's records."""
    bind_correlation_id("run-parent")

    async def child() -> str | None:
        await asyncio.sleep(0)
        return get_correlation_id()

    assert await asyncio.gather(child(), child()) == ["run-parent", "run-parent"]
