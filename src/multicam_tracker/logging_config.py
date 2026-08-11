"""Structured logging built on structlog.

Two output modes share one processor chain: machine-readable JSON for
production, and a coloured human-readable renderer for local development. The
chain is shared deliberately -- if dev and prod formatted through different
processors, a field present in one and absent in the other would only ever be
noticed in production.

Third-party libraries (SQLAlchemy, uvicorn, ultralytics) log through the stdlib
:mod:`logging` module. :func:`configure_logging` routes the stdlib root logger
through the same structlog formatter so those records land in the same stream,
in the same shape, instead of appearing as unstructured noise alongside the
JSON.

A ``correlation_id`` bound via :func:`bind_correlation_id` rides along on every
subsequent record from the same context. Because it lives in a
:class:`~contextvars.ContextVar`, concurrent asyncio tasks each get an isolated
copy: one pipeline run's id cannot leak into another's records.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import structlog

if TYPE_CHECKING:
    from typing import TextIO

    from structlog.typing import Processor

__all__ = [
    "CORRELATION_ID_KEY",
    "bind_correlation_id",
    "clear_correlation_id",
    "configure_logging",
    "correlation_scope",
    "get_correlation_id",
    "get_logger",
]

CORRELATION_ID_KEY = "correlation_id"
"""Context key under which the correlation id is bound onto every record."""

_DEFAULT_LEVEL = "INFO"


def _shared_processors() -> list[Processor]:
    """Return the processor chain applied to structlog and stdlib records alike.

    Returns:
        Processors that merge context variables and stamp logger name, level,
        and an ISO-8601 UTC timestamp onto every record.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(
    *,
    json_output: bool = True,
    level: str | int = _DEFAULT_LEVEL,
    stream: TextIO | None = None,
    colors: bool | None = None,
) -> None:
    """Configure structlog and the stdlib root logger.

    Safe to call repeatedly: existing root handlers are replaced, so tests can
    reconfigure between cases without accumulating duplicate output.

    Args:
        json_output: Emit one JSON object per record when true; emit the
            human-readable console renderer when false.
        level: Minimum level for the root logger, as a name or numeric level.
        stream: Destination stream. Defaults to :data:`sys.stdout`.
        colors: Force colours on or off for the console renderer. Defaults to
            autodetection based on whether the stream is a TTY. Ignored when
            ``json_output`` is true.

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    shared = _shared_processors()
    destination = stream if stream is not None else sys.stdout

    renderer: Processor
    if json_output:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        use_colors = colors if colors is not None else bool(getattr(destination, "isatty", bool)())
        renderer = structlog.dev.ConsoleRenderer(colors=use_colors)

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Not cached: a cached logger keeps the processor chain it was created
        # with, which would make a second configure_logging() call silently
        # ineffective for any logger already handed out. Tests reconfigure often.
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )

    handler = logging.StreamHandler(destination)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``.

    The object returned is structlog's lazy proxy, which materializes into a
    real ``BoundLogger`` on first use. That laziness is deliberate and load
    bearing: modules assign ``logger = get_logger(__name__)`` at import time,
    which happens before the application entry point calls
    :func:`configure_logging`. A logger materialized eagerly would capture the
    unconfigured processor chain and keep it forever.

    Args:
        name: Logger name, conventionally the calling module's ``__name__``.

    Returns:
        A logger that emits through whatever processor chain is configured at
        the moment it is first used. Typed as ``BoundLogger`` because that is
        the interface it presents; call ``.bind()`` to obtain the concrete
        instance.
    """
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation id onto the current context.

    Every record emitted from this context -- and from any asyncio task spawned
    after this call -- carries the id, which is what makes a single pipeline run
    or API request traceable across modules.

    Args:
        correlation_id: The id to bind. A random UUID4 hex string is generated
            when omitted.

    Returns:
        The correlation id that was bound.
    """
    resolved = correlation_id if correlation_id is not None else uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(**{CORRELATION_ID_KEY: resolved})
    return resolved


def get_correlation_id() -> str | None:
    """Return the correlation id bound to the current context, if any.

    Returns:
        The bound id, or ``None`` when no id is bound in this context.
    """
    bound: dict[str, Any] = structlog.contextvars.get_contextvars()
    value = bound.get(CORRELATION_ID_KEY)
    return value if isinstance(value, str) else None


def clear_correlation_id() -> None:
    """Unbind the correlation id from the current context.

    Does nothing when no id is bound.
    """
    structlog.contextvars.unbind_contextvars(CORRELATION_ID_KEY)


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation id for the duration of a ``with`` block.

    Restores whatever id was previously bound on exit, so nested scopes do not
    clobber an outer request's id.

    Args:
        correlation_id: The id to bind. A random UUID4 hex string is generated
            when omitted.

    Yields:
        The correlation id bound for the duration of the block.
    """
    previous = get_correlation_id()
    resolved = bind_correlation_id(correlation_id)
    try:
        yield resolved
    finally:
        if previous is None:
            clear_correlation_id()
        else:
            bind_correlation_id(previous)
