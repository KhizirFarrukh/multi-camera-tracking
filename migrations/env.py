"""Alembic environment.

The connection URL comes from :class:`~multicam_tracker.config.Settings`, not
from ``alembic.ini``. That guarantees a migration targets the same database the
application does, and keeps credentials out of a committed file.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from multicam_tracker.config import get_settings
from multicam_tracker.db.orm import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
"""Schema autogeneration compares against the ORM's metadata."""


def _database_url() -> str:
    """Return the connection URL for migrations.

    An explicit ``sqlalchemy.url`` set on the Alembic config wins, which is what
    lets the test suite point a migration at an ephemeral container without
    rewriting the process environment. Otherwise the DSN comes from settings.

    Returns:
        The connection URL, including credentials.
    """
    override = config.get_main_option("sqlalchemy.url", None)
    if override:
        return override
    return get_settings().database.dsn


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database.

    Used by ``alembic upgrade head --sql`` to review the DDL a migration would
    run, and by the test suite to verify the migration is well formed on a
    machine with no Postgres available.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
