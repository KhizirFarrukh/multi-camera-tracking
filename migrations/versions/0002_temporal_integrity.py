"""Carry the temporal-integrity verdict and the recomputation flag.

Nullable rather than defaulted to an empty verdict: NULL means the route was
assembled without checking whether its timestamps were comparable, which is a
different claim from "checked and found clean". Collapsing the two would make
every trajectory written before stage 09 look verified.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-23
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the temporal-integrity verdict and the recomputation flag."""
    op.add_column(
        "trajectories",
        sa.Column("temporal_integrity", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "trajectories",
        sa.Column(
            "requires_recomputation",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    # Partial index: the flagged set is small and is what an operator queries
    # after correcting a clock. Indexing the false rows would be indexing
    # everything.
    op.create_index(
        "ix_trajectories_requires_recomputation",
        "trajectories",
        ["requires_recomputation"],
        postgresql_where=sa.text("requires_recomputation"),
    )


def downgrade() -> None:
    """Drop the index and both columns."""
    op.drop_index("ix_trajectories_requires_recomputation", table_name="trajectories")
    op.drop_column("trajectories", "requires_recomputation")
    op.drop_column("trajectories", "temporal_integrity")
