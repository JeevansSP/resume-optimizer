"""add user_ai_configs table

Revision ID: j1k2l3m4n5o6
Revises: i0j1k2l3m4n5
Create Date: 2026-04-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "j1k2l3m4n5o6"
down_revision: Union[str, None] = "i0j1k2l3m4n5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_provider_enum = postgresql.ENUM(
    "PLATFORM_GEMINI", "GEMINI", "OPENAI", "ANTHROPIC", "CUSTOM_OPENAI_COMPATIBLE",
    name="aiprovider",
    create_type=False,
)


def upgrade() -> None:
    # Create enum type idempotently — sa.Enum create_type=False doesn't work with asyncpg
    op.execute(
        "DO $$ BEGIN "
        "CREATE TYPE aiprovider AS ENUM "
        "('PLATFORM_GEMINI', 'GEMINI', 'OPENAI', 'ANTHROPIC', 'CUSTOM_OPENAI_COMPATIBLE'); "
        "EXCEPTION WHEN duplicate_object THEN NULL; "
        "END $$"
    )
    op.create_table(
        "user_ai_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("provider", _provider_enum, nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("api_host", sa.String(), nullable=True),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_user_ai_config_user_id"),
    )


def downgrade() -> None:
    op.drop_table("user_ai_configs")
    op.execute("DROP TYPE IF EXISTS aiprovider")
