"""add user coach signal responses

Revision ID: d2a1b4c9e5f7
Revises: b4c2f5d1a6ef
Create Date: 2026-02-25 13:45:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d2a1b4c9e5f7"
down_revision = "b4c2f5d1a6ef"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_coach_signal_responses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("context", sa.String(length=32), nullable=False),
        sa.Column("signal_key", sa.String(length=80), nullable=False),
        sa.Column("signal_title", sa.String(length=180), nullable=True),
        sa.Column("signal_message", sa.Text(), nullable=True),
        sa.Column("response_action", sa.String(length=20), nullable=False, server_default=sa.text("'note'")),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "day",
            "context",
            "signal_key",
            name="uq_user_coach_signal_responses_user_day_context_signal",
        ),
    )
    op.create_index("ix_user_coach_signal_responses_user_id", "user_coach_signal_responses", ["user_id"])
    op.create_index("ix_user_coach_signal_responses_day", "user_coach_signal_responses", ["day"])
    op.create_index("ix_user_coach_signal_responses_context", "user_coach_signal_responses", ["context"])
    op.create_index("ix_user_coach_signal_responses_signal_key", "user_coach_signal_responses", ["signal_key"])
    op.create_index(
        "ix_user_coach_signal_responses_response_action",
        "user_coach_signal_responses",
        ["response_action"],
    )
    op.create_index("ix_user_coach_signal_responses_created_at", "user_coach_signal_responses", ["created_at"])


def downgrade():
    op.drop_index("ix_user_coach_signal_responses_created_at", table_name="user_coach_signal_responses")
    op.drop_index("ix_user_coach_signal_responses_response_action", table_name="user_coach_signal_responses")
    op.drop_index("ix_user_coach_signal_responses_signal_key", table_name="user_coach_signal_responses")
    op.drop_index("ix_user_coach_signal_responses_context", table_name="user_coach_signal_responses")
    op.drop_index("ix_user_coach_signal_responses_day", table_name="user_coach_signal_responses")
    op.drop_index("ix_user_coach_signal_responses_user_id", table_name="user_coach_signal_responses")
    op.drop_table("user_coach_signal_responses")
