"""add user push subscriptions

Revision ID: b4c2f5d1a6ef
Revises: 9e1c2d7f4a55
Create Date: 2026-02-20 12:35:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b4c2f5d1a6ef"
down_revision = "9e1c2d7f4a55"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_push_subscriptions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("endpoint", sa.Text(), nullable=False),
        sa.Column("p256dh", sa.String(length=255), nullable=False),
        sa.Column("auth", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("device_label", sa.String(length=120), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("fail_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("last_sent_at", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_error_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("endpoint"),
    )
    op.create_index("ix_user_push_subscriptions_user_id", "user_push_subscriptions", ["user_id"])
    op.create_index("ix_user_push_subscriptions_is_active", "user_push_subscriptions", ["is_active"])
    op.create_index("ix_user_push_subscriptions_last_seen_at", "user_push_subscriptions", ["last_seen_at"])
    op.create_index("ix_user_push_subscriptions_last_sent_at", "user_push_subscriptions", ["last_sent_at"])


def downgrade():
    op.drop_index("ix_user_push_subscriptions_last_sent_at", table_name="user_push_subscriptions")
    op.drop_index("ix_user_push_subscriptions_last_seen_at", table_name="user_push_subscriptions")
    op.drop_index("ix_user_push_subscriptions_is_active", table_name="user_push_subscriptions")
    op.drop_index("ix_user_push_subscriptions_user_id", table_name="user_push_subscriptions")
    op.drop_table("user_push_subscriptions")
