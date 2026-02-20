"""add user notifications and preferences

Revision ID: 9e1c2d7f4a55
Revises: 3f2d9b7ac0d4
Create Date: 2026-02-20 12:10:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9e1c2d7f4a55"
down_revision = "3f2d9b7ac0d4"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_notification_preferences",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("pause_notifications_until", sa.Date(), nullable=True),
        sa.Column("enable_morning_reminder", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enable_midday_reminder", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enable_evening_reminder", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enable_missing_data_alert", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enable_motivation_alert", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enable_reengagement_alert", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("allow_browser_push", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("allow_device_push", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("last_generated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index(
        "ix_user_notification_preferences_pause_notifications_until",
        "user_notification_preferences",
        ["pause_notifications_until"],
    )
    op.create_index(
        "ix_user_notification_preferences_last_generated_at",
        "user_notification_preferences",
        ["last_generated_at"],
    )

    op.create_table(
        "user_notifications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=180), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("action_url", sa.String(length=255), nullable=True),
        sa.Column("unique_key", sa.String(length=120), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "unique_key", name="uq_user_notifications_user_key"),
    )
    op.create_index("ix_user_notifications_user_id", "user_notifications", ["user_id"])
    op.create_index("ix_user_notifications_kind", "user_notifications", ["kind"])
    op.create_index("ix_user_notifications_is_read", "user_notifications", ["is_read"])
    op.create_index("ix_user_notifications_is_archived", "user_notifications", ["is_archived"])
    op.create_index("ix_user_notifications_is_deleted", "user_notifications", ["is_deleted"])
    op.create_index("ix_user_notifications_created_at", "user_notifications", ["created_at"])


def downgrade():
    op.drop_index("ix_user_notifications_created_at", table_name="user_notifications")
    op.drop_index("ix_user_notifications_is_deleted", table_name="user_notifications")
    op.drop_index("ix_user_notifications_is_archived", table_name="user_notifications")
    op.drop_index("ix_user_notifications_is_read", table_name="user_notifications")
    op.drop_index("ix_user_notifications_kind", table_name="user_notifications")
    op.drop_index("ix_user_notifications_user_id", table_name="user_notifications")
    op.drop_table("user_notifications")

    op.drop_index(
        "ix_user_notification_preferences_last_generated_at",
        table_name="user_notification_preferences",
    )
    op.drop_index(
        "ix_user_notification_preferences_pause_notifications_until",
        table_name="user_notification_preferences",
    )
    op.drop_table("user_notification_preferences")
