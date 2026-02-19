"""add admin tools and moderation fields

Revision ID: b9f327beaf2e
Revises: 6f4a8c2d9b11
Create Date: 2026-02-19 13:20:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b9f327beaf2e"
down_revision = "6f4a8c2d9b11"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("is_blocked", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("is_spam", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("last_active_at", sa.DateTime(), nullable=True))
        batch_op.create_index(batch_op.f("ix_users_last_active_at"), ["last_active_at"], unique=False)

    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("is_hidden", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("is_flagged", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("flag_reason", sa.Text(), nullable=True))
        batch_op.create_index(batch_op.f("ix_community_posts_is_hidden"), ["is_hidden"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_posts_is_flagged"), ["is_flagged"], unique=False)

    with op.batch_alter_table("community_comments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("is_hidden", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("is_flagged", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("flag_reason", sa.Text(), nullable=True))
        batch_op.create_index(batch_op.f("ix_community_comments_is_hidden"), ["is_hidden"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_comments_is_flagged"), ["is_flagged"], unique=False)

    op.create_table(
        "admin_users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=80), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_admin_users_username", "admin_users", ["username"], unique=False)

    op.create_table(
        "blocked_emails",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("reason", sa.String(length=80), nullable=True),
        sa.Column("created_by_admin_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["created_by_admin_id"], ["admin_users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_blocked_emails_email", "blocked_emails", ["email"], unique=False)
    op.create_index("ix_blocked_emails_created_by_admin_id", "blocked_emails", ["created_by_admin_id"], unique=False)

    op.create_table(
        "site_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=120), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_site_settings_key", "site_settings", ["key"], unique=False)


def downgrade():
    op.drop_index("ix_site_settings_key", table_name="site_settings")
    op.drop_table("site_settings")

    op.drop_index("ix_blocked_emails_created_by_admin_id", table_name="blocked_emails")
    op.drop_index("ix_blocked_emails_email", table_name="blocked_emails")
    op.drop_table("blocked_emails")

    op.drop_index("ix_admin_users_username", table_name="admin_users")
    op.drop_table("admin_users")

    with op.batch_alter_table("community_comments", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_community_comments_is_flagged"))
        batch_op.drop_index(batch_op.f("ix_community_comments_is_hidden"))
        batch_op.drop_column("flag_reason")
        batch_op.drop_column("is_flagged")
        batch_op.drop_column("is_hidden")

    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_community_posts_is_flagged"))
        batch_op.drop_index(batch_op.f("ix_community_posts_is_hidden"))
        batch_op.drop_column("flag_reason")
        batch_op.drop_column("is_flagged")
        batch_op.drop_column("is_hidden")

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_last_active_at"))
        batch_op.drop_column("last_active_at")
        batch_op.drop_column("is_spam")
        batch_op.drop_column("is_blocked")
