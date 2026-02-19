"""add support messages table

Revision ID: 7e2b3d1f0a9c
Revises: c1a42de0d119
Create Date: 2026-02-19 11:45:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "7e2b3d1f0a9c"
down_revision = "c1a42de0d119"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "support_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("sender_name", sa.String(length=255), nullable=False),
        sa.Column("sender_email", sa.String(length=255), nullable=False),
        sa.Column("subject", sa.String(length=180), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("attachment_path", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("admin_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("support_messages", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_support_messages_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_support_messages_sender_email"), ["sender_email"], unique=False)
        batch_op.create_index(batch_op.f("ix_support_messages_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_support_messages_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_support_messages_resolved_at"), ["resolved_at"], unique=False)


def downgrade():
    with op.batch_alter_table("support_messages", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_support_messages_resolved_at"))
        batch_op.drop_index(batch_op.f("ix_support_messages_created_at"))
        batch_op.drop_index(batch_op.f("ix_support_messages_status"))
        batch_op.drop_index(batch_op.f("ix_support_messages_sender_email"))
        batch_op.drop_index(batch_op.f("ix_support_messages_user_id"))
    op.drop_table("support_messages")
