"""add mim chat messages

Revision ID: 1b8bb5f0b9a2
Revises: f5d21a9a6b31
Create Date: 2026-02-18 13:20:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1b8bb5f0b9a2"
down_revision = "f5d21a9a6b31"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "mim_chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("image_path", sa.String(length=500), nullable=True),
        sa.Column("context", sa.String(length=40), nullable=True),
        sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("mim_chat_messages", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_mim_chat_messages_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_mim_chat_messages_created_at"), ["created_at"], unique=False)


def downgrade():
    with op.batch_alter_table("mim_chat_messages", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_mim_chat_messages_created_at"))
        batch_op.drop_index(batch_op.f("ix_mim_chat_messages_user_id"))
    op.drop_table("mim_chat_messages")
