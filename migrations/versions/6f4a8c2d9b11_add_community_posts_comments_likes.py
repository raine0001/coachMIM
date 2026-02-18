"""add community posts comments likes

Revision ID: 6f4a8c2d9b11
Revises: 1b8bb5f0b9a2
Create Date: 2026-02-18 14:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6f4a8c2d9b11"
down_revision = "1b8bb5f0b9a2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "community_posts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=False),
        sa.Column("title", sa.String(length=180), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_community_posts_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_posts_category"), ["category"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_posts_created_at"), ["created_at"], unique=False)

    op.create_table(
        "community_comments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["post_id"], ["community_posts.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("community_comments", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_community_comments_post_id"), ["post_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_comments_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_comments_created_at"), ["created_at"], unique=False)

    op.create_table(
        "community_likes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["post_id"], ["community_posts.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("post_id", "user_id", name="uq_community_likes_post_user"),
    )
    with op.batch_alter_table("community_likes", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_community_likes_post_id"), ["post_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_likes_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_community_likes_created_at"), ["created_at"], unique=False)


def downgrade():
    with op.batch_alter_table("community_likes", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_community_likes_created_at"))
        batch_op.drop_index(batch_op.f("ix_community_likes_user_id"))
        batch_op.drop_index(batch_op.f("ix_community_likes_post_id"))
    op.drop_table("community_likes")

    with op.batch_alter_table("community_comments", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_community_comments_created_at"))
        batch_op.drop_index(batch_op.f("ix_community_comments_user_id"))
        batch_op.drop_index(batch_op.f("ix_community_comments_post_id"))
    op.drop_table("community_comments")

    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_community_posts_created_at"))
        batch_op.drop_index(batch_op.f("ix_community_posts_category"))
        batch_op.drop_index(batch_op.f("ix_community_posts_user_id"))
    op.drop_table("community_posts")
