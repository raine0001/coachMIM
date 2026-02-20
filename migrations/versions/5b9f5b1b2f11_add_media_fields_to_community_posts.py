"""add media fields to community posts

Revision ID: 5b9f5b1b2f11
Revises: 0aa70a955feb
Create Date: 2026-02-20 11:15:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5b9f5b1b2f11"
down_revision = "0aa70a955feb"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("media_path", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("media_kind", sa.String(length=16), nullable=True))


def downgrade():
    with op.batch_alter_table("community_posts", schema=None) as batch_op:
        batch_op.drop_column("media_kind")
        batch_op.drop_column("media_path")

