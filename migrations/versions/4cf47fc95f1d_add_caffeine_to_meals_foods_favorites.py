"""add caffeine to meals foods favorites

Revision ID: 4cf47fc95f1d
Revises: e68a68792a9a
Create Date: 2026-02-18 13:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "4cf47fc95f1d"
down_revision = "e68a68792a9a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("food_items", schema=None) as batch_op:
        batch_op.add_column(sa.Column("caffeine_mg", sa.Float(), nullable=True))

    with op.batch_alter_table("meals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("caffeine_mg", sa.Float(), nullable=True))

    with op.batch_alter_table("favorite_meals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("caffeine_mg", sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table("favorite_meals", schema=None) as batch_op:
        batch_op.drop_column("caffeine_mg")

    with op.batch_alter_table("meals", schema=None) as batch_op:
        batch_op.drop_column("caffeine_mg")

    with op.batch_alter_table("food_items", schema=None) as batch_op:
        batch_op.drop_column("caffeine_mg")
