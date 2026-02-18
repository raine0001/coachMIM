"""add application-layer encryption columns

Revision ID: f5d21a9a6b31
Revises: 4cf47fc95f1d
Create Date: 2026-02-18 12:35:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f5d21a9a6b31"
down_revision = "4cf47fc95f1d"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_dek", sa.LargeBinary(), nullable=True))

    with op.batch_alter_table("user_profiles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_sensitive_payload", sa.LargeBinary(), nullable=True))

    with op.batch_alter_table("daily_checkins", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True))

    with op.batch_alter_table("meals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True))

    with op.batch_alter_table("favorite_meals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True))

    with op.batch_alter_table("substances", schema=None) as batch_op:
        batch_op.add_column(sa.Column("encrypted_payload", sa.LargeBinary(), nullable=True))


def downgrade():
    with op.batch_alter_table("substances", schema=None) as batch_op:
        batch_op.drop_column("encrypted_payload")

    with op.batch_alter_table("favorite_meals", schema=None) as batch_op:
        batch_op.drop_column("encrypted_payload")

    with op.batch_alter_table("meals", schema=None) as batch_op:
        batch_op.drop_column("encrypted_payload")

    with op.batch_alter_table("daily_checkins", schema=None) as batch_op:
        batch_op.drop_column("encrypted_payload")

    with op.batch_alter_table("user_profiles", schema=None) as batch_op:
        batch_op.drop_column("encrypted_sensitive_payload")

    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("encrypted_dek")
