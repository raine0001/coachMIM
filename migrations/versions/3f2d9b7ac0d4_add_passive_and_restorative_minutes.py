"""add passive and restorative minute fields to segmented checkins

Revision ID: 3f2d9b7ac0d4
Revises: 8f41d7ec4c22
Create Date: 2026-02-20 07:05:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "3f2d9b7ac0d4"
down_revision = "8f41d7ec4c22"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("daily_checkins", sa.Column("morning_passive_screen_minutes", sa.Integer(), nullable=True))
    op.add_column("daily_checkins", sa.Column("morning_restorative_minutes", sa.Integer(), nullable=True))
    op.add_column("daily_checkins", sa.Column("midday_passive_screen_minutes", sa.Integer(), nullable=True))
    op.add_column("daily_checkins", sa.Column("midday_restorative_minutes", sa.Integer(), nullable=True))
    op.add_column("daily_checkins", sa.Column("evening_passive_screen_minutes", sa.Integer(), nullable=True))
    op.add_column("daily_checkins", sa.Column("evening_restorative_minutes", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("daily_checkins", "evening_restorative_minutes")
    op.drop_column("daily_checkins", "evening_passive_screen_minutes")
    op.drop_column("daily_checkins", "midday_restorative_minutes")
    op.drop_column("daily_checkins", "midday_passive_screen_minutes")
    op.drop_column("daily_checkins", "morning_restorative_minutes")
    op.drop_column("daily_checkins", "morning_passive_screen_minutes")
