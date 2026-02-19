"""add user daily coach insights

Revision ID: 0aa70a955feb
Revises: 7e2b3d1f0a9c
Create Date: 2026-02-19 14:07:51.913282

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0aa70a955feb'
down_revision = '7e2b3d1f0a9c'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('user_daily_coach_insights',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('day', sa.Date(), nullable=False),
    sa.Column('context', sa.String(length=32), nullable=False),
    sa.Column('tip_title', sa.String(length=180), nullable=True),
    sa.Column('tip_text', sa.Text(), nullable=True),
    sa.Column('next_action', sa.String(length=255), nullable=True),
    sa.Column('recommended_post_ids', sa.JSON(), nullable=True),
    sa.Column('source', sa.String(length=20), nullable=False),
    sa.Column('model_name', sa.String(length=80), nullable=True),
    sa.Column('encrypted_payload', sa.LargeBinary(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'day', 'context', name='uq_user_daily_coach_insight_user_day_context')
    )
    with op.batch_alter_table('user_daily_coach_insights', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_user_daily_coach_insights_context'), ['context'], unique=False)
        batch_op.create_index(batch_op.f('ix_user_daily_coach_insights_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_user_daily_coach_insights_day'), ['day'], unique=False)
        batch_op.create_index(batch_op.f('ix_user_daily_coach_insights_source'), ['source'], unique=False)
        batch_op.create_index(batch_op.f('ix_user_daily_coach_insights_user_id'), ['user_id'], unique=False)


def downgrade():
    with op.batch_alter_table('user_daily_coach_insights', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_user_daily_coach_insights_user_id'))
        batch_op.drop_index(batch_op.f('ix_user_daily_coach_insights_source'))
        batch_op.drop_index(batch_op.f('ix_user_daily_coach_insights_day'))
        batch_op.drop_index(batch_op.f('ix_user_daily_coach_insights_created_at'))
        batch_op.drop_index(batch_op.f('ix_user_daily_coach_insights_context'))

    op.drop_table('user_daily_coach_insights')
