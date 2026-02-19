"""add goals and profile nudge flags

Revision ID: c1a42de0d119
Revises: b9f327beaf2e
Create Date: 2026-02-19 10:18:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c1a42de0d119"
down_revision = "b9f327beaf2e"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("user_profiles", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "profile_nudge_opt_out",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(sa.Column("profile_nudge_snooze_until", sa.Date(), nullable=True))

    with op.batch_alter_table("user_profiles", schema=None) as batch_op:
        batch_op.alter_column("profile_nudge_opt_out", server_default=None)

    op.create_table(
        "user_goals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=180), nullable=False),
        sa.Column("goal_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("target_value", sa.Float(), nullable=True),
        sa.Column("target_unit", sa.String(length=40), nullable=True),
        sa.Column("baseline_value", sa.Float(), nullable=True),
        sa.Column("baseline_unit", sa.String(length=40), nullable=True),
        sa.Column("constraints", sa.Text(), nullable=True),
        sa.Column("daily_commitment_minutes", sa.Integer(), nullable=True),
        sa.Column("coach_message", sa.Text(), nullable=True),
        sa.Column("today_action", sa.String(length=255), nullable=True),
        sa.Column("week_plan", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("user_goals", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_user_goals_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goals_goal_type"), ["goal_type"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goals_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goals_start_date"), ["start_date"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goals_target_date"), ["target_date"], unique=False)

    op.create_table(
        "user_goal_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("goal_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("is_done", sa.Boolean(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["user_goals.id"], ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("goal_id", "day", name="uq_user_goal_actions_goal_day"),
    )
    with op.batch_alter_table("user_goal_actions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_user_goal_actions_goal_id"), ["goal_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goal_actions_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_goal_actions_day"), ["day"], unique=False)


def downgrade():
    with op.batch_alter_table("user_goal_actions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_user_goal_actions_day"))
        batch_op.drop_index(batch_op.f("ix_user_goal_actions_user_id"))
        batch_op.drop_index(batch_op.f("ix_user_goal_actions_goal_id"))
    op.drop_table("user_goal_actions")

    with op.batch_alter_table("user_goals", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_user_goals_target_date"))
        batch_op.drop_index(batch_op.f("ix_user_goals_start_date"))
        batch_op.drop_index(batch_op.f("ix_user_goals_status"))
        batch_op.drop_index(batch_op.f("ix_user_goals_goal_type"))
        batch_op.drop_index(batch_op.f("ix_user_goals_user_id"))
    op.drop_table("user_goals")

    with op.batch_alter_table("user_profiles", schema=None) as batch_op:
        batch_op.drop_column("profile_nudge_snooze_until")
        batch_op.drop_column("profile_nudge_opt_out")
