"""add home page visits table

Revision ID: 8f41d7ec4c22
Revises: 5b9f5b1b2f11
Create Date: 2026-02-20 02:25:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "8f41d7ec4c22"
down_revision = "5b9f5b1b2f11"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "home_page_visits",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("host", sa.String(length=120), nullable=False),
        sa.Column("visitor_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("is_authenticated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("user_agent_hash", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("day", "host", "visitor_hash", name="uq_home_page_visits_day_host_visitor"),
    )
    op.create_index(op.f("ix_home_page_visits_day"), "home_page_visits", ["day"], unique=False)
    op.create_index(op.f("ix_home_page_visits_host"), "home_page_visits", ["host"], unique=False)
    op.create_index(op.f("ix_home_page_visits_is_authenticated"), "home_page_visits", ["is_authenticated"], unique=False)
    op.create_index(op.f("ix_home_page_visits_last_seen_at"), "home_page_visits", ["last_seen_at"], unique=False)
    op.create_index(op.f("ix_home_page_visits_user_id"), "home_page_visits", ["user_id"], unique=False)
    op.create_index(op.f("ix_home_page_visits_visitor_hash"), "home_page_visits", ["visitor_hash"], unique=False)

def downgrade():
    op.drop_index(op.f("ix_home_page_visits_visitor_hash"), table_name="home_page_visits")
    op.drop_index(op.f("ix_home_page_visits_user_id"), table_name="home_page_visits")
    op.drop_index(op.f("ix_home_page_visits_last_seen_at"), table_name="home_page_visits")
    op.drop_index(op.f("ix_home_page_visits_is_authenticated"), table_name="home_page_visits")
    op.drop_index(op.f("ix_home_page_visits_host"), table_name="home_page_visits")
    op.drop_index(op.f("ix_home_page_visits_day"), table_name="home_page_visits")
    op.drop_table("home_page_visits")
