from datetime import date, datetime

from app import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    checkins = db.relationship("DailyCheckIn", backref="user", lazy=True)
    meals = db.relationship("Meal", backref="user", lazy=True)
    substances = db.relationship("Substance", backref="user", lazy=True)


class DailyCheckIn(db.Model):
    __tablename__ = "daily_checkins"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    day = db.Column(db.Date, default=date.today, index=True, nullable=False)

    sleep_hours = db.Column(db.Float, nullable=True)
    sleep_quality = db.Column(db.Integer, nullable=True)  # 1-10
    energy = db.Column(db.Integer, nullable=True)  # 1-10
    focus = db.Column(db.Integer, nullable=True)  # 1-10
    mood = db.Column(db.Integer, nullable=True)  # 1-10
    stress = db.Column(db.Integer, nullable=True)  # 1-10
    anxiety = db.Column(db.Integer, nullable=True)  # 1-10

    productivity = db.Column(db.Integer, nullable=True)  # 1-10
    accomplishments = db.Column(db.Text, nullable=True)
    notes = db.Column(db.Text, nullable=True)

    workout_timing = db.Column(db.String(120), nullable=True)
    workout_intensity = db.Column(db.Integer, nullable=True)  # 1-10
    alcohol_drinks = db.Column(db.Float, nullable=True)

    symptoms = db.Column(db.JSON, nullable=True)  # {"headache": 4, "stomach": 2}
    digestion = db.Column(db.JSON, nullable=True)  # {"bm_count": 1, "issues": ["bloat"]}

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.UniqueConstraint("user_id", "day", name="uq_daily_checkin_user_day"),)


class Meal(db.Model):
    __tablename__ = "meals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    eaten_at = db.Column(db.DateTime, index=True, nullable=False)
    label = db.Column(db.String(120), nullable=True)
    description = db.Column(db.Text, nullable=True)

    portion_notes = db.Column(db.String(255), nullable=True)
    tags = db.Column(db.JSON, nullable=True)  # ["processed", "high_carb", "salty"]

    calories = db.Column(db.Integer, nullable=True)
    protein_g = db.Column(db.Float, nullable=True)
    carbs_g = db.Column(db.Float, nullable=True)
    fat_g = db.Column(db.Float, nullable=True)
    sugar_g = db.Column(db.Float, nullable=True)
    sodium_mg = db.Column(db.Float, nullable=True)

    photo_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Substance(db.Model):
    __tablename__ = "substances"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    taken_at = db.Column(db.DateTime, index=True, nullable=False)

    kind = db.Column(db.String(50), nullable=False)  # alcohol, caffeine, nicotine, etc.
    amount = db.Column(db.String(120), nullable=True)  # "2 beers", "300mg"
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
