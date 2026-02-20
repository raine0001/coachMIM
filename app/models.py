from datetime import date, datetime

from app import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(255), nullable=True)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    encrypted_dek = db.Column(db.LargeBinary, nullable=True)
    is_blocked = db.Column(db.Boolean, nullable=False, default=False)
    is_spam = db.Column(db.Boolean, nullable=False, default=False)
    last_active_at = db.Column(db.DateTime, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    profile = db.relationship("UserProfile", backref="user", uselist=False, lazy=True)
    checkins = db.relationship("DailyCheckIn", backref="user", lazy=True)
    meals = db.relationship("Meal", backref="user", lazy=True)
    substances = db.relationship("Substance", backref="user", lazy=True)
    favorite_meals = db.relationship("FavoriteMeal", backref="user", lazy=True)
    mim_chat_messages = db.relationship("MIMChatMessage", backref="user", lazy=True)
    community_posts = db.relationship("CommunityPost", backref="user", lazy=True)
    community_comments = db.relationship("CommunityComment", backref="user", lazy=True)
    community_likes = db.relationship("CommunityLike", backref="user", lazy=True)
    support_messages = db.relationship("SupportMessage", backref="user", lazy=True)
    home_page_visits = db.relationship("HomePageVisit", backref="user", lazy=True)
    goals = db.relationship("UserGoal", backref="user", lazy=True)
    goal_actions = db.relationship("UserGoalAction", backref="user", lazy=True)
    daily_coach_insights = db.relationship("UserDailyCoachInsight", backref="user", lazy=True)


class UserProfile(db.Model):
    __tablename__ = "user_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)

    age = db.Column(db.Integer, nullable=True)
    biological_sex = db.Column(db.String(32), nullable=True)
    time_zone = db.Column(db.String(64), nullable=True)
    unit_system = db.Column(db.String(12), nullable=True, default="imperial")
    phone = db.Column(db.String(32), nullable=True)

    height_cm = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)
    body_fat_pct = db.Column(db.Float, nullable=True)
    waist_cm = db.Column(db.Float, nullable=True)

    general_health_rating = db.Column(db.Integer, nullable=True)  # 1-10
    medical_conditions = db.Column(db.Text, nullable=True)
    known_sleep_issues = db.Column(db.Text, nullable=True)
    family_history_flags = db.Column(db.Text, nullable=True)
    medications = db.Column(db.Text, nullable=True)
    supplements = db.Column(db.Text, nullable=True)
    resting_blood_pressure = db.Column(db.String(32), nullable=True)

    fitness_level = db.Column(db.String(32), nullable=True)  # sedentary/light/moderate/intense
    typical_sleep_duration_hours = db.Column(db.Float, nullable=True)
    work_type = db.Column(db.String(32), nullable=True)
    work_stress_baseline = db.Column(db.Integer, nullable=True)  # 1-10
    typical_alcohol_frequency = db.Column(db.String(120), nullable=True)
    caffeine_baseline = db.Column(db.String(120), nullable=True)
    nicotine_use = db.Column(db.String(120), nullable=True)
    recreational_drug_use = db.Column(db.String(120), nullable=True)

    diet_style = db.Column(db.String(120), nullable=True)
    food_intolerances = db.Column(db.Text, nullable=True)
    food_sensitivities = db.Column(db.Text, nullable=True)
    typical_meal_timing = db.Column(db.String(120), nullable=True)
    cravings_patterns = db.Column(db.String(255), nullable=True)

    baseline_mood = db.Column(db.Integer, nullable=True)  # 1-10
    baseline_anxiety = db.Column(db.Integer, nullable=True)  # 1-10
    baseline_focus = db.Column(db.Integer, nullable=True)  # 1-10
    energy_consistency = db.Column(db.String(32), nullable=True)  # stable/fluctuates
    attention_issues = db.Column(db.String(255), nullable=True)
    emotional_volatility = db.Column(db.String(32), nullable=True)  # low/moderate/high
    burnout_history = db.Column(db.Text, nullable=True)

    primary_goal = db.Column(db.String(255), nullable=True)
    secondary_goals = db.Column(db.Text, nullable=True)
    time_horizon = db.Column(db.String(120), nullable=True)
    great_day_definition = db.Column(db.Text, nullable=True)

    chronotype = db.Column(db.String(32), nullable=True)
    digestive_sensitivity = db.Column(db.Text, nullable=True)
    stress_reactivity = db.Column(db.Text, nullable=True)
    social_pattern = db.Column(db.Text, nullable=True)
    screen_time_evening_hours = db.Column(db.Float, nullable=True)
    profile_nudge_opt_out = db.Column(db.Boolean, nullable=False, default=False)
    profile_nudge_snooze_until = db.Column(db.Date, nullable=True)
    encrypted_sensitive_payload = db.Column(db.LargeBinary, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def missing_required_fields(self):
        required = {
            "age": self.age,
            "biological_sex": self.biological_sex,
            "time_zone": self.time_zone,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "primary_goal": self.primary_goal,
            "fitness_level": self.fitness_level,
            "diet_style": self.diet_style,
            "medical_conditions": self.medical_conditions,
        }
        return [key for key, value in required.items() if value in (None, "")]


class DailyCheckIn(db.Model):
    __tablename__ = "daily_checkins"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    day = db.Column(db.Date, default=date.today, index=True, nullable=False)

    sleep_hours = db.Column(db.Float, nullable=True)
    sleep_quality = db.Column(db.Integer, nullable=True)  # 1-10
    sleep_notes = db.Column(db.Text, nullable=True)

    morning_energy = db.Column(db.Integer, nullable=True)  # 1-10
    morning_focus = db.Column(db.Integer, nullable=True)  # 1-10
    morning_mood = db.Column(db.Integer, nullable=True)  # 1-10
    morning_stress = db.Column(db.Integer, nullable=True)  # 1-10
    morning_weight_kg = db.Column(db.Float, nullable=True)
    morning_notes = db.Column(db.Text, nullable=True)

    midday_energy = db.Column(db.Integer, nullable=True)  # 1-10
    midday_focus = db.Column(db.Integer, nullable=True)  # 1-10
    midday_mood = db.Column(db.Integer, nullable=True)  # 1-10
    midday_stress = db.Column(db.Integer, nullable=True)  # 1-10
    midday_notes = db.Column(db.Text, nullable=True)

    evening_energy = db.Column(db.Integer, nullable=True)  # 1-10
    evening_focus = db.Column(db.Integer, nullable=True)  # 1-10
    evening_mood = db.Column(db.Integer, nullable=True)  # 1-10
    evening_stress = db.Column(db.Integer, nullable=True)  # 1-10
    evening_notes = db.Column(db.Text, nullable=True)

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
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.UniqueConstraint("user_id", "day", name="uq_daily_checkin_user_day"),)


class Meal(db.Model):
    __tablename__ = "meals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    food_item_id = db.Column(db.Integer, db.ForeignKey("food_items.id"), nullable=True)

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
    caffeine_mg = db.Column(db.Float, nullable=True)
    is_beverage = db.Column(db.Boolean, default=False, nullable=False)

    photo_path = db.Column(db.String(500), nullable=True)
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    food_item = db.relationship("FoodItem", backref="meals", lazy=True)


class FoodItem(db.Model):
    __tablename__ = "food_items"

    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(64), unique=True, nullable=True)
    name = db.Column(db.String(255), index=True, nullable=False)
    brand = db.Column(db.String(255), nullable=True)
    serving_size = db.Column(db.Float, nullable=True)
    serving_unit = db.Column(db.String(32), nullable=True)

    calories = db.Column(db.Integer, nullable=True)
    protein_g = db.Column(db.Float, nullable=True)
    carbs_g = db.Column(db.Float, nullable=True)
    fat_g = db.Column(db.Float, nullable=True)
    sugar_g = db.Column(db.Float, nullable=True)
    sodium_mg = db.Column(db.Float, nullable=True)
    caffeine_mg = db.Column(db.Float, nullable=True)

    source = db.Column(db.String(50), nullable=False, default="seed")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (db.UniqueConstraint("name", "brand", "source", name="uq_food_items_name_brand_source"),)

    def display_name(self):
        if self.brand:
            return f"{self.name} ({self.brand})"
        return self.name


class FavoriteMeal(db.Model):
    __tablename__ = "favorite_meals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    food_item_id = db.Column(db.Integer, db.ForeignKey("food_items.id"), nullable=True)

    name = db.Column(db.String(120), nullable=False)
    label = db.Column(db.String(120), nullable=True)
    description = db.Column(db.Text, nullable=True)
    portion_notes = db.Column(db.String(255), nullable=True)
    tags = db.Column(db.JSON, nullable=True)
    is_beverage = db.Column(db.Boolean, default=False, nullable=False)

    calories = db.Column(db.Integer, nullable=True)
    protein_g = db.Column(db.Float, nullable=True)
    carbs_g = db.Column(db.Float, nullable=True)
    fat_g = db.Column(db.Float, nullable=True)
    sugar_g = db.Column(db.Float, nullable=True)
    sodium_mg = db.Column(db.Float, nullable=True)
    caffeine_mg = db.Column(db.Float, nullable=True)
    ingredients = db.Column(db.JSON, nullable=True)
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    food_item = db.relationship("FoodItem", backref="favorite_meals", lazy=True)

    __table_args__ = (db.UniqueConstraint("user_id", "name", name="uq_favorite_meals_user_name"),)


class Substance(db.Model):
    __tablename__ = "substances"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    taken_at = db.Column(db.DateTime, index=True, nullable=False)

    kind = db.Column(db.String(50), nullable=False)  # alcohol, caffeine, nicotine, etc.
    amount = db.Column(db.String(120), nullable=True)  # "2 beers", "300mg"
    notes = db.Column(db.Text, nullable=True)
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class MIMChatMessage(db.Model):
    __tablename__ = "mim_chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)  # user | assistant
    content = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    context = db.Column(db.String(40), nullable=True, default="general")
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


class UserGoal(db.Model):
    __tablename__ = "user_goals"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(180), nullable=False)
    goal_type = db.Column(db.String(40), nullable=False, default="custom", index=True)
    status = db.Column(db.String(20), nullable=False, default="active", index=True)
    priority = db.Column(db.String(20), nullable=False, default="medium", index=True)

    start_date = db.Column(db.Date, nullable=False, default=date.today, index=True)
    target_date = db.Column(db.Date, nullable=True, index=True)
    target_value = db.Column(db.Float, nullable=True)
    target_unit = db.Column(db.String(40), nullable=True)
    baseline_value = db.Column(db.Float, nullable=True)
    baseline_unit = db.Column(db.String(40), nullable=True)
    constraints = db.Column(db.Text, nullable=True)
    daily_commitment_minutes = db.Column(db.Integer, nullable=True)

    coach_message = db.Column(db.Text, nullable=True)
    today_action = db.Column(db.String(255), nullable=True)
    week_plan = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    actions = db.relationship(
        "UserGoalAction",
        backref="goal",
        lazy=True,
        cascade="all, delete-orphan",
    )


class UserGoalAction(db.Model):
    __tablename__ = "user_goal_actions"

    id = db.Column(db.Integer, primary_key=True)
    goal_id = db.Column(db.Integer, db.ForeignKey("user_goals.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    day = db.Column(db.Date, nullable=False, index=True)
    is_done = db.Column(db.Boolean, nullable=False, default=True)
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (db.UniqueConstraint("goal_id", "day", name="uq_user_goal_actions_goal_day"),)


class CommunityPost(db.Model):
    __tablename__ = "community_posts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    category = db.Column(db.String(40), nullable=False, index=True)
    title = db.Column(db.String(180), nullable=False)
    content = db.Column(db.Text, nullable=False)
    media_path = db.Column(db.String(500), nullable=True)
    media_kind = db.Column(db.String(16), nullable=True)
    is_hidden = db.Column(db.Boolean, nullable=False, default=False, index=True)
    is_flagged = db.Column(db.Boolean, nullable=False, default=False, index=True)
    flag_reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    comments = db.relationship(
        "CommunityComment",
        backref="post",
        lazy=True,
        cascade="all, delete-orphan",
    )
    likes = db.relationship(
        "CommunityLike",
        backref="post",
        lazy=True,
        cascade="all, delete-orphan",
    )


class CommunityComment(db.Model):
    __tablename__ = "community_comments"

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey("community_posts.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    is_hidden = db.Column(db.Boolean, nullable=False, default=False, index=True)
    is_flagged = db.Column(db.Boolean, nullable=False, default=False, index=True)
    flag_reason = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


class CommunityLike(db.Model):
    __tablename__ = "community_likes"

    id = db.Column(db.Integer, primary_key=True)
    post_id = db.Column(db.Integer, db.ForeignKey("community_posts.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (db.UniqueConstraint("post_id", "user_id", name="uq_community_likes_post_user"),)


class AdminUser(db.Model):
    __tablename__ = "admin_users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    last_login_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class BlockedEmail(db.Model):
    __tablename__ = "blocked_emails"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    reason = db.Column(db.String(80), nullable=True)
    created_by_admin_id = db.Column(db.Integer, db.ForeignKey("admin_users.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class SiteSetting(db.Model):
    __tablename__ = "site_settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(120), unique=True, nullable=False, index=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class SupportMessage(db.Model):
    __tablename__ = "support_messages"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    sender_name = db.Column(db.String(255), nullable=False)
    sender_email = db.Column(db.String(255), nullable=False, index=True)
    subject = db.Column(db.String(180), nullable=False)
    message = db.Column(db.Text, nullable=False)
    attachment_path = db.Column(db.String(500), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="new", index=True)
    admin_note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved_at = db.Column(db.DateTime, nullable=True, index=True)


class HomePageVisit(db.Model):
    __tablename__ = "home_page_visits"

    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.Date, nullable=False, index=True)
    host = db.Column(db.String(120), nullable=False, index=True)
    visitor_hash = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    is_authenticated = db.Column(db.Boolean, nullable=False, default=False, index=True)
    hit_count = db.Column(db.Integer, nullable=False, default=1)
    first_seen_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_agent_hash = db.Column(db.String(64), nullable=True)

    __table_args__ = (
        db.UniqueConstraint("day", "host", "visitor_hash", name="uq_home_page_visits_day_host_visitor"),
    )


class UserDailyCoachInsight(db.Model):
    __tablename__ = "user_daily_coach_insights"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    day = db.Column(db.Date, nullable=False, index=True)
    context = db.Column(db.String(32), nullable=False, index=True)
    tip_title = db.Column(db.String(180), nullable=True)
    tip_text = db.Column(db.Text, nullable=True)
    next_action = db.Column(db.String(255), nullable=True)
    recommended_post_ids = db.Column(db.JSON, nullable=True)
    source = db.Column(db.String(20), nullable=False, default="rule", index=True)
    model_name = db.Column(db.String(80), nullable=True)
    encrypted_payload = db.Column(db.LargeBinary, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("user_id", "day", "context", name="uq_user_daily_coach_insight_user_day_context"),
    )
