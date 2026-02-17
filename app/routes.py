import os
from datetime import date, datetime, timedelta
from functools import wraps
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from flask import (
    Blueprint,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sqlalchemy import or_
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from app import db
from app.ai import ai_reflection, coach_prompt_missing_fields
from app.food_catalog import import_foods_from_usda, seed_common_foods_if_needed
from app.models import DailyCheckIn, FavoriteMeal, FoodItem, Meal, Substance, User, UserProfile

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic"}

PREFERRED_TIMEZONES = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Phoenix",
    "America/Los_Angeles",
    "America/Anchorage",
    "Pacific/Honolulu",
    "UTC",
]
ALL_TIMEZONES = sorted(available_timezones())
PREFERRED_TIMEZONE_SET = set(PREFERRED_TIMEZONES)
TIMEZONE_OPTIONS = PREFERRED_TIMEZONES + [tz for tz in ALL_TIMEZONES if tz not in PREFERRED_TIMEZONE_SET]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_int(value):
    return int(value) if value not in (None, "") else None


def parse_float(value):
    return float(value) if value not in (None, "") else None


def parse_bool(value):
    return str(value).lower() in {"1", "true", "yes", "on"}


def parse_tags(raw_value):
    raw = raw_value or ""
    tags = [item.strip() for item in raw.split(",") if item.strip()]
    return tags or None


def get_user_zoneinfo(user: User):
    tz_name = None
    if user and user.profile and user.profile.time_zone:
        tz_name = user.profile.time_zone
    if not tz_name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def get_user_local_today(user: User):
    tz = get_user_zoneinfo(user)
    return datetime.now(tz).date()


def checkin_has_any_data(record: DailyCheckIn | None):
    if not record:
        return False

    for field in [
        "sleep_hours",
        "sleep_quality",
        "sleep_notes",
        "morning_energy",
        "morning_focus",
        "morning_mood",
        "morning_stress",
        "morning_weight_kg",
        "morning_notes",
        "midday_energy",
        "midday_focus",
        "midday_mood",
        "midday_stress",
        "midday_notes",
        "evening_energy",
        "evening_focus",
        "evening_mood",
        "evening_stress",
        "evening_notes",
        "energy",
        "focus",
        "mood",
        "stress",
        "anxiety",
        "productivity",
        "accomplishments",
        "notes",
        "workout_timing",
        "workout_intensity",
        "alcohol_drinks",
        "symptoms",
        "digestion",
    ]:
        value = getattr(record, field)
        if value not in (None, "", [], {}):
            return True
    return False


def checkin_segment_status(record: DailyCheckIn | None):
    def has_values(fields):
        if not record:
            return False
        for field in fields:
            value = getattr(record, field)
            if value not in (None, "", [], {}):
                return True
        return False

    return {
        "sleep": has_values(["sleep_hours", "sleep_quality", "sleep_notes"]),
        "morning": has_values(
            [
                "morning_energy",
                "morning_focus",
                "morning_mood",
                "morning_stress",
                "morning_weight_kg",
                "morning_notes",
            ]
        ),
        "midday": has_values(
            ["midday_energy", "midday_focus", "midday_mood", "midday_stress", "midday_notes"]
        ),
        "evening": has_values(
            ["evening_energy", "evening_focus", "evening_mood", "evening_stress", "evening_notes"]
        ),
        "overall": has_values(
            [
                "energy",
                "focus",
                "mood",
                "stress",
                "anxiety",
                "productivity",
                "accomplishments",
                "notes",
                "workout_timing",
                "workout_intensity",
                "alcohol_drinks",
            ]
        ),
    }


def day_bounds(target_day: date):
    start = datetime.combine(target_day, datetime.min.time())
    end = start + timedelta(days=1)
    return start, end


def build_meal_context(selected_day: date, edit_meal: Meal | None = None):
    seed_common_foods_if_needed()

    start, end = day_bounds(selected_day)
    day_meals = (
        Meal.query.filter(
            Meal.user_id == g.user.id,
            Meal.eaten_at >= start,
            Meal.eaten_at < end,
        )
        .order_by(Meal.eaten_at.asc())
        .all()
    )
    favorites = FavoriteMeal.query.filter_by(user_id=g.user.id).order_by(FavoriteMeal.name.asc()).all()

    favorite_payload = [
        {
            "id": f.id,
            "name": f.name,
            "label": f.label,
            "food_item_id": f.food_item_id,
            "description": f.description,
            "portion_notes": f.portion_notes,
            "tags": ", ".join(f.tags) if f.tags else "",
            "calories": f.calories,
            "protein_g": f.protein_g,
            "carbs_g": f.carbs_g,
            "fat_g": f.fat_g,
            "sugar_g": f.sugar_g,
            "sodium_mg": f.sodium_mg,
            "is_beverage": f.is_beverage,
        }
        for f in favorites
    ]

    default_time = datetime.now().strftime("%H:%M") if selected_day == date.today() else "12:00"

    return {
        "selected_day": selected_day.isoformat(),
        "day_meals": day_meals,
        "favorites": favorites,
        "favorite_payload": favorite_payload,
        "edit_meal": edit_meal,
        "default_eaten_at": f"{selected_day.isoformat()}T{default_time}",
    }


def apply_meal_fields_from_request(meal: Meal):
    eaten_at_raw = request.form.get("eaten_at") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    eaten_at_dt = datetime.fromisoformat(eaten_at_raw)
    food_item_id = parse_int(request.form.get("food_item_id"))
    food_item = db.session.get(FoodItem, food_item_id) if food_item_id else None

    meal.user_id = g.user.id
    meal.food_item_id = food_item.id if food_item else None
    meal.eaten_at = eaten_at_dt
    meal.label = request.form.get("label") or None
    meal.description = request.form.get("description") or None
    meal.portion_notes = request.form.get("portion_notes") or None
    meal.tags = parse_tags(request.form.get("tags"))
    meal.is_beverage = parse_bool(request.form.get("is_beverage"))

    meal.calories = parse_int(request.form.get("calories"))
    meal.protein_g = parse_float(request.form.get("protein_g"))
    meal.carbs_g = parse_float(request.form.get("carbs_g"))
    meal.fat_g = parse_float(request.form.get("fat_g"))
    meal.sugar_g = parse_float(request.form.get("sugar_g"))
    meal.sodium_mg = parse_float(request.form.get("sodium_mg"))

    if food_item:
        if not meal.description:
            meal.description = food_item.display_name()
        if meal.calories is None:
            meal.calories = food_item.calories
        if meal.protein_g is None:
            meal.protein_g = food_item.protein_g
        if meal.carbs_g is None:
            meal.carbs_g = food_item.carbs_g
        if meal.fat_g is None:
            meal.fat_g = food_item.fat_g
        if meal.sugar_g is None:
            meal.sugar_g = food_item.sugar_g
        if meal.sodium_mg is None:
            meal.sodium_mg = food_item.sodium_mg

    return eaten_at_dt


def upsert_favorite_from_request():
    if not parse_bool(request.form.get("save_favorite")):
        return

    favorite_name = (request.form.get("favorite_name") or "").strip()
    if not favorite_name:
        return

    favorite = FavoriteMeal.query.filter_by(user_id=g.user.id, name=favorite_name).first()
    if not favorite:
        favorite = FavoriteMeal(user_id=g.user.id, name=favorite_name)

    favorite.food_item_id = parse_int(request.form.get("food_item_id"))
    favorite.label = request.form.get("label") or None
    favorite.description = request.form.get("description") or None
    favorite.portion_notes = request.form.get("portion_notes") or None
    favorite.tags = parse_tags(request.form.get("tags"))
    favorite.is_beverage = parse_bool(request.form.get("is_beverage"))

    favorite.calories = parse_int(request.form.get("calories"))
    favorite.protein_g = parse_float(request.form.get("protein_g"))
    favorite.carbs_g = parse_float(request.form.get("carbs_g"))
    favorite.fat_g = parse_float(request.form.get("fat_g"))
    favorite.sugar_g = parse_float(request.form.get("sugar_g"))
    favorite.sodium_mg = parse_float(request.form.get("sodium_mg"))

    db.session.add(favorite)


def inches_to_cm(inches):
    return round(inches * 2.54, 2) if inches is not None else None


def cm_to_inches(cm):
    return round(cm / 2.54, 2) if cm is not None else None


def lb_to_kg(lb):
    return round(lb / 2.2046226218, 2) if lb is not None else None


def kg_to_lb(kg):
    return round(kg * 2.2046226218, 1) if kg is not None else None


def feet_inches_to_cm(feet, inches):
    if feet is None and inches is None:
        return None
    return inches_to_cm((feet or 0) * 12 + (inches or 0))


def cm_to_feet_inches(cm):
    if cm is None:
        return (None, None)
    total_inches = cm / 2.54
    feet = int(total_inches // 12)
    inches = round(total_inches - (feet * 12), 1)
    if inches >= 12:
        feet += 1
        inches -= 12
    return (feet, inches)


def build_profile_template_context(profile):
    unit_system = profile.unit_system or "imperial"
    height_ft, height_in = cm_to_feet_inches(profile.height_cm)

    return {
        "profile": profile,
        "missing_required": profile.missing_required_fields(),
        "timezones": TIMEZONE_OPTIONS,
        "unit_system": unit_system,
        "height_ft": height_ft,
        "height_in": height_in,
        "weight_lb": kg_to_lb(profile.weight_kg),
        "waist_in": cm_to_inches(profile.waist_cm),
        "body_fat_slider": profile.body_fat_pct if profile.body_fat_pct is not None else 25,
    }


def normalize_email(value: str | None):
    if not value:
        return None
    return value.strip().lower()


def get_or_create_profile(user: User):
    profile = user.profile
    if not profile:
        profile = UserProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()
    return profile


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Please log in first.", "error")
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def profile_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        profile = get_or_create_profile(g.user)
        missing = profile.missing_required_fields()
        if missing:
            flash("Complete your core profile fields before logging daily data.", "error")
            return redirect(url_for("main.profile"))
        return view(*args, **kwargs)

    return wrapped


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.user = db.session.get(User, user_id) if user_id else None


@bp.app_context_processor
def inject_user():
    current_user = g.get("user")
    profile_complete = False
    if current_user and current_user.profile:
        profile_complete = len(current_user.profile.missing_required_fields()) == 0
    return {"current_user": current_user, "profile_complete": profile_complete}


@bp.get("/")
def index():
    if g.user is None:
        return render_template("index.html", is_authenticated=False)

    profile = get_or_create_profile(g.user)
    checkin_count = DailyCheckIn.query.filter_by(user_id=g.user.id).count()
    meal_count = Meal.query.filter_by(user_id=g.user.id).count()
    substance_count = Substance.query.filter_by(user_id=g.user.id).count()
    return render_template(
        "index.html",
        is_authenticated=True,
        checkin_count=checkin_count,
        meal_count=meal_count,
        substance_count=substance_count,
        missing_required=profile.missing_required_fields(),
    )


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        password_confirm = request.form.get("password_confirm") or ""

        if not full_name:
            flash("Full name is required.", "error")
            return render_template("register.html")
        if not email:
            flash("Email is required.", "error")
            return render_template("register.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("register.html")
        if password != password_confirm:
            flash("Password confirmation does not match.", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("register.html")

        user = User(
            full_name=full_name,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.flush()
        db.session.add(UserProfile(user_id=user.id))
        db.session.commit()

        session.clear()
        session["user_id"] = user.id
        flash("Account created. Complete your profile to begin tracking.", "success")
        return redirect(url_for("main.profile"))

    return render_template("register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("main.index"))

    if request.method == "POST":
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        next_url = request.form.get("next") or request.args.get("next")

        user = User.query.filter_by(email=email).first() if email else None
        if not user or not user.password_hash or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return render_template("login.html", next=next_url)

        session.clear()
        session["user_id"] = user.id
        flash("Logged in.", "success")

        if next_url and next_url.startswith("/"):
            return redirect(next_url)
        return redirect(url_for("main.index"))

    return render_template("login.html", next=request.args.get("next"))


@bp.post("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("main.login"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    profile = get_or_create_profile(g.user)

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = normalize_email(request.form.get("email"))
        if not full_name:
            flash("Full name is required.", "error")
            return render_template("profile.html", **build_profile_template_context(profile))
        if not email:
            flash("Email is required.", "error")
            return render_template("profile.html", **build_profile_template_context(profile))
        existing_user = User.query.filter(User.email == email, User.id != g.user.id).first()
        if existing_user:
            flash("Email is already in use by another account.", "error")
            return render_template("profile.html", **build_profile_template_context(profile))

        g.user.full_name = full_name
        g.user.email = email

        profile.age = parse_int(request.form.get("age"))
        profile.biological_sex = request.form.get("biological_sex") or None
        profile.time_zone = request.form.get("time_zone") or None
        selected_unit_system = (request.form.get("unit_system") or "imperial").lower()
        profile.unit_system = selected_unit_system if selected_unit_system in {"imperial", "metric"} else "imperial"
        profile.phone = request.form.get("phone") or None

        if profile.unit_system == "imperial":
            height_ft = parse_int(request.form.get("height_ft"))
            height_in = parse_float(request.form.get("height_in"))
            weight_lb = parse_float(request.form.get("weight_lb"))
            waist_in = parse_float(request.form.get("waist_in"))
            profile.height_cm = feet_inches_to_cm(height_ft, height_in)
            profile.weight_kg = lb_to_kg(weight_lb)
            profile.waist_cm = inches_to_cm(waist_in)
        else:
            profile.height_cm = parse_float(request.form.get("height_cm"))
            profile.weight_kg = parse_float(request.form.get("weight_kg"))
            profile.waist_cm = parse_float(request.form.get("waist_cm"))

        profile.body_fat_pct = parse_float(request.form.get("body_fat_pct"))

        profile.general_health_rating = parse_int(request.form.get("general_health_rating"))
        profile.medical_conditions = request.form.get("medical_conditions") or None
        profile.known_sleep_issues = request.form.get("known_sleep_issues") or None
        profile.family_history_flags = request.form.get("family_history_flags") or None
        profile.medications = request.form.get("medications") or None
        profile.supplements = request.form.get("supplements") or None
        profile.resting_blood_pressure = request.form.get("resting_blood_pressure") or None

        profile.fitness_level = request.form.get("fitness_level") or None
        profile.typical_sleep_duration_hours = parse_float(request.form.get("typical_sleep_duration_hours"))
        profile.work_type = request.form.get("work_type") or None
        profile.work_stress_baseline = parse_int(request.form.get("work_stress_baseline"))
        profile.typical_alcohol_frequency = request.form.get("typical_alcohol_frequency") or None
        profile.caffeine_baseline = request.form.get("caffeine_baseline") or None
        profile.nicotine_use = request.form.get("nicotine_use") or None
        profile.recreational_drug_use = request.form.get("recreational_drug_use") or None

        profile.diet_style = request.form.get("diet_style") or None
        profile.food_intolerances = request.form.get("food_intolerances") or None
        profile.food_sensitivities = request.form.get("food_sensitivities") or None
        profile.typical_meal_timing = request.form.get("typical_meal_timing") or None
        profile.cravings_patterns = request.form.get("cravings_patterns") or None

        profile.baseline_mood = parse_int(request.form.get("baseline_mood"))
        profile.baseline_anxiety = parse_int(request.form.get("baseline_anxiety"))
        profile.baseline_focus = parse_int(request.form.get("baseline_focus"))
        profile.energy_consistency = request.form.get("energy_consistency") or None
        profile.attention_issues = request.form.get("attention_issues") or None
        profile.emotional_volatility = request.form.get("emotional_volatility") or None
        profile.burnout_history = request.form.get("burnout_history") or None

        profile.primary_goal = request.form.get("primary_goal") or None
        profile.secondary_goals = request.form.get("secondary_goals") or None
        profile.time_horizon = request.form.get("time_horizon") or None
        profile.great_day_definition = request.form.get("great_day_definition") or None

        profile.chronotype = request.form.get("chronotype") or None
        profile.digestive_sensitivity = request.form.get("digestive_sensitivity") or None
        profile.stress_reactivity = request.form.get("stress_reactivity") or None
        profile.social_pattern = request.form.get("social_pattern") or None
        profile.screen_time_evening_hours = parse_float(request.form.get("screen_time_evening_hours"))

        db.session.add(g.user)
        db.session.add(profile)
        db.session.commit()

        missing_required = profile.missing_required_fields()
        if missing_required:
            flash(
                "Profile saved. Core fields still missing: " + ", ".join(missing_required),
                "error",
            )
        else:
            flash("Profile saved.", "success")
            return redirect(url_for("main.index"))

    return render_template("profile.html", **build_profile_template_context(profile))


@bp.get("/checkin")
@login_required
@profile_required
def checkin_form():
    local_today = get_user_local_today(g.user)
    day_str = request.args.get("day")
    if day_str:
        try:
            selected_day = date.fromisoformat(day_str)
        except ValueError:
            selected_day = local_today
            flash("Invalid date format. Showing current local day.", "error")
    else:
        selected_day = local_today

    if selected_day > local_today:
        selected_day = local_today
        flash("Future check-ins are disabled. Showing current local day.", "error")

    record = DailyCheckIn.query.filter_by(user_id=g.user.id, day=selected_day).first()
    today_record = DailyCheckIn.query.filter_by(user_id=g.user.id, day=local_today).first()
    unit_system = "imperial"
    if g.user.profile and g.user.profile.unit_system in {"imperial", "metric"}:
        unit_system = g.user.profile.unit_system
    weight_unit = "lb" if unit_system == "imperial" else "kg"
    morning_weight_display = None
    if record and record.morning_weight_kg is not None:
        morning_weight_display = kg_to_lb(record.morning_weight_kg) if unit_system == "imperial" else round(record.morning_weight_kg, 2)

    history_records = (
        DailyCheckIn.query.filter_by(user_id=g.user.id).order_by(DailyCheckIn.day.desc()).limit(30).all()
    )
    history_rows = [
        {
            "record": row,
            "segments": checkin_segment_status(row),
            "is_today": row.day == local_today,
            "has_data": checkin_has_any_data(row),
        }
        for row in history_records
    ]

    prev_day = selected_day - timedelta(days=1)
    next_day = selected_day + timedelta(days=1)
    can_go_next = next_day <= local_today

    return render_template(
        "checkin.html",
        record=record,
        selected_day=selected_day.isoformat(),
        local_today=local_today.isoformat(),
        checked_in_today=checkin_has_any_data(today_record),
        selected_segments=checkin_segment_status(record),
        history_rows=history_rows,
        prev_day=prev_day.isoformat(),
        next_day=next_day.isoformat(),
        can_go_next=can_go_next,
        is_viewing_today=(selected_day == local_today),
        checkin_weight_unit=weight_unit,
        morning_weight_display=morning_weight_display,
        checkin_unit_system=unit_system,
    )


@bp.post("/checkin")
@login_required
@profile_required
def checkin_save():
    local_today = get_user_local_today(g.user)
    selected_day_raw = request.form.get("day", local_today.isoformat())
    try:
        selected_day = date.fromisoformat(selected_day_raw)
    except ValueError:
        selected_day = local_today

    if selected_day > local_today:
        selected_day = local_today

    unit_system = "imperial"
    if g.user.profile and g.user.profile.unit_system in {"imperial", "metric"}:
        unit_system = g.user.profile.unit_system

    record = DailyCheckIn.query.filter_by(user_id=g.user.id, day=selected_day).first()
    if not record:
        record = DailyCheckIn(user_id=g.user.id, day=selected_day)

    for field in [
        "sleep_hours",
        "sleep_quality",
        "morning_energy",
        "morning_focus",
        "morning_mood",
        "morning_stress",
        "midday_energy",
        "midday_focus",
        "midday_mood",
        "midday_stress",
        "evening_energy",
        "evening_focus",
        "evening_mood",
        "evening_stress",
        "energy",
        "focus",
        "mood",
        "stress",
        "anxiety",
        "productivity",
        "workout_intensity",
        "alcohol_drinks",
    ]:
        value = request.form.get(field)
        if field in {"sleep_hours", "alcohol_drinks"}:
            setattr(record, field, parse_float(value))
        else:
            setattr(record, field, parse_int(value))

    morning_weight_value = parse_float(request.form.get("morning_weight"))
    if morning_weight_value is None:
        record.morning_weight_kg = None
    else:
        record.morning_weight_kg = lb_to_kg(morning_weight_value) if unit_system == "imperial" else morning_weight_value

    record.sleep_notes = request.form.get("sleep_notes") or None
    record.morning_notes = request.form.get("morning_notes") or None
    record.midday_notes = request.form.get("midday_notes") or None
    record.evening_notes = request.form.get("evening_notes") or None
    record.workout_timing = request.form.get("workout_timing") or None
    record.accomplishments = request.form.get("accomplishments") or None
    record.notes = request.form.get("notes") or None

    symptoms = {}
    for key in ["headache", "stomach", "nausea"]:
        parsed = parse_int(request.form.get(key))
        if parsed is not None:
            symptoms[key] = parsed
    record.symptoms = symptoms or None

    bm_count = parse_int(request.form.get("bm_count"))
    digestion_issues = request.form.get("digestion_issues") or ""
    digestion = {}
    if bm_count is not None:
        digestion["bm_count"] = bm_count
    if digestion_issues.strip():
        digestion["issues"] = [item.strip() for item in digestion_issues.split(",") if item.strip()]
    record.digestion = digestion or None

    db.session.add(record)
    db.session.commit()
    flash(f"Check-in saved for {selected_day.isoformat()}.", "success")
    return redirect(url_for("main.checkin_form", day=selected_day.isoformat()))


@bp.get("/meal")
@login_required
@profile_required
def meal_form():
    day_str = request.args.get("day")
    selected_day = date.fromisoformat(day_str) if day_str else date.today()
    return render_template("meal.html", form_action=url_for("main.meal_save"), **build_meal_context(selected_day))


@bp.post("/meal")
@login_required
@profile_required
def meal_save():
    meal = Meal()
    try:
        eaten_at_dt = apply_meal_fields_from_request(meal)
    except ValueError:
        flash("Invalid meal timestamp. Use the date/time picker and try again.", "error")
        return redirect(url_for("main.meal_form"))

    photo = request.files.get("photo")
    if photo and photo.filename:
        if not allowed_file(photo.filename):
            flash("Unsupported file type. Use png, jpg, jpeg, webp, or heic.", "error")
            return redirect(url_for("main.meal_form"))
        safe_name = secure_filename(photo.filename)
        upload_name = f"{uuid4().hex}_{safe_name}"
        upload_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        local_path = os.path.join(upload_dir, upload_name)
        photo.save(local_path)
        meal.photo_path = f"uploads/{upload_name}"

    upsert_favorite_from_request()
    db.session.add(meal)
    db.session.commit()
    flash("Meal logged.", "success")
    return redirect(url_for("main.meal_form", day=eaten_at_dt.date().isoformat()))


@bp.route("/meal/<int:meal_id>/edit", methods=["GET", "POST"])
@login_required
@profile_required
def meal_edit(meal_id: int):
    meal = Meal.query.filter_by(id=meal_id, user_id=g.user.id).first_or_404()

    if request.method == "POST":
        try:
            eaten_at_dt = apply_meal_fields_from_request(meal)
        except ValueError:
            flash("Invalid meal timestamp. Use the date/time picker and try again.", "error")
            return redirect(url_for("main.meal_edit", meal_id=meal.id))
        upsert_favorite_from_request()

        photo = request.files.get("photo")
        if photo and photo.filename:
            if not allowed_file(photo.filename):
                flash("Unsupported file type. Use png, jpg, jpeg, webp, or heic.", "error")
                return redirect(url_for("main.meal_edit", meal_id=meal.id))
            safe_name = secure_filename(photo.filename)
            upload_name = f"{uuid4().hex}_{safe_name}"
            upload_dir = current_app.config["UPLOAD_FOLDER"]
            os.makedirs(upload_dir, exist_ok=True)
            local_path = os.path.join(upload_dir, upload_name)
            photo.save(local_path)
            meal.photo_path = f"uploads/{upload_name}"

        db.session.add(meal)
        db.session.commit()
        flash("Meal updated.", "success")
        return redirect(url_for("main.meal_form", day=eaten_at_dt.date().isoformat()))

    selected_day = meal.eaten_at.date()
    return render_template(
        "meal.html",
        form_action=url_for("main.meal_edit", meal_id=meal.id),
        **build_meal_context(selected_day, edit_meal=meal),
    )


@bp.post("/meal/<int:meal_id>/delete")
@login_required
@profile_required
def meal_delete(meal_id: int):
    meal = Meal.query.filter_by(id=meal_id, user_id=g.user.id).first_or_404()
    selected_day = request.form.get("day") or meal.eaten_at.date().isoformat()
    db.session.delete(meal)
    db.session.commit()
    flash("Meal deleted.", "success")
    return redirect(url_for("main.meal_form", day=selected_day))


@bp.get("/foods/search")
@login_required
@profile_required
def food_search():
    query = (request.args.get("q") or "").strip()
    include_remote = parse_bool(request.args.get("remote"))
    if len(query) < 2:
        return jsonify({"results": [], "message": "Type at least 2 characters."})

    seed_common_foods_if_needed()
    results = (
        FoodItem.query.filter(
            or_(
                FoodItem.name.ilike(f"%{query}%"),
                FoodItem.brand.ilike(f"%{query}%"),
            )
        )
        .order_by(FoodItem.name.asc())
        .limit(15)
        .all()
    )

    imported = 0
    message = None
    if include_remote and len(results) < 8:
        imported = import_foods_from_usda(query, max_results=12)
        if imported > 0:
            results = (
                FoodItem.query.filter(
                    or_(
                        FoodItem.name.ilike(f"%{query}%"),
                        FoodItem.brand.ilike(f"%{query}%"),
                    )
                )
                .order_by(FoodItem.name.asc())
                .limit(15)
                .all()
            )
        elif len(results) == 0:
            message = "No USDA matches found for that term. Try a broader keyword."
    elif len(results) == 0:
        message = "No local matches. Click Search USDA for a larger catalog."

    payload = [
        {
            "id": item.id,
            "name": item.name,
            "brand": item.brand,
            "display_name": item.display_name(),
            "serving_size": item.serving_size,
            "serving_unit": item.serving_unit,
            "calories": item.calories,
            "protein_g": item.protein_g,
            "carbs_g": item.carbs_g,
            "fat_g": item.fat_g,
            "sugar_g": item.sugar_g,
            "sodium_mg": item.sodium_mg,
            "source": item.source,
        }
        for item in results
    ]
    return jsonify({"results": payload, "message": message, "imported": imported})


@bp.get("/substance")
@login_required
@profile_required
def substance_form():
    return render_template("substance.html")


@bp.post("/substance")
@login_required
@profile_required
def substance_save():
    taken_at_raw = request.form.get("taken_at") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    taken_at = datetime.fromisoformat(taken_at_raw)

    kind = request.form.get("kind")
    if not kind:
        flash("Substance kind is required.", "error")
        return redirect(url_for("main.substance_form"))

    entry = Substance(
        user_id=g.user.id,
        taken_at=taken_at,
        kind=kind,
        amount=request.form.get("amount") or None,
        notes=request.form.get("notes") or None,
    )
    db.session.add(entry)
    db.session.commit()
    flash("Substance entry logged.", "success")
    return redirect(url_for("main.timeline"))


@bp.get("/timeline")
@login_required
@profile_required
def timeline():
    meals = Meal.query.filter_by(user_id=g.user.id).order_by(Meal.eaten_at.desc()).limit(80).all()
    checkins = DailyCheckIn.query.filter_by(user_id=g.user.id).order_by(DailyCheckIn.day.desc()).limit(45).all()
    substances = Substance.query.filter_by(user_id=g.user.id).order_by(Substance.taken_at.desc()).limit(45).all()

    prompts = coach_prompt_missing_fields(g.user, meals, checkins)

    return render_template(
        "timeline.html",
        meals=meals,
        checkins=checkins,
        substances=substances,
        prompts=prompts,
    )


@bp.get("/insights")
@login_required
@profile_required
def insights():
    today = date.today()
    start_day = today - timedelta(days=6)

    checkins = (
        DailyCheckIn.query.filter(
            DailyCheckIn.user_id == g.user.id,
            DailyCheckIn.day >= start_day,
            DailyCheckIn.day <= today,
        )
        .order_by(DailyCheckIn.day.asc())
        .all()
    )
    meals = Meal.query.filter(
        Meal.user_id == g.user.id,
        Meal.eaten_at >= datetime.combine(start_day, datetime.min.time()),
    ).all()

    days_with_checkins = {c.day for c in checkins}
    sleep_values = [c.sleep_hours for c in checkins if c.sleep_hours is not None]
    focus_values = [c.focus for c in checkins if c.focus is not None]
    mood_values = [c.mood for c in checkins if c.mood is not None]
    productivity_values = [c.productivity for c in checkins if c.productivity is not None]

    avg_sleep = round(sum(sleep_values) / len(sleep_values), 2) if sleep_values else None
    avg_focus = round(sum(focus_values) / len(focus_values), 2) if focus_values else None
    avg_mood = round(sum(mood_values) / len(mood_values), 2) if mood_values else None
    avg_productivity = round(sum(productivity_values) / len(productivity_values), 2) if productivity_values else None

    short_sleep_days = len([v for v in sleep_values if v < 6.5])
    total_alcohol = sum(c.alcohol_drinks or 0 for c in checkins)
    coverage = round((len(days_with_checkins) / 7) * 100, 1)

    summary_lines = [
        f"Coverage over last 7 days: {coverage}%",
        f"Average sleep hours: {avg_sleep if avg_sleep is not None else 'n/a'}",
        f"Average focus: {avg_focus if avg_focus is not None else 'n/a'}",
        f"Average mood: {avg_mood if avg_mood is not None else 'n/a'}",
        f"Average productivity: {avg_productivity if avg_productivity is not None else 'n/a'}",
        f"Days with sleep under 6.5h: {short_sleep_days}",
        f"Total alcohol drinks logged: {total_alcohol}",
        f"Meals logged: {len(meals)}",
    ]
    summary_text = "\n".join(summary_lines)
    reflection = ai_reflection(summary_text)

    return render_template(
        "insights.html",
        start_day=start_day,
        today=today,
        coverage=coverage,
        avg_sleep=avg_sleep,
        avg_focus=avg_focus,
        avg_mood=avg_mood,
        avg_productivity=avg_productivity,
        short_sleep_days=short_sleep_days,
        total_alcohol=total_alcohol,
        meal_count=len(meals),
        reflection=reflection,
    )
