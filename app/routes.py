import os
from datetime import date, datetime, timedelta
from uuid import uuid4

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from app import db
from app.ai import ai_reflection, coach_prompt_missing_fields
from app.models import DailyCheckIn, Meal, Substance, User

bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_int(value):
    return int(value) if value not in (None, "") else None


def parse_float(value):
    return float(value) if value not in (None, "") else None


def get_default_user():
    user = User.query.first()
    if not user:
        user = User(email=None)
        db.session.add(user)
        db.session.commit()
    return user


@bp.get("/")
def index():
    user = get_default_user()
    checkin_count = DailyCheckIn.query.filter_by(user_id=user.id).count()
    meal_count = Meal.query.filter_by(user_id=user.id).count()
    substance_count = Substance.query.filter_by(user_id=user.id).count()
    return render_template(
        "index.html",
        checkin_count=checkin_count,
        meal_count=meal_count,
        substance_count=substance_count,
    )


@bp.get("/checkin")
def checkin_form():
    user = get_default_user()
    day_str = request.args.get("day")
    selected_day = date.fromisoformat(day_str) if day_str else date.today()
    record = DailyCheckIn.query.filter_by(user_id=user.id, day=selected_day).first()
    return render_template("checkin.html", record=record, selected_day=selected_day.isoformat())


@bp.post("/checkin")
def checkin_save():
    user = get_default_user()
    selected_day = date.fromisoformat(request.form.get("day", date.today().isoformat()))

    record = DailyCheckIn.query.filter_by(user_id=user.id, day=selected_day).first()
    if not record:
        record = DailyCheckIn(user_id=user.id, day=selected_day)

    for field in [
        "sleep_hours",
        "sleep_quality",
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
    flash("Check-in saved.", "success")
    return redirect(url_for("main.timeline"))


@bp.get("/meal")
def meal_form():
    return render_template("meal.html")


@bp.post("/meal")
def meal_save():
    user = get_default_user()
    eaten_at_raw = request.form.get("eaten_at") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    eaten_at_dt = datetime.fromisoformat(eaten_at_raw)

    tags_raw = request.form.get("tags") or ""
    tags = [item.strip() for item in tags_raw.split(",") if item.strip()]

    meal = Meal(
        user_id=user.id,
        eaten_at=eaten_at_dt,
        label=request.form.get("label") or None,
        description=request.form.get("description") or None,
        portion_notes=request.form.get("portion_notes") or None,
        tags=tags or None,
        calories=parse_int(request.form.get("calories")),
        protein_g=parse_float(request.form.get("protein_g")),
        carbs_g=parse_float(request.form.get("carbs_g")),
        fat_g=parse_float(request.form.get("fat_g")),
        sugar_g=parse_float(request.form.get("sugar_g")),
        sodium_mg=parse_float(request.form.get("sodium_mg")),
    )

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

    db.session.add(meal)
    db.session.commit()
    flash("Meal logged.", "success")
    return redirect(url_for("main.timeline"))


@bp.get("/substance")
def substance_form():
    return render_template("substance.html")


@bp.post("/substance")
def substance_save():
    user = get_default_user()
    taken_at_raw = request.form.get("taken_at") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    taken_at = datetime.fromisoformat(taken_at_raw)

    kind = request.form.get("kind")
    if not kind:
        flash("Substance kind is required.", "error")
        return redirect(url_for("main.substance_form"))

    entry = Substance(
        user_id=user.id,
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
def timeline():
    user = get_default_user()
    meals = Meal.query.filter_by(user_id=user.id).order_by(Meal.eaten_at.desc()).limit(80).all()
    checkins = (
        DailyCheckIn.query.filter_by(user_id=user.id).order_by(DailyCheckIn.day.desc()).limit(45).all()
    )
    substances = (
        Substance.query.filter_by(user_id=user.id).order_by(Substance.taken_at.desc()).limit(45).all()
    )

    prompts = coach_prompt_missing_fields(user, meals, checkins)

    return render_template(
        "timeline.html",
        meals=meals,
        checkins=checkins,
        substances=substances,
        prompts=prompts,
    )


@bp.get("/insights")
def insights():
    user = get_default_user()
    today = date.today()
    start_day = today - timedelta(days=6)

    checkins = (
        DailyCheckIn.query.filter(
            DailyCheckIn.user_id == user.id,
            DailyCheckIn.day >= start_day,
            DailyCheckIn.day <= today,
        )
        .order_by(DailyCheckIn.day.asc())
        .all()
    )
    meals = Meal.query.filter(Meal.user_id == user.id, Meal.eaten_at >= datetime.combine(start_day, datetime.min.time())).all()

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
