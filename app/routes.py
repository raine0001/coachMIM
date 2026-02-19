import json
import os
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from functools import wraps
from urllib.parse import quote_plus
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
from sqlalchemy import case, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from app import db
from app.ai import (
    ask_mim_general_chat,
    ai_reflection,
    community_content_is_blocked,
    coach_prompt_missing_fields,
    parse_day_manager_context_assist,
    parse_meal_sentence,
    parse_nutrition_label_image,
    parse_product_page_url,
)
from app.food_catalog import import_foods_from_usda, seed_common_foods_if_needed
from app.models import (
    AdminUser,
    BlockedEmail,
    CommunityComment,
    CommunityLike,
    CommunityPost,
    DailyCheckIn,
    FavoriteMeal,
    FoodItem,
    MIMChatMessage,
    Meal,
    SiteSetting,
    Substance,
    User,
    UserProfile,
)
from app.security import encrypt_model_fields, hydrate_model_fields

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

ADDED_SUGAR_HINTS = (
    "soda",
    "cookie",
    "cake",
    "candy",
    "chocolate",
    "pastry",
    "dessert",
    "ice cream",
    "donut",
    "syrup",
    "sweetened",
    "energy drink",
    "sports drink",
)
NATURAL_SUGAR_HINTS = (
    "fruit",
    "banana",
    "apple",
    "orange",
    "berries",
    "berry",
    "grape",
    "mango",
    "pineapple",
    "milk",
    "yogurt",
    "smoothie",
)

GENERIC_MEAL_NAMES = {
    "meal",
    "food",
    "drink",
    "entry",
    "custom entry",
    "custom built meal",
}
SEARCH_STOPWORDS = {"and", "or", "the", "a", "an", "of", "with", "to", "for"}
DAY_MANAGER_VIEWS = {"checkin", "meal", "drink", "substance", "activity", "medications"}
DAY_MANAGER_FAVORITE_SCOPE_PREFIX = "__dmv:"
ADMIN_SESSION_KEY = "admin_user_id"
ADMIN_BOOTSTRAP_USERNAME = os.getenv("ADMIN_BOOTSTRAP_USERNAME", "testpilot")
ADMIN_BOOTSTRAP_PASSWORD = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "1234")
COMMUNITY_CATEGORY_OPTIONS = [
    {"key": "health", "label": "Health"},
    {"key": "fitness", "label": "Fitness"},
    {"key": "food", "label": "Food"},
    {"key": "lifestyle", "label": "Lifestyle"},
    {"key": "supplements", "label": "Supplements"},
    {"key": "recipes", "label": "Recipes"},
    {"key": "exercise", "label": "Exercise"},
    {"key": "stories", "label": "Stories"},
    {"key": "general", "label": "General"},
]
COMMUNITY_CATEGORY_KEYS = {item["key"] for item in COMMUNITY_CATEGORY_OPTIONS}
SITE_SETTING_DEFAULTS = {
    "home_intro": (
        "CoachMIM is a structured self-report AI system for longitudinal behavioral "
        "pattern detection."
    ),
    "contact_email": "support@coachmim.com",
    "privacy_summary": (
        "CoachMIM stores your data to provide tracking, trend analysis, and personalized "
        "feedback. You can request account deletion at any time."
    ),
    "terms_summary": (
        "CoachMIM provides educational wellness guidance and is not medical diagnosis "
        "or treatment. In emergencies, contact local emergency services."
    ),
}
SYSTEM_USER_EMAILS = {"mim-bot@coachmim.local", "admin-team@coachmim.local"}
MASS_UNIT_TO_GRAMS = {
    "g": 1.0,
    "oz": 28.349523125,
    "lb": 453.59237,
}
VOLUME_UNIT_TO_ML = {
    "ml": 1.0,
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "oz": 29.5735,
}

PROFILE_ENCRYPTED_FIELDS = [
    "phone",
    "known_sleep_issues",
    "family_history_flags",
    "medications",
    "supplements",
    "food_intolerances",
    "food_sensitivities",
    "caffeine_baseline",
    "nicotine_use",
    "recreational_drug_use",
    "attention_issues",
    "burnout_history",
    "secondary_goals",
    "great_day_definition",
    "digestive_sensitivity",
    "stress_reactivity",
    "social_pattern",
]
CHECKIN_ENCRYPTED_FIELDS = [
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
]
MEAL_ENCRYPTED_FIELDS = [
    "label",
    "description",
    "portion_notes",
    "tags",
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "sugar_g",
    "sodium_mg",
    "caffeine_mg",
]
FAVORITE_ENCRYPTED_FIELDS = [
    "label",
    "description",
    "portion_notes",
    "tags",
    "is_beverage",
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "sugar_g",
    "sodium_mg",
    "caffeine_mg",
    "ingredients",
]
SUBSTANCE_ENCRYPTED_FIELDS = ["amount", "notes"]
CHAT_ENCRYPTED_FIELDS = ["content"]


def hydrate_profile_secure_fields(user: User, profile: UserProfile | None):
    if user is None or profile is None:
        return
    hydrate_model_fields(
        user=user,
        model=profile,
        encrypted_attr="encrypted_sensitive_payload",
        fields=PROFILE_ENCRYPTED_FIELDS,
        scope="user_profile",
    )


def hydrate_checkin_secure_fields(user: User, record: DailyCheckIn | None):
    if user is None or record is None:
        return
    hydrate_model_fields(
        user=user,
        model=record,
        encrypted_attr="encrypted_payload",
        fields=CHECKIN_ENCRYPTED_FIELDS,
        scope="daily_checkin",
    )


def hydrate_meal_secure_fields(user: User, meal: Meal | None):
    if user is None or meal is None:
        return
    hydrate_model_fields(
        user=user,
        model=meal,
        encrypted_attr="encrypted_payload",
        fields=MEAL_ENCRYPTED_FIELDS,
        scope="meal",
    )


def hydrate_substance_secure_fields(user: User, entry: Substance | None):
    if user is None or entry is None:
        return
    hydrate_model_fields(
        user=user,
        model=entry,
        encrypted_attr="encrypted_payload",
        fields=SUBSTANCE_ENCRYPTED_FIELDS,
        scope="substance",
    )


def hydrate_favorite_secure_fields(user: User, favorite: FavoriteMeal | None):
    if user is None or favorite is None:
        return
    hydrate_model_fields(
        user=user,
        model=favorite,
        encrypted_attr="encrypted_payload",
        fields=FAVORITE_ENCRYPTED_FIELDS,
        scope="favorite_meal",
    )


def persist_profile_secure_fields(user: User, profile: UserProfile):
    encrypt_model_fields(
        user=user,
        model=profile,
        encrypted_attr="encrypted_sensitive_payload",
        fields=PROFILE_ENCRYPTED_FIELDS,
        scope="user_profile",
    )


def persist_checkin_secure_fields(user: User, record: DailyCheckIn):
    encrypt_model_fields(
        user=user,
        model=record,
        encrypted_attr="encrypted_payload",
        fields=CHECKIN_ENCRYPTED_FIELDS,
        scope="daily_checkin",
    )


def persist_meal_secure_fields(user: User, meal: Meal):
    encrypt_model_fields(
        user=user,
        model=meal,
        encrypted_attr="encrypted_payload",
        fields=MEAL_ENCRYPTED_FIELDS,
        scope="meal",
    )


def persist_substance_secure_fields(user: User, entry: Substance):
    encrypt_model_fields(
        user=user,
        model=entry,
        encrypted_attr="encrypted_payload",
        fields=SUBSTANCE_ENCRYPTED_FIELDS,
        scope="substance",
    )


def persist_favorite_secure_fields(user: User, favorite: FavoriteMeal):
    encrypt_model_fields(
        user=user,
        model=favorite,
        encrypted_attr="encrypted_payload",
        fields=FAVORITE_ENCRYPTED_FIELDS,
        scope="favorite_meal",
    )


def hydrate_chat_secure_fields(user: User, message: MIMChatMessage | None):
    if user is None or message is None:
        return
    hydrate_model_fields(
        user=user,
        model=message,
        encrypted_attr="encrypted_payload",
        fields=CHAT_ENCRYPTED_FIELDS,
        scope="mim_chat_message",
    )


def persist_chat_secure_fields(user: User, message: MIMChatMessage):
    encrypt_model_fields(
        user=user,
        model=message,
        encrypted_attr="encrypted_payload",
        fields=CHAT_ENCRYPTED_FIELDS,
        scope="mim_chat_message",
    )


def _favorite_scope_from_tags(tags: list[str] | None):
    if not tags:
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(DAY_MANAGER_FAVORITE_SCOPE_PREFIX):
            scope = tag[len(DAY_MANAGER_FAVORITE_SCOPE_PREFIX) :].strip().lower()
            if scope:
                return scope
    return None


def _apply_favorite_scope(tags: list[str] | None, scope: str | None):
    cleaned_tags = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        trimmed = tag.strip()
        if not trimmed:
            continue
        if trimmed.startswith(DAY_MANAGER_FAVORITE_SCOPE_PREFIX):
            continue
        cleaned_tags.append(trimmed)

    if scope:
        cleaned_tags.append(f"{DAY_MANAGER_FAVORITE_SCOPE_PREFIX}{scope}")
    return cleaned_tags or None


def _build_day_manager_favorites_for_user(user_id: int):
    user = g.get("user")
    if user is None or user.id != user_id:
        user = db.session.get(User, user_id)

    favorites = (
        FavoriteMeal.query.filter_by(user_id=user_id)
        .order_by(FavoriteMeal.updated_at.desc(), FavoriteMeal.name.asc())
        .limit(200)
        .all()
    )
    grouped = {
        "meal": [],
        "drink": [],
        "substance": [],
        "activity": [],
        "medications": [],
    }

    for favorite in favorites:
        hydrate_favorite_secure_fields(user, favorite)
        scope = _favorite_scope_from_tags(favorite.tags)
        payload = favorite.ingredients if isinstance(favorite.ingredients, dict) else {}
        favorite_base = {
            "id": favorite.id,
            "name": favorite.name,
            "label": favorite.label or "",
            "description": favorite.description or "",
            "portion_notes": favorite.portion_notes or "",
            "calories": favorite.calories if favorite.calories is not None else "",
            "protein_g": favorite.protein_g if favorite.protein_g is not None else "",
            "carbs_g": favorite.carbs_g if favorite.carbs_g is not None else "",
            "fat_g": favorite.fat_g if favorite.fat_g is not None else "",
            "sugar_g": favorite.sugar_g if favorite.sugar_g is not None else "",
            "sodium_mg": favorite.sodium_mg if favorite.sodium_mg is not None else "",
            "caffeine_mg": favorite.caffeine_mg if favorite.caffeine_mg is not None else "",
        }

        if scope == "substance":
            grouped["substance"].append(
                {
                    "id": favorite.id,
                    "name": favorite.name,
                    "kind": str(payload.get("kind") or "other").lower(),
                    "amount": str(payload.get("amount") or favorite.portion_notes or favorite.description or ""),
                    "notes": str(payload.get("notes") or ""),
                }
            )
            continue

        if scope == "activity":
            grouped["activity"].append(
                {
                    "id": favorite.id,
                    "name": favorite.name,
                    "activity_type": str(payload.get("activity_type") or favorite.description or ""),
                    "duration_min": payload.get("duration_min") if payload.get("duration_min") is not None else "",
                    "intensity": payload.get("intensity") if payload.get("intensity") is not None else "",
                    "notes": str(payload.get("notes") or ""),
                }
            )
            continue

        if scope == "medications":
            grouped["medications"].append(
                {
                    "id": favorite.id,
                    "name": favorite.name,
                    "kind": str(payload.get("kind") or "medication").lower(),
                    "med_name": str(payload.get("med_name") or favorite.description or ""),
                    "dose": str(payload.get("dose") or favorite.portion_notes or ""),
                    "notes": str(payload.get("notes") or ""),
                }
            )
            continue

        if favorite.is_beverage or scope == "drink":
            grouped["drink"].append(favorite_base)
        else:
            grouped["meal"].append(favorite_base)

    return grouped


def _find_user_quick_favorite(user_id: int, favorite_id: int | None):
    if favorite_id is None:
        return None
    favorite = FavoriteMeal.query.filter_by(user_id=user_id, id=favorite_id).first()
    user = g.get("user")
    if user is None or user.id != user_id:
        user = db.session.get(User, user_id)
    hydrate_favorite_secure_fields(user, favorite)
    return favorite


def _favorite_scope_matches(existing_scope: str | None, target_scope: str):
    if target_scope in {"meal", "drink"}:
        return existing_scope in {None, "meal", "drink"}
    return existing_scope == target_scope


def _resolve_favorite_slot(user_id: int, requested_name: str, scope: str):
    normalized_name = normalize_text(requested_name)
    if not normalized_name:
        return (None, None)
    normalized_name = normalized_name[:120]

    existing = FavoriteMeal.query.filter_by(user_id=user_id, name=normalized_name).first()
    user = g.get("user")
    if user is None or user.id != user_id:
        user = db.session.get(User, user_id)
    hydrate_favorite_secure_fields(user, existing)
    if existing is None:
        return (normalized_name, None)

    existing_scope = _favorite_scope_from_tags(existing.tags)
    if _favorite_scope_matches(existing_scope, scope):
        return (normalized_name, existing)

    suffix = f" ({scope})"
    max_base_len = max(1, 120 - len(suffix))
    alt_name = f"{normalized_name[:max_base_len]}{suffix}"
    alt_existing = FavoriteMeal.query.filter_by(user_id=user_id, name=alt_name).first()
    hydrate_favorite_secure_fields(user, alt_existing)
    return (alt_name, alt_existing)


def _save_day_manager_meal_favorite(meal: Meal, view: str):
    if not parse_bool(request.form.get("save_favorite")):
        return

    favorite_scope = "drink" if view == "drink" else "meal"
    favorite_name = normalize_text(request.form.get("favorite_name"))
    if not favorite_name:
        favorite_name = derive_name_from_meal_request(meal=meal, max_len=120)
    if not favorite_name:
        return

    favorite_name, favorite = _resolve_favorite_slot(g.user.id, favorite_name, favorite_scope)
    if not favorite_name:
        return
    if not favorite:
        favorite = FavoriteMeal(user_id=g.user.id, name=favorite_name)

    favorite.name = favorite_name
    favorite.food_item_id = meal.food_item_id
    favorite.label = meal.label
    favorite.description = meal.description
    favorite.portion_notes = meal.portion_notes
    favorite.tags = _apply_favorite_scope(meal.tags, favorite_scope)
    favorite.is_beverage = meal.is_beverage
    favorite.calories = meal.calories
    favorite.protein_g = meal.protein_g
    favorite.carbs_g = meal.carbs_g
    favorite.fat_g = meal.fat_g
    favorite.sugar_g = meal.sugar_g
    favorite.sodium_mg = meal.sodium_mg
    favorite.caffeine_mg = meal.caffeine_mg
    favorite.ingredients = None

    persist_favorite_secure_fields(g.user, favorite)
    db.session.add(favorite)


def _save_day_manager_nonmeal_favorite(scope: str, *, label: str, description: str | None, portion_notes: str | None, payload: dict):
    if not parse_bool(request.form.get("save_favorite")):
        return

    favorite_name = normalize_text(request.form.get("favorite_name"))
    if not favorite_name:
        favorite_name = normalize_text(description) or normalize_text(portion_notes) or label
    if not favorite_name:
        return

    favorite_name, favorite = _resolve_favorite_slot(g.user.id, favorite_name, scope)
    if not favorite_name:
        return
    if not favorite:
        favorite = FavoriteMeal(user_id=g.user.id, name=favorite_name)

    favorite.name = favorite_name
    favorite.food_item_id = None
    favorite.label = label
    favorite.description = normalize_text(description)
    favorite.portion_notes = normalize_text(portion_notes)
    favorite.tags = _apply_favorite_scope(None, scope)
    favorite.is_beverage = False
    favorite.calories = None
    favorite.protein_g = None
    favorite.carbs_g = None
    favorite.fat_g = None
    favorite.sugar_g = None
    favorite.sodium_mg = None
    favorite.caffeine_mg = None
    favorite.ingredients = payload
    persist_favorite_secure_fields(g.user, favorite)
    db.session.add(favorite)


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


def normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null"}:
        return None
    return text


def normalize_community_category(value: str | None) -> str:
    key = (value or "").strip().lower()
    return key if key in COMMUNITY_CATEGORY_KEYS else "general"


def get_community_filter_or_all(value: str | None) -> str:
    key = (value or "").strip().lower()
    if key == "all":
        return "all"
    return normalize_community_category(key)


def community_display_name(user: User | None) -> str:
    if user is None:
        return "Member"
    raw = (user.full_name or "").strip()
    if not raw:
        return "Member"
    pieces = [part for part in raw.split(" ") if part]
    if len(pieces) < 2:
        return pieces[0][:36]
    return f"{pieces[0][:20]} {pieces[-1][:1].upper()}."


def build_community_share_links(post: CommunityPost):
    post_url = url_for("main.community_page", _external=True) + f"#post-{post.id}"
    title = quote_plus(post.title or "CoachMIM Community Post")
    encoded_url = quote_plus(post_url)
    return {
        "copy_url": post_url,
        "x": f"https://x.com/intent/tweet?text={title}&url={encoded_url}",
        "facebook": f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}",
        "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}",
        "reddit": f"https://www.reddit.com/submit?url={encoded_url}&title={title}",
        "email": f"mailto:?subject={title}&body={encoded_url}",
    }


def normalize_search_text(value: str | None) -> str:
    raw = (value or "").lower()
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def tokenize_search_text(value: str | None) -> list[str]:
    normalized = normalize_search_text(value)
    if not normalized:
        return []
    tokens = [token for token in normalized.split() if len(token) >= 2 and token not in SEARCH_STOPWORDS]
    return tokens or [normalized]


def parse_ingredients_json(raw_value):
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    def safe_float(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def safe_int(value):
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    cleaned = []
    for item in parsed[:60]:
        if not isinstance(item, dict):
            continue
        cleaned_item = {
            "food_item_id": safe_int(item.get("food_item_id")),
            "food_name": (str(item.get("food_name") or "").strip()[:255] or None),
            "quantity": safe_float(item.get("quantity")),
            "unit": (str(item.get("unit") or "").strip()[:32] or None),
            "serving_size": safe_float(item.get("serving_size")),
            "serving_unit": (str(item.get("serving_unit") or "").strip()[:32] or None),
            "calories": safe_float(item.get("calories")),
            "protein_g": safe_float(item.get("protein_g")),
            "carbs_g": safe_float(item.get("carbs_g")),
            "fat_g": safe_float(item.get("fat_g")),
            "sugar_g": safe_float(item.get("sugar_g")),
            "sodium_mg": safe_float(item.get("sodium_mg")),
            "caffeine_mg": safe_float(item.get("caffeine_mg")),
        }
        cleaned.append(cleaned_item)
    return cleaned or None


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


def first_pending_checkin_tab(segments: dict[str, bool]) -> str:
    for tab in ["sleep", "morning", "midday", "evening", "overall"]:
        if not segments.get(tab):
            return tab
    return "overall"


def resolve_checkin_default_tab(*, segments: dict[str, bool], is_viewing_today: bool, local_hour: int) -> str:
    if not segments.get("sleep"):
        return "sleep"
    if not is_viewing_today:
        return first_pending_checkin_tab(segments)
    if local_hour < 11:
        return "morning"
    if local_hour < 17:
        return "midday"
    return "evening"


def day_bounds(target_day: date):
    start = datetime.combine(target_day, datetime.min.time())
    end = start + timedelta(days=1)
    return start, end


def sum_meal_nutrient(day_meals: list[Meal], field_name: str) -> float:
    total = 0.0
    for meal in day_meals:
        value = getattr(meal, field_name)
        if value is not None:
            total += float(value)
    return total


def meal_context_text(meal: Meal) -> str:
    parts = [meal.label, meal.description]
    if meal.food_item:
        parts.extend([meal.food_item.name, meal.food_item.brand])
    if meal.tags:
        parts.extend(meal.tags)
    return " ".join(str(part) for part in parts if part).lower()


def estimate_sugar_sources(day_meals: list[Meal]) -> tuple[float, float]:
    added_sugar = 0.0
    natural_sugar = 0.0

    for meal in day_meals:
        sugar = float(meal.sugar_g or 0)
        if sugar <= 0:
            continue

        text = meal_context_text(meal)
        has_added_hint = any(token in text for token in ADDED_SUGAR_HINTS)
        has_natural_hint = any(token in text for token in NATURAL_SUGAR_HINTS)

        if has_added_hint and not has_natural_hint:
            added_sugar += sugar
        elif has_natural_hint and not has_added_hint:
            natural_sugar += sugar
        elif has_added_hint and has_natural_hint:
            added_sugar += sugar * 0.5
            natural_sugar += sugar * 0.5
        elif meal.is_beverage:
            added_sugar += sugar * 0.7
            natural_sugar += sugar * 0.3
        else:
            added_sugar += sugar * 0.5
            natural_sugar += sugar * 0.5

    return (round(added_sugar, 1), round(natural_sugar, 1))


def build_meal_summary(selected_day: date, day_meals: list[Meal]) -> dict:
    calories_total = round(sum_meal_nutrient(day_meals, "calories"))
    carbs_total = round(sum_meal_nutrient(day_meals, "carbs_g"), 1)
    sugar_total = round(sum_meal_nutrient(day_meals, "sugar_g"), 1)
    protein_total = round(sum_meal_nutrient(day_meals, "protein_g"), 1)
    caffeine_total = round(sum_meal_nutrient(day_meals, "caffeine_mg"), 1)
    added_sugar_total, natural_sugar_total = estimate_sugar_sources(day_meals)

    entries_without_nutrition = sum(
        1
        for meal in day_meals
        if all(
            getattr(meal, field) is None
            for field in ["calories", "protein_g", "carbs_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"]
        )
    )

    mim_notes = []
    if not day_meals:
        mim_notes.append("No meals logged for this day yet. Add your first entry to start your signal.")
    else:
        if protein_total < 30:
            mim_notes.append("Protein is low so far. Add a protein anchor (eggs, yogurt, fish, chicken, tofu, beans).")
        if carbs_total >= 240 and protein_total < 70:
            mim_notes.append("Day is carb-heavy versus protein. Add protein + fiber to reduce energy swings.")
        if added_sugar_total >= 45:
            mim_notes.append("Added sugar is high. Consider swapping one sweetened item for whole-food carbs.")
        if caffeine_total >= 400:
            mim_notes.append("Caffeine is high today (>400mg). Expect possible sleep and anxiety impact.")
        elif caffeine_total >= 250:
            mim_notes.append("Caffeine is moderate-high. Keep late-day caffeine low to protect sleep.")
        elif any(meal.is_beverage for meal in day_meals) and caffeine_total == 0:
            mim_notes.append("Drinks are logged but caffeine is missing. Add caffeine mg for cleaner performance analysis.")
        if entries_without_nutrition > 0:
            mim_notes.append(
                f"{entries_without_nutrition} entr{'y' if entries_without_nutrition == 1 else 'ies'} "
                "are missing nutrition values. Fill them to improve pattern detection."
            )
        if selected_day < date.today() and calories_total < 900 and len(day_meals) <= 2:
            mim_notes.append("This past day may be under-logged. Add missed meals/drinks for cleaner analysis.")

    if not mim_notes and day_meals:
        mim_notes.append("Day looks balanced so far. Keep logging timing and portions to preserve signal quality.")

    return {
        "calories_total": int(calories_total),
        "carbs_total": carbs_total,
        "sugar_total": sugar_total,
        "protein_total": protein_total,
        "caffeine_total": caffeine_total,
        "added_sugar_total": added_sugar_total,
        "natural_sugar_total": natural_sugar_total,
        "entries_count": len(day_meals),
        "entries_without_nutrition": entries_without_nutrition,
        "mim_notes": mim_notes,
    }


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
    for meal in day_meals:
        hydrate_meal_secure_fields(g.user, meal)
    all_favorites = (
        FavoriteMeal.query.filter_by(user_id=g.user.id)
        .order_by(FavoriteMeal.updated_at.desc(), FavoriteMeal.name.asc())
        .all()
    )
    for favorite in all_favorites:
        hydrate_favorite_secure_fields(g.user, favorite)
    favorites = [
        fav
        for fav in all_favorites
        if _favorite_scope_from_tags(fav.tags) in {None, "meal", "drink"}
    ]
    meal_summary = build_meal_summary(selected_day, day_meals)
    local_today = get_user_local_today(g.user)
    local_now = datetime.now(get_user_zoneinfo(g.user))

    favorite_payload = [
        {
            "id": f.id,
            "name": f.name,
            "label": f.label,
            "food_item_id": f.food_item_id,
            "description": f.description,
            "portion_notes": f.portion_notes,
            "tags": ", ".join(_apply_favorite_scope(f.tags, None) or []),
            "calories": f.calories,
            "protein_g": f.protein_g,
            "carbs_g": f.carbs_g,
            "fat_g": f.fat_g,
            "sugar_g": f.sugar_g,
            "sodium_mg": f.sodium_mg,
            "caffeine_mg": f.caffeine_mg,
            "is_beverage": f.is_beverage,
            "ingredients": f.ingredients if isinstance(f.ingredients, list) else [],
        }
        for f in favorites
    ]

    default_time = local_now.strftime("%H:%M")

    return {
        "selected_day": selected_day.isoformat(),
        "selected_day_weekday": selected_day.strftime("%A"),
        "selected_day_pretty": selected_day.strftime("%B %d, %Y"),
        "prev_day": (selected_day - timedelta(days=1)).isoformat(),
        "next_day": (selected_day + timedelta(days=1)).isoformat(),
        "can_go_next": selected_day < local_today,
        "day_meals": day_meals,
        "meal_summary": meal_summary,
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
    meal.label = normalize_text(request.form.get("label"))
    meal.description = normalize_text(request.form.get("description"))
    meal.portion_notes = normalize_text(request.form.get("portion_notes"))
    meal.tags = parse_tags(request.form.get("tags"))
    meal.is_beverage = parse_bool(request.form.get("is_beverage"))

    meal.calories = parse_int(request.form.get("calories"))
    meal.protein_g = parse_float(request.form.get("protein_g"))
    meal.carbs_g = parse_float(request.form.get("carbs_g"))
    meal.fat_g = parse_float(request.form.get("fat_g"))
    meal.sugar_g = parse_float(request.form.get("sugar_g"))
    meal.sodium_mg = parse_float(request.form.get("sodium_mg"))
    meal.caffeine_mg = parse_float(request.form.get("caffeine_mg"))

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
        if meal.caffeine_mg is None:
            meal.caffeine_mg = food_item.caffeine_mg

    return eaten_at_dt


def meal_has_meaningful_content(meal: Meal, has_new_photo: bool = False) -> bool:
    if meal.food_item_id is not None:
        return True
    if any(value not in (None, "") for value in [meal.label, meal.description, meal.portion_notes]):
        return True
    if meal.tags:
        return True
    if any(
        getattr(meal, field) is not None
        for field in ["calories", "protein_g", "carbs_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"]
    ):
        return True
    if meal.photo_path or has_new_photo:
        return True
    return False


def derive_name_from_meal_request(meal: Meal | None = None, max_len: int = 255):
    explicit_favorite_name = normalize_text(request.form.get("favorite_name"))
    if explicit_favorite_name:
        return explicit_favorite_name[:max_len]

    builder_title = normalize_text(request.form.get("builder_title"))
    if builder_title and builder_title.lower() not in GENERIC_MEAL_NAMES:
        return builder_title[:max_len]

    label = normalize_text(request.form.get("label")) or (meal.label if meal else None)
    if label and label.lower() not in GENERIC_MEAL_NAMES:
        return label[:max_len]

    description = normalize_text(request.form.get("description")) or (meal.description if meal else None)
    if description:
        primary = normalize_text(description.split(":", 1)[0])
        if primary and primary.lower() not in GENERIC_MEAL_NAMES:
            return primary[:max_len]
        if description.lower() not in GENERIC_MEAL_NAMES:
            return description[:max_len]

    if meal and meal.food_item_id:
        linked = db.session.get(FoodItem, meal.food_item_id)
        if linked:
            linked_name = normalize_text(linked.display_name())
            if linked_name:
                return linked_name[:max_len]

    return None


def parse_serving_hint(portion_notes: str | None):
    text = normalize_text(portion_notes)
    if not text:
        return (None, None)
    match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text)
    if not match:
        return (None, None)

    value = parse_float(match.group(1))
    unit = (match.group(2) or "").strip().lower()
    unit_map = {
        "grams": "g",
        "gram": "g",
        "g": "g",
        "milliliter": "ml",
        "milliliters": "ml",
        "ml": "ml",
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "tbsp": "tbsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "tsp": "tsp",
        "cup": "cup",
        "cups": "cup",
        "oz": "oz",
        "ounce": "oz",
        "ounces": "oz",
        "item": "item",
        "items": "item",
        "slice": "item",
        "slices": "item",
    }
    mapped = unit_map.get(unit, unit[:32] if unit else None)
    return (value, mapped)


def upsert_shared_food_from_request(meal: Meal):
    if meal.food_item_id is not None:
        return None
    if not parse_bool(request.form.get("save_shared_food")):
        return None

    shared_name = derive_name_from_meal_request(meal=meal, max_len=255)
    if not shared_name:
        return None

    has_any_nutrition = any(
        getattr(meal, field) is not None
        for field in ["calories", "protein_g", "carbs_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"]
    )
    if not has_any_nutrition:
        return None

    brand_value = normalize_text(request.form.get("shared_brand"))
    if brand_value:
        brand_value = brand_value[:255]
    name_key = shared_name.lower()
    brand_key = (brand_value or "").lower()

    query = FoodItem.query.filter(
        FoodItem.source == "community",
        func.lower(FoodItem.name) == name_key,
    )
    if brand_key:
        query = query.filter(func.lower(FoodItem.brand) == brand_key)
    else:
        query = query.filter(or_(FoodItem.brand.is_(None), FoodItem.brand == ""))

    existing = query.first()
    serving_size, serving_unit = parse_serving_hint(meal.portion_notes)

    if existing:
        if existing.brand is None and brand_value:
            existing.brand = brand_value
        if existing.serving_size is None and serving_size is not None:
            existing.serving_size = serving_size
        if existing.serving_unit is None and serving_unit:
            existing.serving_unit = serving_unit
        if existing.calories is None and meal.calories is not None:
            existing.calories = meal.calories
        if existing.protein_g is None and meal.protein_g is not None:
            existing.protein_g = meal.protein_g
        if existing.carbs_g is None and meal.carbs_g is not None:
            existing.carbs_g = meal.carbs_g
        if existing.fat_g is None and meal.fat_g is not None:
            existing.fat_g = meal.fat_g
        if existing.sugar_g is None and meal.sugar_g is not None:
            existing.sugar_g = meal.sugar_g
        if existing.sodium_mg is None and meal.sodium_mg is not None:
            existing.sodium_mg = meal.sodium_mg
        if existing.caffeine_mg is None and meal.caffeine_mg is not None:
            existing.caffeine_mg = meal.caffeine_mg
        db.session.add(existing)
        db.session.flush()
        return existing

    shared_food = FoodItem(
        external_id=f"community:{uuid4().hex}",
        name=shared_name,
        brand=brand_value,
        serving_size=serving_size,
        serving_unit=serving_unit,
        calories=meal.calories,
        protein_g=meal.protein_g,
        carbs_g=meal.carbs_g,
        fat_g=meal.fat_g,
        sugar_g=meal.sugar_g,
        sodium_mg=meal.sodium_mg,
        caffeine_mg=meal.caffeine_mg,
        source="community",
    )
    db.session.add(shared_food)
    db.session.flush()
    return shared_food


def upsert_favorite_from_request(meal: Meal | None = None):
    if not parse_bool(request.form.get("save_favorite")):
        return

    favorite_name = normalize_text(request.form.get("favorite_name"))
    if not favorite_name:
        favorite_name = derive_name_from_meal_request(meal=meal, max_len=120)
    if not favorite_name:
        return

    favorite = FavoriteMeal.query.filter_by(user_id=g.user.id, name=favorite_name).first()
    hydrate_favorite_secure_fields(g.user, favorite)
    if not favorite:
        favorite = FavoriteMeal(user_id=g.user.id, name=favorite_name)

    favorite.food_item_id = parse_int(request.form.get("food_item_id"))
    favorite.label = normalize_text(request.form.get("label"))
    favorite.description = normalize_text(request.form.get("description"))
    favorite.portion_notes = normalize_text(request.form.get("portion_notes"))
    favorite.tags = parse_tags(request.form.get("tags"))
    favorite.is_beverage = parse_bool(request.form.get("is_beverage"))

    favorite.calories = parse_int(request.form.get("calories"))
    favorite.protein_g = parse_float(request.form.get("protein_g"))
    favorite.carbs_g = parse_float(request.form.get("carbs_g"))
    favorite.fat_g = parse_float(request.form.get("fat_g"))
    favorite.sugar_g = parse_float(request.form.get("sugar_g"))
    favorite.sodium_mg = parse_float(request.form.get("sodium_mg"))
    favorite.caffeine_mg = parse_float(request.form.get("caffeine_mg"))
    favorite.ingredients = parse_ingredients_json(request.form.get("favorite_ingredients"))

    persist_favorite_secure_fields(g.user, favorite)
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


def average_or_none(values: list[float], digits: int = 1):
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), digits)


def _parse_clock_token(token: str):
    text = (token or "").strip().lower().replace(".", "")
    if not text:
        return None

    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3)
    if minute > 59:
        return None

    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    else:
        if hour > 23:
            return None

    return hour * 60 + minute


def parse_workout_minutes(workout_timing: str | None):
    text = normalize_text(workout_timing)
    if not text:
        return None

    normalized = text.lower()

    # Duration like "1 hr 20 min" or "45 min".
    hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", normalized)
    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", normalized)
    if hour_match or minute_match:
        hours = float(hour_match.group(1)) if hour_match else 0.0
        minutes = float(minute_match.group(1)) if minute_match else 0.0
        total = int(round((hours * 60) + minutes))
        return total if total > 0 else None

    # Time range like "7:00am-8:15am" or "18:00 to 19:00".
    range_match = re.search(
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:-|to)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        normalized,
    )
    if range_match:
        start_minutes = _parse_clock_token(range_match.group(1))
        end_minutes = _parse_clock_token(range_match.group(2))
        if start_minutes is not None and end_minutes is not None:
            diff = end_minutes - start_minutes
            if diff < 0:
                diff += 24 * 60
            if 0 < diff <= 8 * 60:
                return diff

    return None


def checkin_metric_value(record: DailyCheckIn, overall_field: str, segment_fields: list[str]):
    overall_value = getattr(record, overall_field, None)
    if overall_value is not None:
        return float(overall_value)

    segment_values = [getattr(record, field, None) for field in segment_fields]
    return average_or_none([float(v) for v in segment_values if v is not None], digits=2)


def build_home_weekly_context(user: User, profile: UserProfile):
    local_today = get_user_local_today(user)
    week_start = local_today - timedelta(days=6)

    checkins = (
        DailyCheckIn.query.filter(
            DailyCheckIn.user_id == user.id,
            DailyCheckIn.day >= week_start,
            DailyCheckIn.day <= local_today,
        )
        .order_by(DailyCheckIn.day.asc())
        .all()
    )
    for entry in checkins:
        hydrate_checkin_secure_fields(user, entry)

    meals = (
        Meal.query.filter(
            Meal.user_id == user.id,
            Meal.eaten_at >= datetime.combine(week_start, datetime.min.time()),
            Meal.eaten_at < datetime.combine(local_today + timedelta(days=1), datetime.min.time()),
        )
        .all()
    )
    for meal in meals:
        hydrate_meal_secure_fields(user, meal)

    days_with_checkins = {entry.day for entry in checkins}
    coverage_pct = round((len(days_with_checkins) / 7) * 100, 1)

    workout_sessions = 0
    workout_minutes = 0
    for entry in checkins:
        has_workout = bool(normalize_text(entry.workout_timing)) or entry.workout_intensity is not None
        if has_workout:
            workout_sessions += 1
        parsed_minutes = parse_workout_minutes(entry.workout_timing)
        if parsed_minutes:
            workout_minutes += parsed_minutes

    calorie_values = [meal.calories for meal in meals if meal.calories is not None]
    avg_calories_per_logged_day = average_or_none(calorie_values, digits=0)
    total_calories_week = int(round(sum(float(v) for v in calorie_values))) if calorie_values else 0

    avg_sleep = average_or_none([entry.sleep_hours for entry in checkins if entry.sleep_hours is not None], digits=2)
    avg_energy = average_or_none(
        [
            checkin_metric_value(entry, "energy", ["morning_energy", "midday_energy", "evening_energy"])
            for entry in checkins
        ],
        digits=1,
    )
    avg_focus = average_or_none(
        [
            checkin_metric_value(entry, "focus", ["morning_focus", "midday_focus", "evening_focus"])
            for entry in checkins
        ],
        digits=1,
    )
    avg_mood = average_or_none(
        [
            checkin_metric_value(entry, "mood", ["morning_mood", "midday_mood", "evening_mood"])
            for entry in checkins
        ],
        digits=1,
    )
    avg_stress = average_or_none(
        [
            checkin_metric_value(entry, "stress", ["morning_stress", "midday_stress", "evening_stress"])
            for entry in checkins
        ],
        digits=1,
    )
    avg_productivity = average_or_none(
        [entry.productivity for entry in checkins if entry.productivity is not None], digits=1
    )
    avg_anxiety = average_or_none([entry.anxiety for entry in checkins if entry.anxiety is not None], digits=1)

    mim_notes = []
    primary_goal = normalize_text(profile.primary_goal) or "Consistency"

    if coverage_pct < 60:
        mim_notes.append("Data coverage is low this week. More daily check-ins will improve model quality.")
    if workout_sessions == 0:
        mim_notes.append("No workouts logged this week. Add workout timing/intensity to improve performance signal.")
    elif workout_minutes > 0 and workout_minutes < 90:
        mim_notes.append("Workout time is light this week. Add one more session to strengthen signal quality.")

    goal_text = primary_goal.lower()
    if "sleep" in goal_text:
        if avg_sleep is not None and avg_sleep < 7:
            mim_notes.append("Sleep goal is off-track this week (avg below 7h). Protect sleep window and evening routine.")
        elif avg_sleep is not None:
            mim_notes.append("Sleep goal trend is solid this week. Keep bedtime and wake time consistent.")
    if "focus" in goal_text or "productivity" in goal_text:
        if avg_focus is not None and avg_focus < 6:
            mim_notes.append("Focus trend is softer than target. Review sleep + lunch carb load + caffeine timing.")
        if avg_productivity is not None and avg_productivity < 6:
            mim_notes.append("Productivity average is low this week. Try one planned deep-work block each morning.")
    if "energy" in goal_text:
        if avg_energy is not None and avg_energy < 6:
            mim_notes.append("Energy trend is low this week. Tighten meal timing and hydration consistency.")
    if "anxiety" in goal_text or "stress" in goal_text:
        if avg_stress is not None and avg_stress > 6:
            mim_notes.append("Stress is elevated this week. Add one recovery block and reduce late-day stimulants.")
        if avg_anxiety is not None and avg_anxiety > 5:
            mim_notes.append("Anxiety trend is elevated this week. Track alcohol, sleep quality, and caffeine more tightly.")

    if not mim_notes:
        mim_notes.append("Weekly trend looks stable. Keep logging meals and check-ins to strengthen pattern detection.")

    return {
        "week_start": week_start,
        "week_end": local_today,
        "coverage_pct": coverage_pct,
        "workout_sessions": workout_sessions,
        "workout_minutes": workout_minutes,
        "avg_calories_per_logged_day": avg_calories_per_logged_day,
        "total_calories_week": total_calories_week,
        "meals_logged_week": len(meals),
        "checkins_logged_week": len(checkins),
        "avg_sleep": avg_sleep,
        "avg_energy": avg_energy,
        "avg_focus": avg_focus,
        "avg_mood": avg_mood,
        "avg_stress": avg_stress,
        "avg_productivity": avg_productivity,
        "avg_anxiety": avg_anxiety,
        "primary_goal": primary_goal,
        "mim_notes": mim_notes,
    }


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


def ensure_default_admin_user():
    admin = AdminUser.query.filter_by(username=ADMIN_BOOTSTRAP_USERNAME).first()
    if admin:
        return admin
    admin = AdminUser(
        username=ADMIN_BOOTSTRAP_USERNAME,
        password_hash=generate_password_hash(ADMIN_BOOTSTRAP_PASSWORD),
    )
    db.session.add(admin)
    db.session.commit()
    return admin


def get_site_setting(key: str):
    row = SiteSetting.query.filter_by(key=key).first()
    if row and row.value is not None:
        return row.value
    return SITE_SETTING_DEFAULTS.get(key)


def get_site_settings(keys: list[str]):
    values = {}
    rows = SiteSetting.query.filter(SiteSetting.key.in_(keys)).all() if keys else []
    row_map = {row.key: row.value for row in rows}
    for key in keys:
        current_value = row_map.get(key)
        values[key] = current_value if current_value not in (None, "") else SITE_SETTING_DEFAULTS.get(key)
    return values


def set_site_setting(key: str, value: str | None):
    row = SiteSetting.query.filter_by(key=key).first()
    if not row:
        row = SiteSetting(key=key)
    row.value = value
    db.session.add(row)
    return row


def add_blocked_email(email: str | None, reason: str, admin_id: int | None):
    normalized = normalize_email(email)
    if not normalized:
        return
    existing = BlockedEmail.query.filter_by(email=normalized).first()
    if existing:
        existing.reason = reason
        existing.created_by_admin_id = admin_id
        db.session.add(existing)
        return
    db.session.add(
        BlockedEmail(
            email=normalized,
            reason=reason,
            created_by_admin_id=admin_id,
        )
    )


def remove_blocked_email(email: str | None):
    normalized = normalize_email(email)
    if not normalized:
        return
    BlockedEmail.query.filter_by(email=normalized).delete(synchronize_session=False)


def hard_delete_user_account(user_id: int):
    CommunityLike.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    CommunityComment.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    CommunityPost.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    MIMChatMessage.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    FavoriteMeal.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    Meal.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    Substance.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    DailyCheckIn.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    UserProfile.query.filter_by(user_id=user_id).delete(synchronize_session=False)
    User.query.filter_by(id=user_id).delete(synchronize_session=False)


def get_or_create_system_user(*, email: str, full_name: str):
    bot_email = normalize_email(email)
    if not bot_email:
        raise ValueError("system user email is required")
    bot = User.query.filter_by(email=bot_email).first()
    if bot:
        return bot
    bot = User(
        full_name=full_name,
        email=bot_email,
        password_hash=None,
        is_blocked=False,
        is_spam=False,
    )
    db.session.add(bot)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        return User.query.filter_by(email=bot_email).first()
    db.session.add(UserProfile(user_id=bot.id))
    return bot


def get_or_create_profile(user: User):
    profile = user.profile
    if not profile:
        profile = UserProfile(user_id=user.id)
        db.session.add(profile)
        db.session.commit()
    hydrate_profile_secure_fields(user, profile)
    return profile


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Please log in first.", "error")
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("admin_user") is None:
            flash("Admin login required.", "error")
            return redirect(url_for("main.admin_login", next=request.path))
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
    admin_user_id = session.get(ADMIN_SESSION_KEY)
    g.admin_user = db.session.get(AdminUser, admin_user_id) if admin_user_id else None
    now_utc = datetime.utcnow()

    if g.user and g.user.profile:
        hydrate_profile_secure_fields(g.user, g.user.profile)
    if g.user:
        if g.user.is_blocked:
            session.pop("user_id", None)
            g.user = None
            flash("Your account is blocked. Contact support for help.", "error")
        else:
            if g.user.last_active_at is None or (now_utc - g.user.last_active_at) > timedelta(minutes=5):
                g.user.last_active_at = now_utc
                db.session.add(g.user)
                db.session.commit()


@bp.app_context_processor
def inject_user():
    current_user = g.get("user")
    current_admin = g.get("admin_user")
    profile_complete = False
    if current_user and current_user.profile:
        profile_complete = len(current_user.profile.missing_required_fields()) == 0
    return {
        "current_user": current_user,
        "current_admin": current_admin,
        "profile_complete": profile_complete,
    }


@bp.get("/")
def index():
    public_content = get_site_settings(["home_intro", "contact_email"])
    if g.user is None:
        return render_template(
            "index.html",
            is_authenticated=False,
            public_content=public_content,
        )

    profile = get_or_create_profile(g.user)
    checkin_count = DailyCheckIn.query.filter_by(user_id=g.user.id).count()
    meal_count = Meal.query.filter_by(user_id=g.user.id).count()
    substance_count = Substance.query.filter_by(user_id=g.user.id).count()
    weekly = build_home_weekly_context(g.user, profile)
    return render_template(
        "index.html",
        is_authenticated=True,
        checkin_count=checkin_count,
        meal_count=meal_count,
        substance_count=substance_count,
        missing_required=profile.missing_required_fields(),
        weekly=weekly,
        public_content=public_content,
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
        if email and BlockedEmail.query.filter_by(email=email).first():
            flash("This email is blocked from registration. Contact support if this is a mistake.", "error")
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
        session.permanent = True
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
        if user.is_blocked:
            flash("This account is blocked. Contact support if you need help.", "error")
            return render_template("login.html", next=next_url)

        session.clear()
        session["user_id"] = user.id
        session.permanent = True
        user.last_active_at = datetime.utcnow()
        db.session.add(user)
        db.session.commit()
        flash("Logged in.", "success")

        if next_url and next_url.startswith("/"):
            return redirect(next_url)
        return redirect(url_for("main.index"))

    return render_template("login.html", next=request.args.get("next"))


@bp.route("/lost-password", methods=["GET", "POST"])
def lost_password():
    if request.method == "POST":
        email = normalize_email(request.form.get("email"))
        contact_email = get_site_setting("contact_email") or "support@coachmim.com"
        if email:
            flash(
                "Password reset is currently handled by support. "
                f"Please email {contact_email} from your account email.",
                "success",
            )
        else:
            flash("Enter the email used on your account.", "error")
    return render_template("lost_password.html")


@bp.post("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("main.login"))


@bp.route("/dontcrash", methods=["GET", "POST"])
def admin_login():
    ensure_default_admin_user()
    next_url = request.args.get("next") or request.form.get("next")

    if request.method == "POST":
        username = normalize_text(request.form.get("username"))
        password = request.form.get("password") or ""
        admin = AdminUser.query.filter_by(username=username).first() if username else None
        if not admin or not check_password_hash(admin.password_hash, password):
            flash("Invalid admin username or password.", "error")
            return render_template("admin_login.html", next=next_url)

        session[ADMIN_SESSION_KEY] = admin.id
        admin.last_login_at = datetime.utcnow()
        db.session.add(admin)
        db.session.commit()
        flash("Admin login successful.", "success")

        if next_url and next_url.startswith("/"):
            return redirect(next_url)
        return redirect(url_for("main.admin_dashboard"))

    if g.get("admin_user"):
        return redirect(url_for("main.admin_dashboard"))
    return render_template("admin_login.html", next=next_url)


@bp.post("/dontcrash/logout")
@admin_login_required
def admin_logout():
    session.pop(ADMIN_SESSION_KEY, None)
    flash("Admin logged out.", "success")
    return redirect(url_for("main.admin_login"))


@bp.get("/dontcrash/dashboard")
@admin_login_required
def admin_dashboard():
    active_since = datetime.utcnow() - timedelta(minutes=15)

    total_users = User.query.filter(~User.email.in_(SYSTEM_USER_EMAILS)).count()
    active_users = (
        User.query.filter(
            ~User.email.in_(SYSTEM_USER_EMAILS),
            User.is_blocked.is_(False),
            User.last_active_at.isnot(None),
            User.last_active_at >= active_since,
        ).count()
    )
    blocked_users = User.query.filter(~User.email.in_(SYSTEM_USER_EMAILS), User.is_blocked.is_(True)).count()
    spam_users = User.query.filter(~User.email.in_(SYSTEM_USER_EMAILS), User.is_spam.is_(True)).count()

    community_post_count = CommunityPost.query.count()
    flagged_posts_count = CommunityPost.query.filter_by(is_flagged=True).count()
    flagged_comments_count = CommunityComment.query.filter_by(is_flagged=True).count()

    signups_last_7d = User.query.filter(
        ~User.email.in_(SYSTEM_USER_EMAILS),
        User.created_at >= datetime.utcnow() - timedelta(days=7),
    ).count()

    return render_template(
        "admin_dashboard.html",
        total_users=total_users,
        active_users=active_users,
        blocked_users=blocked_users,
        spam_users=spam_users,
        signups_last_7d=signups_last_7d,
        community_post_count=community_post_count,
        flagged_posts_count=flagged_posts_count,
        flagged_comments_count=flagged_comments_count,
    )


@bp.get("/dontcrash/users")
@admin_login_required
def admin_users():
    users = (
        User.query.filter(~User.email.in_(SYSTEM_USER_EMAILS))
        .order_by(User.created_at.desc(), User.id.desc())
        .all()
    )

    checkin_counts = dict(
        db.session.query(DailyCheckIn.user_id, func.count(DailyCheckIn.id))
        .group_by(DailyCheckIn.user_id)
        .all()
    )
    post_counts = dict(
        db.session.query(CommunityPost.user_id, func.count(CommunityPost.id))
        .group_by(CommunityPost.user_id)
        .all()
    )

    rows = []
    for user in users:
        rows.append(
            {
                "user": user,
                "checkin_days": int(checkin_counts.get(user.id, 0)),
                "community_posts": int(post_counts.get(user.id, 0)),
            }
        )

    return render_template("admin_users.html", rows=rows)


@bp.post("/dontcrash/users/<int:user_id>/action")
@admin_login_required
def admin_user_action(user_id: int):
    action = (request.form.get("action") or "").strip().lower()
    user = db.session.get(User, user_id)
    if not user or normalize_email(user.email) in SYSTEM_USER_EMAILS:
        flash("User account was not found.", "error")
        return redirect(url_for("main.admin_users"))

    if action == "delete":
        email_snapshot = user.email
        hard_delete_user_account(user_id)
        remove_blocked_email(email_snapshot)
        db.session.commit()
        flash("Account deleted. This email can sign up again.", "success")
        return redirect(url_for("main.admin_users"))

    if action == "block":
        user.is_blocked = True
        user.is_spam = False
        add_blocked_email(user.email, "blocked", g.admin_user.id)
        db.session.add(user)
        db.session.commit()
        flash("Account blocked. Email cannot sign up again.", "success")
        return redirect(url_for("main.admin_users"))

    if action == "spam_block":
        user.is_blocked = True
        user.is_spam = True
        add_blocked_email(user.email, "spam", g.admin_user.id)
        CommunityPost.query.filter_by(user_id=user.id).update(
            {
                "is_hidden": True,
                "is_flagged": True,
                "flag_reason": "Auto-hidden after spam block.",
            },
            synchronize_session=False,
        )
        CommunityComment.query.filter_by(user_id=user.id).update(
            {
                "is_hidden": True,
                "is_flagged": True,
                "flag_reason": "Auto-hidden after spam block.",
            },
            synchronize_session=False,
        )
        db.session.add(user)
        db.session.commit()
        flash("User marked spam + blocked. Community content hidden.", "success")
        return redirect(url_for("main.admin_users"))

    if action == "unblock":
        user.is_blocked = False
        user.is_spam = False
        remove_blocked_email(user.email)
        db.session.add(user)
        db.session.commit()
        flash("User unblocked.", "success")
        return redirect(url_for("main.admin_users"))

    flash("Unknown user action.", "error")
    return redirect(url_for("main.admin_users"))


def _extract_title_and_body_from_mim_response(text: str):
    raw = str(text or "").strip()
    if not raw:
        return (None, None)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    title = None
    body_lines = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("title:") and not title:
            title = line.split(":", 1)[1].strip()
            continue
        if lowered.startswith("body:"):
            body_lines.append(line.split(":", 1)[1].strip())
            continue
        body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return (normalize_text(title), normalize_text(body))


@bp.get("/dontcrash/community")
@admin_login_required
def admin_community():
    author_id = parse_int(request.args.get("author_id"))
    include_hidden = parse_bool(request.args.get("include_hidden"))

    post_query = CommunityPost.query.options(
        selectinload(CommunityPost.user),
        selectinload(CommunityPost.comments).selectinload(CommunityComment.user),
        selectinload(CommunityPost.likes),
    )
    if author_id:
        post_query = post_query.filter(CommunityPost.user_id == author_id)
    if not include_hidden:
        post_query = post_query.filter(CommunityPost.is_hidden.is_(False))
    posts = post_query.order_by(CommunityPost.created_at.desc(), CommunityPost.id.desc()).limit(180).all()

    flagged_posts = (
        CommunityPost.query.options(selectinload(CommunityPost.user))
        .filter(CommunityPost.is_flagged.is_(True))
        .order_by(CommunityPost.created_at.desc())
        .limit(120)
        .all()
    )
    flagged_comments = (
        CommunityComment.query.options(
            selectinload(CommunityComment.user),
            selectinload(CommunityComment.post),
        )
        .filter(CommunityComment.is_flagged.is_(True))
        .order_by(CommunityComment.created_at.desc())
        .limit(120)
        .all()
    )

    return render_template(
        "admin_community.html",
        posts=posts,
        flagged_posts=flagged_posts,
        flagged_comments=flagged_comments,
        include_hidden=include_hidden,
        author_id=author_id,
        category_options=COMMUNITY_CATEGORY_OPTIONS,
    )


@bp.post("/dontcrash/community/post")
@admin_login_required
def admin_community_post():
    category = normalize_community_category(request.form.get("category"))
    title = normalize_text(request.form.get("title"))
    content = normalize_text(request.form.get("content"))
    generate_prompt = normalize_text(request.form.get("generate_prompt"))
    post_as = (request.form.get("post_as") or "mim").strip().lower()

    if generate_prompt:
        try:
            mim_result = ask_mim_general_chat(
                first_name="Admin",
                question=(
                    "Write a short community post for a health app.\n"
                    f"Category: {category}\n"
                    f"Topic: {generate_prompt}\n"
                    "Format:\nTitle: ...\nBody: ..."
                ),
                history=[],
            )
            guessed_title, guessed_body = _extract_title_and_body_from_mim_response(mim_result)
            if not title and guessed_title:
                title = guessed_title
            if not content and guessed_body:
                content = guessed_body
        except Exception:
            current_app.logger.exception("admin community MIM generation failed")

    if not title or len(title) < 4:
        flash("Title is required (min 4 chars).", "error")
        return redirect(url_for("main.admin_community"))
    if not content or len(content) < 12:
        flash("Content is required (min 12 chars).", "error")
        return redirect(url_for("main.admin_community"))

    blocked, reason = community_content_is_blocked(f"{title}\n{content}")
    if blocked:
        flash(reason or "This content was blocked by moderation.", "error")
        return redirect(url_for("main.admin_community"))

    if post_as == "admin":
        author = get_or_create_system_user(
            email="admin-team@coachmim.local",
            full_name="CoachMIM Admin",
        )
    else:
        author = get_or_create_system_user(
            email="mim-bot@coachmim.local",
            full_name="MIM",
        )
    if author is None:
        flash("Could not resolve a system author account for this post.", "error")
        return redirect(url_for("main.admin_community"))

    post = CommunityPost(
        user_id=author.id,
        category=category,
        title=title[:180],
        content=content,
        is_hidden=False,
        is_flagged=False,
    )
    db.session.add(post)
    db.session.commit()
    flash("Community content posted.", "success")
    return redirect(url_for("main.admin_community") + f"#admin-post-{post.id}")


@bp.post("/dontcrash/community/post/<int:post_id>/moderate")
@admin_login_required
def admin_community_moderate_post(post_id: int):
    action = (request.form.get("action") or "").strip().lower()
    post = db.session.get(CommunityPost, post_id)
    if not post:
        flash("Post not found.", "error")
        return redirect(url_for("main.admin_community"))

    if action == "approve":
        post.is_flagged = False
        post.flag_reason = None
        db.session.add(post)
        db.session.commit()
        flash("Post approved.", "success")
        return redirect(url_for("main.admin_community"))

    if action == "decline":
        post.is_hidden = True
        post.is_flagged = False
        post.flag_reason = "Declined by admin."
        db.session.add(post)
        db.session.commit()
        flash("Post declined and hidden.", "success")
        return redirect(url_for("main.admin_community"))

    if action == "restore":
        post.is_hidden = False
        post.is_flagged = False
        post.flag_reason = None
        db.session.add(post)
        db.session.commit()
        flash("Post restored.", "success")
        return redirect(url_for("main.admin_community"))

    if action == "block_user":
        user = db.session.get(User, post.user_id)
        if user:
            user.is_blocked = True
            user.is_spam = True
            add_blocked_email(user.email, "spam", g.admin_user.id)
            db.session.add(user)
            CommunityPost.query.filter_by(user_id=user.id).update(
                {
                    "is_hidden": True,
                    "is_flagged": True,
                    "flag_reason": "User blocked by admin moderation.",
                },
                synchronize_session=False,
            )
            CommunityComment.query.filter_by(user_id=user.id).update(
                {
                    "is_hidden": True,
                    "is_flagged": True,
                    "flag_reason": "User blocked by admin moderation.",
                },
                synchronize_session=False,
            )
        post.is_hidden = True
        post.is_flagged = True
        post.flag_reason = "User blocked by admin moderation."
        db.session.add(post)
        db.session.commit()
        flash("Post author blocked and content hidden.", "success")
        return redirect(url_for("main.admin_community"))

    flash("Unknown moderation action.", "error")
    return redirect(url_for("main.admin_community"))


@bp.post("/dontcrash/community/comment/<int:comment_id>/moderate")
@admin_login_required
def admin_community_moderate_comment(comment_id: int):
    action = (request.form.get("action") or "").strip().lower()
    comment = db.session.get(CommunityComment, comment_id)
    if not comment:
        flash("Comment not found.", "error")
        return redirect(url_for("main.admin_community"))

    if action == "approve":
        comment.is_flagged = False
        comment.flag_reason = None
        db.session.add(comment)
        db.session.commit()
        flash("Comment approved.", "success")
        return redirect(url_for("main.admin_community"))

    if action == "decline":
        comment.is_hidden = True
        comment.is_flagged = False
        comment.flag_reason = "Declined by admin."
        db.session.add(comment)
        db.session.commit()
        flash("Comment declined and hidden.", "success")
        return redirect(url_for("main.admin_community"))

    if action == "block_user":
        user = db.session.get(User, comment.user_id)
        if user:
            user.is_blocked = True
            user.is_spam = True
            add_blocked_email(user.email, "spam", g.admin_user.id)
            db.session.add(user)
            CommunityPost.query.filter_by(user_id=user.id).update(
                {
                    "is_hidden": True,
                    "is_flagged": True,
                    "flag_reason": "User blocked by admin moderation.",
                },
                synchronize_session=False,
            )
            CommunityComment.query.filter_by(user_id=user.id).update(
                {
                    "is_hidden": True,
                    "is_flagged": True,
                    "flag_reason": "User blocked by admin moderation.",
                },
                synchronize_session=False,
            )
        comment.is_hidden = True
        comment.is_flagged = True
        comment.flag_reason = "User blocked by admin moderation."
        db.session.add(comment)
        db.session.commit()
        flash("Comment author blocked and content hidden.", "success")
        return redirect(url_for("main.admin_community"))

    flash("Unknown moderation action.", "error")
    return redirect(url_for("main.admin_community"))


@bp.route("/dontcrash/frontpage", methods=["GET", "POST"])
@admin_login_required
def admin_frontpage():
    if request.method == "POST":
        section = (request.form.get("section") or "").strip().lower()
        if section == "settings":
            set_site_setting("home_intro", normalize_text(request.form.get("home_intro")))
            set_site_setting("contact_email", normalize_text(request.form.get("contact_email")))
            set_site_setting("privacy_summary", normalize_text(request.form.get("privacy_summary")))
            set_site_setting("terms_summary", normalize_text(request.form.get("terms_summary")))
            db.session.commit()
            flash("Front page settings updated.", "success")
            return redirect(url_for("main.admin_frontpage"))

        if section == "password":
            current_password = request.form.get("current_password") or ""
            new_password = request.form.get("new_password") or ""
            confirm_password = request.form.get("confirm_password") or ""
            if not check_password_hash(g.admin_user.password_hash, current_password):
                flash("Current admin password is incorrect.", "error")
                return redirect(url_for("main.admin_frontpage"))
            if len(new_password) < 8:
                flash("New password must be at least 8 characters.", "error")
                return redirect(url_for("main.admin_frontpage"))
            if new_password != confirm_password:
                flash("New password confirmation does not match.", "error")
                return redirect(url_for("main.admin_frontpage"))

            g.admin_user.password_hash = generate_password_hash(new_password)
            db.session.add(g.admin_user)
            db.session.commit()
            flash("Admin password updated.", "success")
            return redirect(url_for("main.admin_frontpage"))

    settings = get_site_settings(["home_intro", "contact_email", "privacy_summary", "terms_summary"])
    using_bootstrap_password = check_password_hash(g.admin_user.password_hash, ADMIN_BOOTSTRAP_PASSWORD)
    return render_template(
        "admin_frontpage.html",
        settings=settings,
        using_bootstrap_password=using_bootstrap_password,
    )


@bp.get("/privacy")
def privacy_page():
    summary = get_site_setting("privacy_summary")
    contact_email = get_site_setting("contact_email")
    return render_template("privacy.html", summary=summary, contact_email=contact_email)


@bp.get("/terms")
def terms_page():
    summary = get_site_setting("terms_summary")
    contact_email = get_site_setting("contact_email")
    return render_template("terms.html", summary=summary, contact_email=contact_email)


@bp.get("/about")
def about_page():
    return render_template("about.html")


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
        blocked_entry = BlockedEmail.query.filter_by(email=email).first()
        if blocked_entry and email != normalize_email(g.user.email):
            flash("This email is blocked and cannot be used on CoachMIM.", "error")
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

        persist_profile_secure_fields(g.user, profile)
        db.session.add(g.user)
        db.session.add(profile)
        db.session.commit()
        hydrate_profile_secure_fields(g.user, profile)

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
    manager_view = (request.args.get("view") or "checkin").strip().lower()
    if manager_view not in DAY_MANAGER_VIEWS:
        manager_view = "checkin"
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
    hydrate_checkin_secure_fields(g.user, record)
    hydrate_checkin_secure_fields(g.user, today_record)
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
    for history_record in history_records:
        hydrate_checkin_secure_fields(g.user, history_record)
    history_rows = [
        {
            "record": row,
            "segments": checkin_segment_status(row),
            "is_today": row.day == local_today,
            "has_data": checkin_has_any_data(row),
        }
        for row in history_records
    ]

    start, end = day_bounds(selected_day)
    day_meals = (
        Meal.query.filter(
            Meal.user_id == g.user.id,
            Meal.eaten_at >= start,
            Meal.eaten_at < end,
        )
        .order_by(Meal.eaten_at.desc())
        .all()
    )
    for meal in day_meals:
        hydrate_meal_secure_fields(g.user, meal)
    day_substances = (
        Substance.query.filter(
            Substance.user_id == g.user.id,
            Substance.taken_at >= start,
            Substance.taken_at < end,
        )
        .order_by(Substance.taken_at.desc())
        .all()
    )
    for entry in day_substances:
        hydrate_substance_secure_fields(g.user, entry)

    day_food_entries = [meal for meal in day_meals if not meal.is_beverage]
    day_drink_entries = [meal for meal in day_meals if meal.is_beverage]
    day_substance_entries = [entry for entry in day_substances if entry.kind in {"alcohol", "caffeine", "nicotine", "other"}]
    day_activity_entries = [entry for entry in day_substances if entry.kind == "activity"]
    day_medication_entries = [entry for entry in day_substances if entry.kind in {"medication", "supplement"}]

    selected_day_meal_count = len(day_food_entries)
    selected_day_drink_count = len(day_drink_entries)
    selected_day_substance_count = len(day_substance_entries)
    selected_day_activity_count = len(day_activity_entries)
    selected_day_medication_count = len(day_medication_entries)
    local_now = datetime.now(get_user_zoneinfo(g.user))
    default_entry_time = local_now.strftime("%H:%M")
    default_entry_datetime = f"{selected_day.isoformat()}T{default_entry_time}"
    quick_favorites = _build_day_manager_favorites_for_user(g.user.id)
    selected_segments = checkin_segment_status(record)
    checkin_default_tab = resolve_checkin_default_tab(
        segments=selected_segments,
        is_viewing_today=(selected_day == local_today),
        local_hour=local_now.hour,
    )

    prev_day = selected_day - timedelta(days=1)
    next_day = selected_day + timedelta(days=1)
    can_go_next = next_day <= local_today

    return render_template(
        "checkin.html",
        record=record,
        selected_day=selected_day.isoformat(),
        selected_day_weekday=selected_day.strftime("%A"),
        selected_day_pretty=selected_day.strftime("%B %d, %Y"),
        manager_view=manager_view,
        local_today=local_today.isoformat(),
        checked_in_today=checkin_has_any_data(today_record),
        selected_day_checked_in=checkin_has_any_data(record),
        selected_segments=selected_segments,
        checkin_default_tab=checkin_default_tab,
        history_rows=history_rows,
        prev_day=prev_day.isoformat(),
        next_day=next_day.isoformat(),
        can_go_next=can_go_next,
        is_viewing_today=(selected_day == local_today),
        checkin_weight_unit=weight_unit,
        morning_weight_display=morning_weight_display,
        checkin_unit_system=unit_system,
        selected_day_meal_count=selected_day_meal_count,
        selected_day_drink_count=selected_day_drink_count,
        selected_day_substance_count=selected_day_substance_count,
        selected_day_activity_count=selected_day_activity_count,
        selected_day_medication_count=selected_day_medication_count,
        day_food_entries=day_food_entries,
        day_drink_entries=day_drink_entries,
        day_substance_entries=day_substance_entries,
        day_activity_entries=day_activity_entries,
        day_medication_entries=day_medication_entries,
        default_entry_datetime=default_entry_datetime,
        meal_quick_favorites=quick_favorites["meal"],
        drink_quick_favorites=quick_favorites["drink"],
        substance_quick_favorites=quick_favorites["substance"],
        activity_quick_favorites=quick_favorites["activity"],
        medication_quick_favorites=quick_favorites["medications"],
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
    else:
        hydrate_checkin_secure_fields(g.user, record)

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

    persist_checkin_secure_fields(g.user, record)
    db.session.add(record)
    db.session.commit()
    flash(f"Check-in saved for {selected_day.isoformat()}.", "success")
    return redirect(url_for("main.checkin_form", day=selected_day.isoformat()))


@bp.post("/checkin/meal-quick")
@login_required
@profile_required
def checkin_meal_quick_save():
    local_today = get_user_local_today(g.user)
    selected_day_raw = request.form.get("day", local_today.isoformat())
    manager_view = (request.form.get("view") or "meal").strip().lower()
    if manager_view not in DAY_MANAGER_VIEWS:
        manager_view = "meal"

    try:
        selected_day = date.fromisoformat(selected_day_raw)
    except ValueError:
        selected_day = local_today

    if selected_day > local_today:
        selected_day = local_today

    favorite_id = parse_int(request.form.get("favorite_id"))
    selected_favorite = _find_user_quick_favorite(g.user.id, favorite_id)
    favorite_scope = _favorite_scope_from_tags(selected_favorite.tags) if selected_favorite else None
    if selected_favorite:
        if manager_view == "drink":
            is_usable_favorite = selected_favorite.is_beverage or favorite_scope == "drink"
        else:
            is_usable_favorite = (not selected_favorite.is_beverage) and favorite_scope in {None, "meal"}
        if not is_usable_favorite:
            selected_favorite = None

    eaten_at_raw = request.form.get("eaten_at")
    if not eaten_at_raw:
        fallback_time = datetime.now(get_user_zoneinfo(g.user)).strftime("%H:%M")
        eaten_at_raw = f"{selected_day.isoformat()}T{fallback_time}"

    try:
        eaten_at_dt = datetime.fromisoformat(eaten_at_raw)
    except ValueError:
        flash("Invalid date/time. Use the picker and try again.", "error")
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))

    description = normalize_text(request.form.get("description"))
    if not description and selected_favorite:
        description = selected_favorite.description or selected_favorite.name
    if not description:
        flash("Name/description is required for quick meal logging.", "error")
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))

    is_beverage = parse_bool(request.form.get("is_beverage")) or manager_view == "drink"
    label_value = normalize_text(request.form.get("label"))
    if not label_value and selected_favorite:
        label_value = selected_favorite.label
    if not label_value and is_beverage:
        label_value = "Drink"

    portion_notes = normalize_text(request.form.get("portion_notes"))
    if not portion_notes and selected_favorite:
        portion_notes = selected_favorite.portion_notes

    tags_value = parse_tags(request.form.get("tags"))
    if not tags_value and selected_favorite:
        tags_value = _apply_favorite_scope(selected_favorite.tags, None)

    calories_value = parse_int(request.form.get("calories"))
    protein_value = parse_float(request.form.get("protein_g"))
    carbs_value = parse_float(request.form.get("carbs_g"))
    fat_value = parse_float(request.form.get("fat_g"))
    sugar_value = parse_float(request.form.get("sugar_g"))
    sodium_value = parse_float(request.form.get("sodium_mg"))
    caffeine_value = parse_float(request.form.get("caffeine_mg"))
    if selected_favorite:
        if calories_value is None:
            calories_value = selected_favorite.calories
        if protein_value is None:
            protein_value = selected_favorite.protein_g
        if carbs_value is None:
            carbs_value = selected_favorite.carbs_g
        if fat_value is None:
            fat_value = selected_favorite.fat_g
        if sugar_value is None:
            sugar_value = selected_favorite.sugar_g
        if sodium_value is None:
            sodium_value = selected_favorite.sodium_mg
        if caffeine_value is None:
            caffeine_value = selected_favorite.caffeine_mg

    meal = Meal(
        user_id=g.user.id,
        eaten_at=eaten_at_dt,
        label=label_value,
        description=description,
        portion_notes=portion_notes,
        tags=tags_value,
        calories=calories_value,
        protein_g=protein_value,
        carbs_g=carbs_value,
        fat_g=fat_value,
        sugar_g=sugar_value,
        sodium_mg=sodium_value,
        caffeine_mg=caffeine_value,
        is_beverage=is_beverage,
    )

    _save_day_manager_meal_favorite(meal, view=manager_view)
    persist_meal_secure_fields(g.user, meal)
    db.session.add(meal)
    db.session.commit()
    flash("Entry logged.", "success")
    return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))


@bp.post("/checkin/substance-quick")
@login_required
@profile_required
def checkin_substance_quick_save():
    local_today = get_user_local_today(g.user)
    selected_day_raw = request.form.get("day", local_today.isoformat())
    manager_view = (request.form.get("view") or "substance").strip().lower()
    if manager_view not in DAY_MANAGER_VIEWS:
        manager_view = "substance"

    try:
        selected_day = date.fromisoformat(selected_day_raw)
    except ValueError:
        selected_day = local_today

    if selected_day > local_today:
        selected_day = local_today

    favorite_id = parse_int(request.form.get("favorite_id"))
    selected_favorite = _find_user_quick_favorite(g.user.id, favorite_id)
    expected_scope = "substance"
    if manager_view == "activity":
        expected_scope = "activity"
    elif manager_view == "medications":
        expected_scope = "medications"

    if selected_favorite and _favorite_scope_from_tags(selected_favorite.tags) != expected_scope:
        selected_favorite = None

    favorite_payload = selected_favorite.ingredients if selected_favorite and isinstance(selected_favorite.ingredients, dict) else {}

    taken_at_raw = request.form.get("taken_at")
    if not taken_at_raw:
        fallback_time = datetime.now(get_user_zoneinfo(g.user)).strftime("%H:%M")
        taken_at_raw = f"{selected_day.isoformat()}T{fallback_time}"

    try:
        taken_at_dt = datetime.fromisoformat(taken_at_raw)
    except ValueError:
        flash("Invalid date/time. Use the picker and try again.", "error")
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))

    kind = (request.form.get("kind") or "").strip().lower()
    if manager_view == "activity":
        kind = "activity"
    elif manager_view == "medications" and not kind:
        kind = str(favorite_payload.get("kind") or "medication").strip().lower()
    elif not kind and manager_view == "substance":
        kind = str(favorite_payload.get("kind") or "").strip().lower()

    allowed_kinds = {"alcohol", "caffeine", "nicotine", "other", "activity", "medication", "supplement"}
    if kind not in allowed_kinds:
        flash("Select a valid type for this entry.", "error")
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))

    amount = normalize_text(request.form.get("amount"))
    notes = normalize_text(request.form.get("notes")) or normalize_text(favorite_payload.get("notes"))

    activity_type = None
    duration_min = None
    intensity = None
    med_name = None
    dose = None

    if kind == "activity":
        activity_type = normalize_text(request.form.get("activity_type")) or normalize_text(favorite_payload.get("activity_type"))
        duration_min = parse_int(request.form.get("duration_min"))
        if duration_min is None and favorite_payload.get("duration_min") not in (None, ""):
            duration_min = parse_int(favorite_payload.get("duration_min"))
        intensity = parse_int(request.form.get("intensity"))
        if intensity is None and favorite_payload.get("intensity") not in (None, ""):
            intensity = parse_int(favorite_payload.get("intensity"))
        built_amount_parts = []
        if activity_type:
            built_amount_parts.append(activity_type)
        if duration_min is not None:
            built_amount_parts.append(f"{duration_min} min")
        if intensity is not None:
            built_amount_parts.append(f"intensity {intensity}/10")
        if built_amount_parts:
            amount = " | ".join(built_amount_parts)
    elif kind in {"medication", "supplement"}:
        med_name = normalize_text(request.form.get("med_name")) or normalize_text(favorite_payload.get("med_name"))
        dose = normalize_text(request.form.get("dose")) or normalize_text(favorite_payload.get("dose"))
        built_amount_parts = []
        if med_name:
            built_amount_parts.append(med_name)
        if dose:
            built_amount_parts.append(dose)
        if built_amount_parts:
            amount = " - ".join(built_amount_parts)
    else:
        if not amount:
            amount = normalize_text(favorite_payload.get("amount")) or normalize_text(selected_favorite.portion_notes if selected_favorite else None)

    if not amount:
        flash("Amount/details are required for this entry.", "error")
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))

    if kind == "activity":
        _save_day_manager_nonmeal_favorite(
            "activity",
            label="Activity",
            description=activity_type,
            portion_notes=f"{duration_min} min" if duration_min is not None else None,
            payload={
                "activity_type": activity_type,
                "duration_min": duration_min,
                "intensity": intensity,
                "notes": notes,
            },
        )
    elif kind in {"medication", "supplement"}:
        _save_day_manager_nonmeal_favorite(
            "medications",
            label="Medication/Supplement",
            description=med_name,
            portion_notes=dose,
            payload={
                "kind": kind,
                "med_name": med_name,
                "dose": dose,
                "notes": notes,
            },
        )
    else:
        _save_day_manager_nonmeal_favorite(
            "substance",
            label="Substance",
            description=kind,
            portion_notes=amount,
            payload={
                "kind": kind,
                "amount": amount,
                "notes": notes,
            },
        )

    entry = Substance(
        user_id=g.user.id,
        taken_at=taken_at_dt,
        kind=kind,
        amount=amount,
        notes=notes,
    )
    persist_substance_secure_fields(g.user, entry)
    db.session.add(entry)
    db.session.commit()
    flash("Entry logged.", "success")
    return redirect(url_for("main.checkin_form", day=selected_day.isoformat(), view=manager_view))


@bp.get("/meal")
@login_required
@profile_required
def meal_form():
    day_str = request.args.get("day")
    if day_str:
        try:
            selected_day = date.fromisoformat(day_str)
        except ValueError:
            selected_day = date.today()
            flash("Invalid day format. Showing today.", "error")
    else:
        selected_day = date.today()
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
    has_new_photo = bool(photo and photo.filename)
    if not meal_has_meaningful_content(meal, has_new_photo=has_new_photo):
        flash("Meal entry is empty. Select a food or add nutrition/details before saving.", "error")
        return redirect(url_for("main.meal_form", day=eaten_at_dt.date().isoformat()))

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

    upsert_favorite_from_request(meal=meal)
    shared_food = upsert_shared_food_from_request(meal)
    if meal.food_item_id is None and shared_food is not None:
        meal.food_item_id = shared_food.id
        if not meal.description:
            meal.description = shared_food.display_name()
    persist_meal_secure_fields(g.user, meal)
    db.session.add(meal)
    db.session.commit()
    flash("Meal logged.", "success")
    return redirect(url_for("main.meal_form", day=eaten_at_dt.date().isoformat()))


@bp.route("/meal/<int:meal_id>/edit", methods=["GET", "POST"])
@login_required
@profile_required
def meal_edit(meal_id: int):
    meal = Meal.query.filter_by(id=meal_id, user_id=g.user.id).first_or_404()
    hydrate_meal_secure_fields(g.user, meal)

    if request.method == "POST":
        try:
            eaten_at_dt = apply_meal_fields_from_request(meal)
        except ValueError:
            flash("Invalid meal timestamp. Use the date/time picker and try again.", "error")
            return redirect(url_for("main.meal_edit", meal_id=meal.id))

        photo = request.files.get("photo")
        has_new_photo = bool(photo and photo.filename)
        if not meal_has_meaningful_content(meal, has_new_photo=has_new_photo):
            flash("Meal entry is empty. Add at least one detail or delete the entry.", "error")
            return redirect(url_for("main.meal_edit", meal_id=meal.id))

        upsert_favorite_from_request(meal=meal)
        shared_food = upsert_shared_food_from_request(meal)
        if meal.food_item_id is None and shared_food is not None:
            meal.food_item_id = shared_food.id
            if not meal.description:
                meal.description = shared_food.display_name()
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

        persist_meal_secure_fields(g.user, meal)
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


def score_food_match(query: str, tokens: list[str], item: FoodItem) -> float:
    query_norm = normalize_search_text(query)
    name_norm = normalize_search_text(item.name)
    brand_norm = normalize_search_text(item.brand)
    merged = " ".join(part for part in [name_norm, brand_norm] if part)
    if not query_norm or not merged:
        return 0.0

    ratio_full = SequenceMatcher(None, query_norm, merged).ratio()
    ratio_name = SequenceMatcher(None, query_norm, name_norm).ratio() if name_norm else 0.0
    ratio = max(ratio_full, ratio_name)

    token_hits = 0.0
    for token in tokens:
        if token in merged:
            token_hits += 1.0
            continue
        if len(token) >= 3 and token[:3] in merged:
            token_hits += 0.75

    token_score = token_hits / max(len(tokens), 1)
    score = (ratio * 0.72) + (token_score * 0.28)

    if name_norm.startswith(query_norm):
        score += 0.18
    elif len(query_norm) >= 3 and query_norm[:3] in name_norm:
        score += 0.06

    if item.source == "seed":
        score += 0.03
    elif item.source == "community":
        score += 0.02

    return score


def fuzzy_food_matches(query: str, existing_ids: set[int], limit: int = 8):
    query_norm = normalize_search_text(query)
    tokens = tokenize_search_text(query_norm)
    if len(query_norm) < 3:
        return []

    conditions = []
    for token in tokens[:5]:
        conditions.append(FoodItem.name.ilike(f"%{token}%"))
        conditions.append(FoodItem.brand.ilike(f"%{token}%"))
        if len(token) >= 3:
            snippet = token[:3]
            conditions.append(FoodItem.name.ilike(f"%{snippet}%"))
            conditions.append(FoodItem.brand.ilike(f"%{snippet}%"))

    if query_norm:
        first_char = query_norm[0]
        conditions.append(FoodItem.name.ilike(f"{first_char}%"))
        conditions.append(FoodItem.brand.ilike(f"{first_char}%"))

    if not conditions:
        return []

    candidates = (
        FoodItem.query.filter(or_(*conditions))
        .order_by(
            case(
                (FoodItem.source == "seed", 0),
                (FoodItem.source == "community", 1),
                (FoodItem.source == "usda", 2),
                else_=3,
            ),
            case((FoodItem.calories.isnot(None), 0), else_=1),
            FoodItem.name.asc(),
        )
        .limit(350)
        .all()
    )

    scored = []
    for item in candidates:
        if item.id in existing_ids:
            continue
        score = score_food_match(query_norm, tokens, item)
        if score >= 0.42:
            scored.append((score, item))

    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in scored[:limit]]


def normalize_builder_unit(value: str | None) -> str:
    raw = (value or "").strip().lower().replace(".", "")
    mapping = {
        "serving": "serving",
        "servings": "serving",
        "portion": "serving",
        "portions": "serving",
        "g": "g",
        "gram": "g",
        "grams": "g",
        "oz": "oz",
        "ounce": "oz",
        "ounces": "oz",
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "ml": "ml",
        "milliliter": "ml",
        "milliliters": "ml",
        "cup": "cup",
        "cups": "cup",
        "tbsp": "tbsp",
        "tablespoon": "tbsp",
        "tablespoons": "tbsp",
        "tsp": "tsp",
        "teaspoon": "tsp",
        "teaspoons": "tsp",
        "item": "item",
        "items": "item",
        "piece": "item",
        "pieces": "item",
        "slice": "item",
        "slices": "item",
    }
    normalized = mapping.get(raw, raw)
    return normalized if normalized in {"serving", "g", "oz", "lb", "ml", "cup", "tbsp", "tsp", "item"} else "serving"


def _food_item_to_payload(item: FoodItem) -> dict:
    return {
        "id": item.id,
        "name": ((item.name or "").strip() or None),
        "brand": ((item.brand or "").strip() or None),
        "display_name": (
            f"{(item.name or '').strip()} ({(item.brand or '').strip()})"
            if (item.name or "").strip() and (item.brand or "").strip()
            else ((item.name or "").strip() or (item.brand or "").strip() or f"Food item #{item.id}")
        ),
        "serving_size": item.serving_size,
        "serving_unit": item.serving_unit,
        "calories": item.calories,
        "protein_g": item.protein_g,
        "carbs_g": item.carbs_g,
        "fat_g": item.fat_g,
        "sugar_g": item.sugar_g,
        "sodium_mg": item.sodium_mg,
        "caffeine_mg": item.caffeine_mg,
        "source": item.source,
    }


def find_best_food_item_match(query: str) -> tuple[FoodItem | None, float]:
    q = (query or "").strip()
    if len(q) < 2:
        return (None, 0.0)

    tokens = tokenize_search_text(q)
    if not tokens:
        return (None, 0.0)

    conditions = [FoodItem.name.ilike(f"%{q}%"), FoodItem.brand.ilike(f"%{q}%")]
    for token in tokens[:5]:
        conditions.append(FoodItem.name.ilike(f"%{token}%"))
        conditions.append(FoodItem.brand.ilike(f"%{token}%"))
        if len(token) >= 3:
            snippet = token[:3]
            conditions.append(FoodItem.name.ilike(f"%{snippet}%"))
            conditions.append(FoodItem.brand.ilike(f"%{snippet}%"))

    candidates = (
        FoodItem.query.filter(or_(*conditions))
        .order_by(
            case(
                (FoodItem.source == "seed", 0),
                (FoodItem.source == "community", 1),
                (FoodItem.source == "usda", 2),
                else_=3,
            ),
            case((FoodItem.calories.isnot(None), 0), else_=1),
            FoodItem.name.asc(),
        )
        .limit(250)
        .all()
    )

    best_item: FoodItem | None = None
    best_score = 0.0
    for item in candidates:
        score = score_food_match(q, tokens, item)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item and best_score >= 0.42:
        return (best_item, best_score)

    fuzzy = fuzzy_food_matches(q, existing_ids=set(), limit=1)
    if fuzzy:
        fuzzy_item = fuzzy[0]
        fuzzy_score = score_food_match(q, tokens, fuzzy_item)
        if fuzzy_score > best_score and fuzzy_score >= 0.42:
            return (fuzzy_item, fuzzy_score)

    return (None, best_score)


def _safe_float(value, fallback: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _estimate_meal_factor(quantity: float, unit: str, item: FoodItem) -> float:
    qty = _safe_float(quantity, fallback=1.0)
    if qty <= 0:
        qty = 1.0

    unit_norm = normalize_builder_unit(unit)
    serving_size = _safe_float(item.serving_size, fallback=0.0)
    serving_unit = normalize_builder_unit(item.serving_unit)

    if unit_norm == "serving":
        return qty

    if serving_size > 0:
        if serving_unit == unit_norm:
            return qty / serving_size

        if unit_norm in MASS_UNIT_TO_GRAMS and serving_unit in MASS_UNIT_TO_GRAMS:
            qty_g = qty * MASS_UNIT_TO_GRAMS[unit_norm]
            serving_g = serving_size * MASS_UNIT_TO_GRAMS[serving_unit]
            if serving_g > 0:
                return qty_g / serving_g

        if unit_norm in VOLUME_UNIT_TO_ML and serving_unit in VOLUME_UNIT_TO_ML:
            qty_ml = qty * VOLUME_UNIT_TO_ML[unit_norm]
            serving_ml = serving_size * VOLUME_UNIT_TO_ML[serving_unit]
            if serving_ml > 0:
                return qty_ml / serving_ml

    if unit_norm == "item":
        if serving_size > 0 and serving_unit == "item":
            return qty / serving_size
        return qty

    return 1.0


def _round_or_none(value: float, digits: int = 1):
    if value is None:
        return None
    return round(float(value), digits)


def _compact_ingredient_summary(ingredients: list[dict]) -> str | None:
    if not ingredients:
        return None
    chunks = []
    for item in ingredients[:3]:
        qty = _safe_float(item.get("quantity"), fallback=1.0)
        unit = normalize_builder_unit(item.get("unit"))
        name = normalize_text(item.get("name"))
        if not name:
            continue
        qty_text = str(int(qty)) if abs(qty - int(qty)) < 0.01 else f"{qty:.2f}".rstrip("0").rstrip(".")
        if unit == "serving":
            chunks.append(f"{qty_text} serving {name}")
        else:
            chunks.append(f"{qty_text} {unit} {name}")
    if not chunks:
        return None
    extra_count = max(0, len(ingredients) - len(chunks))
    tail = f" +{extra_count} more" if extra_count else ""
    return ", ".join(chunks) + tail


@bp.get("/foods/search")
@login_required
@profile_required
def food_search():
    query = (request.args.get("q") or "").strip()
    include_remote = parse_bool(request.args.get("remote"))
    if len(query) < 2:
        return jsonify({"results": [], "message": "Type at least 2 characters."})

    seed_common_foods_if_needed()

    def run_search(limit: int = 15):
        return (
            FoodItem.query.filter(
                or_(
                    FoodItem.name.ilike(f"%{query}%"),
                    FoodItem.brand.ilike(f"%{query}%"),
                )
            )
            .order_by(
                case(
                    (FoodItem.source == "seed", 0),
                    (FoodItem.source == "community", 1),
                    (FoodItem.source == "usda", 2),
                    else_=3,
                ),
                case((FoodItem.name.ilike(f"{query}%"), 0), else_=1),
                case((FoodItem.calories.isnot(None), 0), else_=1),
                FoodItem.name.asc(),
            )
            .limit(limit)
            .all()
        )

    results = run_search()
    used_fuzzy = False

    imported = 0
    message = None
    if include_remote and len(results) < 8:
        imported = import_foods_from_usda(query, max_results=12)
        if imported > 0:
            results = run_search()
        elif len(results) == 0:
            message = "No USDA matches found for that term. Try a broader keyword."
    elif len(results) == 0:
        message = "No local matches. Click Search USDA for a larger catalog."

    if len(results) < 15:
        fuzzy = fuzzy_food_matches(query, existing_ids={item.id for item in results}, limit=15 - len(results))
        if fuzzy:
            results.extend(fuzzy)
            used_fuzzy = True
            if not message:
                message = "Showing close matches (spelling-friendly)."

    deduped_results = []
    seen_keys = set()
    for item in results:
        dedupe_key = ((item.name or "").strip().lower(), (item.brand or "").strip().lower(), item.source or "")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_results.append(item)
    results = deduped_results[:15]

    payload = [_food_item_to_payload(item) for item in results]
    return jsonify({"results": payload, "message": message, "imported": imported, "used_fuzzy": used_fuzzy})


@bp.post("/meal/parse-text")
@login_required
@profile_required
def meal_parse_text():
    if request.is_json:
        body = request.get_json(silent=True) or {}
        sentence = normalize_text(body.get("text"))
    else:
        sentence = normalize_text(request.form.get("text"))

    if not sentence or len(sentence) < 6:
        return jsonify({"ok": False, "error": "Enter one sentence with ingredients (for example: 2 tbsp honey, 1 cup milk)."}), 400

    try:
        parsed = parse_meal_sentence(sentence)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Meal parse failed. Try a clearer sentence or add ingredients manually."}), 500

    seed_common_foods_if_needed()

    parsed_ingredients = parsed.get("ingredients") if isinstance(parsed.get("ingredients"), list) else []
    resolved_ingredients = []
    matched_count = 0

    for item in parsed_ingredients[:24]:
        if not isinstance(item, dict):
            continue

        ingredient_name = normalize_text(item.get("name"))
        if not ingredient_name:
            continue

        quantity = parse_float(item.get("quantity"))
        if quantity is None or quantity <= 0:
            quantity = 1.0

        parsed_unit = normalize_builder_unit(item.get("unit"))
        matched_item, score = find_best_food_item_match(ingredient_name)

        calories = parse_float(item.get("calories"))
        protein_g = parse_float(item.get("protein_g"))
        carbs_g = parse_float(item.get("carbs_g"))
        fat_g = parse_float(item.get("fat_g"))
        sugar_g = parse_float(item.get("sugar_g"))
        sodium_mg = parse_float(item.get("sodium_mg"))
        caffeine_mg = parse_float(item.get("caffeine_mg"))

        payload_item = {
            "food_item_id": None,
            "food_name": ingredient_name,
            "quantity": round(float(quantity), 3),
            "unit": parsed_unit,
            "serving_size": None,
            "serving_unit": None,
            "calories": int(round(calories)) if calories is not None else None,
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "fat_g": fat_g,
            "sugar_g": sugar_g,
            "sodium_mg": sodium_mg,
            "caffeine_mg": caffeine_mg,
            "match_score": round(float(score), 3) if score else 0.0,
            "match_source": "none",
            "matched_display_name": None,
        }

        if matched_item:
            matched_count += 1
            payload_item.update(
                {
                    "food_item_id": matched_item.id,
                    "food_name": matched_item.display_name(),
                    "serving_size": matched_item.serving_size,
                    "serving_unit": matched_item.serving_unit,
                    "calories": matched_item.calories if matched_item.calories is not None else payload_item["calories"],
                    "protein_g": matched_item.protein_g if matched_item.protein_g is not None else payload_item["protein_g"],
                    "carbs_g": matched_item.carbs_g if matched_item.carbs_g is not None else payload_item["carbs_g"],
                    "fat_g": matched_item.fat_g if matched_item.fat_g is not None else payload_item["fat_g"],
                    "sugar_g": matched_item.sugar_g if matched_item.sugar_g is not None else payload_item["sugar_g"],
                    "sodium_mg": matched_item.sodium_mg if matched_item.sodium_mg is not None else payload_item["sodium_mg"],
                    "caffeine_mg": matched_item.caffeine_mg if matched_item.caffeine_mg is not None else payload_item["caffeine_mg"],
                    "match_source": matched_item.source or "local",
                    "matched_display_name": matched_item.display_name(),
                }
            )
            if parsed_unit == "serving":
                payload_item["unit"] = "serving"
        resolved_ingredients.append(payload_item)

    if not resolved_ingredients:
        return jsonify({"ok": False, "error": "No ingredients detected. Try commas between ingredients."}), 400

    meal_title = normalize_text(parsed.get("meal_title"))
    meal_label = normalize_text(parsed.get("meal_label"))
    is_beverage = parse_bool(parsed.get("is_beverage")) if parsed.get("is_beverage") is not None else False
    if not meal_label and is_beverage:
        meal_label = "Drink"

    return jsonify(
        {
            "ok": True,
            "parsed": {
                "source": parsed.get("source") or "parser",
                "meal_title": meal_title,
                "meal_label": meal_label,
                "is_beverage": is_beverage,
                "ingredients": resolved_ingredients,
                "match_count": matched_count,
                "total_count": len(resolved_ingredients),
            },
        }
    )


@bp.post("/ai/day-manager-assist")
@login_required
@profile_required
def ai_day_manager_assist():
    body = request.get_json(silent=True) if request.is_json else {}
    context = (body.get("context") if isinstance(body, dict) else None) or request.form.get("context")
    context = (context or "meal").strip().lower()
    if context not in {"meal", "drink", "substance", "activity", "medications"}:
        return jsonify({"ok": False, "error": "Invalid context. Use meal, drink, substance, activity, or medications."}), 400

    raw_text = (body.get("text") if isinstance(body, dict) else None) or request.form.get("text")
    text = normalize_text(raw_text)
    first_name = (normalize_text(g.user.full_name) or "there").split(" ", 1)[0]
    entry_word_map = {
        "meal": "meal",
        "drink": "drink",
        "substance": "substance",
        "activity": "activity",
        "medications": "medication or supplement",
    }
    entry_word = entry_word_map.get(context, "entry")

    if context in {"substance", "activity", "medications"}:
        result = parse_day_manager_context_assist(context=context, text=text or "", first_name=first_name)
        return jsonify(
            {
                "ok": True,
                "needs_more": bool(result.get("needs_more")),
                "reply": result.get("reply") or f"Hi {first_name}, I prefilled this entry. Review before saving.",
                "follow_up_prompt": result.get("follow_up_prompt"),
                "suggested_fields": result.get("suggested_fields") or {},
                "matches": [],
                "unmatched_ingredients": [],
            }
        )

    if not text or len(text) < 6:
        return jsonify(
            {
                "ok": True,
                "needs_more": True,
                "reply": f"Hi {first_name}, tell me more about what was in your {entry_word} and I'll get this form started.",
                "follow_up_prompt": "Include ingredients and rough amounts (for example: pasta with chicken, olive oil, and parmesan).",
                "suggested_fields": {},
                "matches": [],
                "unmatched_ingredients": [],
            }
        )

    try:
        parsed = parse_meal_sentence(text)
    except RuntimeError as exc:
        return jsonify(
            {
                "ok": True,
                "needs_more": True,
                "reply": f"Hi {first_name}, {str(exc)}",
                "follow_up_prompt": "Try one sentence with ingredient amounts (for example: 2 cups pasta, 4 oz chicken, 1 tbsp olive oil).",
                "suggested_fields": {"description": text},
                "matches": [],
                "unmatched_ingredients": [],
            }
        )
    except Exception:
        return jsonify(
            {
                "ok": False,
                "error": "MIM could not parse that description. Try a clearer sentence with ingredient amounts.",
            }
        ), 500

    ingredients = parsed.get("ingredients") if isinstance(parsed.get("ingredients"), list) else []
    if not ingredients:
        return jsonify(
            {
                "ok": True,
                "needs_more": True,
                "reply": f"Hi {first_name}, I need more detail to estimate nutrition. Add ingredients and rough amounts.",
                "follow_up_prompt": "Example: pasta with tomato sauce, chicken breast, and parmesan.",
                "suggested_fields": {"description": text},
                "matches": [],
                "unmatched_ingredients": [],
            }
        )

    totals = {
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "sugar_g": 0.0,
        "sodium_mg": 0.0,
        "caffeine_mg": 0.0,
    }
    matched_items = []
    unmatched_ingredients = []

    for ingredient in ingredients[:20]:
        ingredient_name = normalize_text(ingredient.get("name"))
        if not ingredient_name:
            continue

        quantity = _safe_float(ingredient.get("quantity"), fallback=1.0)
        unit = normalize_builder_unit(ingredient.get("unit"))
        matched_item, score = find_best_food_item_match(ingredient_name)
        if not matched_item:
            unmatched_ingredients.append(ingredient_name)
            continue

        factor = _estimate_meal_factor(quantity, unit, matched_item)
        totals["calories"] += _safe_float(matched_item.calories) * factor
        totals["protein_g"] += _safe_float(matched_item.protein_g) * factor
        totals["carbs_g"] += _safe_float(matched_item.carbs_g) * factor
        totals["fat_g"] += _safe_float(matched_item.fat_g) * factor
        totals["sugar_g"] += _safe_float(matched_item.sugar_g) * factor
        totals["sodium_mg"] += _safe_float(matched_item.sodium_mg) * factor
        totals["caffeine_mg"] += _safe_float(matched_item.caffeine_mg) * factor
        matched_items.append(
            {
                "ingredient": ingredient_name,
                "food_item_id": matched_item.id,
                "display_name": matched_item.display_name(),
                "score": round(float(score), 3),
            }
        )

    is_beverage = context == "drink" or parse_bool(parsed.get("is_beverage"))
    label_value = "Drink" if is_beverage else (normalize_text(parsed.get("meal_label")) or None)
    suggested_fields = {
        "description": text,
        "label": label_value,
        "is_beverage": is_beverage,
        "portion_notes": _compact_ingredient_summary(ingredients),
    }

    if matched_items:
        suggested_fields.update(
            {
                "calories": int(round(totals["calories"])) if totals["calories"] > 0 else None,
                "protein_g": _round_or_none(totals["protein_g"], 1),
                "carbs_g": _round_or_none(totals["carbs_g"], 1),
                "fat_g": _round_or_none(totals["fat_g"], 1),
                "sugar_g": _round_or_none(totals["sugar_g"], 1),
                "sodium_mg": _round_or_none(totals["sodium_mg"], 1),
                "caffeine_mg": _round_or_none(totals["caffeine_mg"], 1),
            }
        )

    matched_count = len(matched_items)
    total_count = len(ingredients)
    if matched_count == 0:
        reply = (
            f"Hi {first_name}, I captured your {entry_word} description but I could not confidently match ingredients "
            "to nutrition data yet. Use Food Search or add nutrition manually."
        )
    elif matched_count < total_count:
        reply = (
            f"Hi {first_name}, I matched {matched_count} of {total_count} ingredients and prefilled estimated nutrition. "
            "Please review unmatched items."
        )
    else:
        reply = (
            f"Hi {first_name}, I matched your ingredients and prefilled estimated calories/macros. "
            "Review and adjust before saving."
        )

    return jsonify(
        {
            "ok": True,
            "needs_more": False,
            "reply": reply,
            "suggested_fields": suggested_fields,
            "matches": matched_items,
            "unmatched_ingredients": unmatched_ingredients,
        }
    )


@bp.post("/nutrition/label/parse")
@login_required
@profile_required
def nutrition_label_parse():
    photo = request.files.get("label_photo")
    if photo is None:
        return jsonify({"ok": False, "error": "Attach a nutrition label image first."}), 400
    filename = photo.filename or ""
    mime_type = (photo.mimetype or "").lower()
    if not allowed_file(filename) and not mime_type.startswith("image/"):
        return jsonify({"ok": False, "error": "Unsupported file type. Use png/jpg/jpeg/webp/heic."}), 400

    raw = photo.read()
    if not raw:
        return jsonify({"ok": False, "error": "The uploaded image was empty."}), 400

    hint_name = (request.form.get("ingredient_name") or "").strip() or None
    try:
        parsed = parse_nutrition_label_image(raw, photo.mimetype, hint_name=hint_name)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Label parsing failed. Enter values manually or retry."}), 500

    return jsonify({"ok": True, "parsed": parsed})


@bp.post("/nutrition/product/parse")
@login_required
@profile_required
def nutrition_product_parse():
    product_url = (request.form.get("product_url") or "").strip()
    ingredient_name = (request.form.get("ingredient_name") or "").strip() or None

    try:
        parsed = parse_product_page_url(product_url, hint_name=ingredient_name)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Product link parse failed. Use manual entry or label photo."}), 500

    return jsonify({"ok": True, "parsed": parsed})


def _chat_history_for_user(user: User, limit: int = 80):
    rows = (
        MIMChatMessage.query.filter_by(user_id=user.id)
        .order_by(MIMChatMessage.created_at.desc(), MIMChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    for row in rows:
        hydrate_chat_secure_fields(user, row)
    return rows


@bp.get("/ask-mim")
@login_required
def ask_mim_page():
    messages = _chat_history_for_user(g.user, limit=120)
    return render_template("ask_mim.html", messages=messages)


@bp.post("/ask-mim/send")
@login_required
def ask_mim_send():
    message_text = normalize_text(request.form.get("message"))
    image = request.files.get("image")

    has_image = bool(image and image.filename)
    if not message_text and not has_image:
        return jsonify({"ok": False, "error": "Enter a question or attach an image."}), 400

    image_bytes = None
    image_path = None
    image_mime = None

    if has_image:
        filename = image.filename or ""
        mime_type = (image.mimetype or "").lower()
        if not allowed_file(filename) and not mime_type.startswith("image/"):
            return jsonify({"ok": False, "error": "Unsupported image format. Use png/jpg/jpeg/webp/heic."}), 400

        image_bytes = image.read()
        if not image_bytes:
            return jsonify({"ok": False, "error": "Uploaded image was empty."}), 400

        image_mime = image.mimetype or "image/jpeg"
        safe_name = secure_filename(filename or "mim-question.jpg")
        upload_name = f"{uuid4().hex}_{safe_name}"
        upload_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        local_path = os.path.join(upload_dir, upload_name)
        with open(local_path, "wb") as fh:
            fh.write(image_bytes)
        image_path = f"uploads/{upload_name}"

    history_rows = _chat_history_for_user(g.user, limit=32)
    history_payload = []
    for row in history_rows:
        if row.role not in {"user", "assistant"}:
            continue
        if row.content:
            history_payload.append({"role": row.role, "content": row.content})

    first_name = (normalize_text(g.user.full_name) or "there").split(" ", 1)[0]
    user_prompt = message_text or "What is this and what should I know about it?"

    try:
        answer = ask_mim_general_chat(
            first_name=first_name,
            question=user_prompt,
            history=history_payload,
            image_bytes=image_bytes,
            image_mime_type=image_mime,
        )
    except Exception:
        current_app.logger.exception("ask-mim response generation failed for user_id=%s", g.user.id)
        answer = (
            "I couldn't process that request right now. "
            "Try again in a moment, or ask with slightly more detail."
        )

    user_message = MIMChatMessage(
        user_id=g.user.id,
        role="user",
        content=user_prompt,
        image_path=image_path,
        context="general",
    )
    persist_chat_secure_fields(g.user, user_message)

    assistant_message = MIMChatMessage(
        user_id=g.user.id,
        role="assistant",
        content=answer,
        context="general",
    )
    persist_chat_secure_fields(g.user, assistant_message)

    db.session.add(user_message)
    db.session.add(assistant_message)
    db.session.commit()

    return jsonify(
        {
            "ok": True,
            "user_message": {
                "id": user_message.id,
                "role": "user",
                "content": user_prompt,
                "image_path": image_path,
                "created_at": user_message.created_at.isoformat(),
            },
            "assistant_message": {
                "id": assistant_message.id,
                "role": "assistant",
                "content": answer,
                "created_at": assistant_message.created_at.isoformat(),
            },
        }
    )


@bp.post("/ask-mim/clear")
@login_required
def ask_mim_clear():
    MIMChatMessage.query.filter_by(user_id=g.user.id).delete(synchronize_session=False)
    db.session.commit()
    flash("MIM chat history cleared.", "success")
    return redirect(url_for("main.ask_mim_page"))


@bp.get("/community")
@login_required
def community_page():
    selected_category = get_community_filter_or_all(request.args.get("category"))
    author_id = parse_int(request.args.get("author"))

    post_query = (
        CommunityPost.query.join(User, CommunityPost.user_id == User.id).options(
            selectinload(CommunityPost.user),
            selectinload(CommunityPost.comments).selectinload(CommunityComment.user),
            selectinload(CommunityPost.likes),
        )
        .filter(
            CommunityPost.is_hidden.is_(False),
            User.is_blocked.is_(False),
            User.is_spam.is_(False),
        )
        .order_by(CommunityPost.created_at.desc(), CommunityPost.id.desc())
    )
    if selected_category != "all":
        post_query = post_query.filter(CommunityPost.category == selected_category)
    if author_id:
        post_query = post_query.filter(CommunityPost.user_id == author_id)

    posts = post_query.limit(120).all()
    post_cards = []
    for post in posts:
        ordered_comments = sorted(
            [
                row
                for row in post.comments
                if not row.is_hidden and row.user and not row.user.is_blocked and not row.user.is_spam
            ],
            key=lambda row: (row.created_at, row.id),
        )
        post_cards.append(
            {
                "post": post,
                "author_name": community_display_name(post.user),
                "like_count": len(post.likes),
                "comment_count": len(ordered_comments),
                "liked_by_me": any(like.user_id == g.user.id for like in post.likes),
                "comments": ordered_comments,
                "share": build_community_share_links(post),
            }
        )

    grouped_counts = dict(
        db.session.query(CommunityPost.category, func.count(CommunityPost.id))
        .join(User, CommunityPost.user_id == User.id)
        .filter(
            CommunityPost.is_hidden.is_(False),
            User.is_blocked.is_(False),
            User.is_spam.is_(False),
        )
        .group_by(CommunityPost.category)
        .all()
    )
    total_count = sum(grouped_counts.values())
    category_tabs = [
        {
            "key": "all",
            "label": "All",
            "count": total_count,
            "active": selected_category == "all",
        }
    ]
    for option in COMMUNITY_CATEGORY_OPTIONS:
        category_tabs.append(
            {
                "key": option["key"],
                "label": option["label"],
                "count": grouped_counts.get(option["key"], 0),
                "active": selected_category == option["key"],
            }
        )

    return render_template(
        "community.html",
        posts=post_cards,
        category_tabs=category_tabs,
        selected_category=selected_category,
        new_post_category=("general" if selected_category == "all" else selected_category),
        category_options=COMMUNITY_CATEGORY_OPTIONS,
    )


@bp.post("/community/post")
@login_required
def community_create_post():
    return_category = get_community_filter_or_all(request.form.get("return_category"))
    category = normalize_community_category(request.form.get("category"))
    title = normalize_text(request.form.get("title"))
    content = normalize_text(request.form.get("content"))

    if not title or len(title) < 4:
        flash("Post title must be at least 4 characters.", "error")
        return redirect(url_for("main.community_page", category=return_category))
    if not content or len(content) < 12:
        flash("Post content must be at least 12 characters.", "error")
        return redirect(url_for("main.community_page", category=return_category))
    if len(content) > 4000:
        flash("Post content is too long. Keep it under 4000 characters.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    blocked, reason = community_content_is_blocked(f"{title}\n{content}")
    if blocked:
        flash(reason or "This content was blocked by community safety rules.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    post = CommunityPost(
        user_id=g.user.id,
        category=category,
        title=title[:180],
        content=content,
    )
    db.session.add(post)
    db.session.commit()
    flash("Community post published.", "success")
    return redirect(url_for("main.community_page", category=category) + f"#post-{post.id}")


@bp.post("/community/<int:post_id>/comment")
@login_required
def community_add_comment(post_id: int):
    return_category = get_community_filter_or_all(request.form.get("category"))
    post = db.session.get(CommunityPost, post_id)
    if post is None or post.is_hidden:
        flash("That community post no longer exists.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    content = normalize_text(request.form.get("comment"))
    if not content or len(content) < 2:
        flash("Comment cannot be empty.", "error")
        return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")
    if len(content) > 1200:
        flash("Comment is too long. Keep it under 1200 characters.", "error")
        return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")

    blocked, reason = community_content_is_blocked(content)
    if blocked:
        flash(reason or "This comment was blocked by community safety rules.", "error")
        return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")

    db.session.add(
        CommunityComment(
            post_id=post.id,
            user_id=g.user.id,
            content=content,
        )
    )
    db.session.commit()
    return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")


@bp.post("/community/<int:post_id>/report")
@login_required
def community_report_post(post_id: int):
    return_category = get_community_filter_or_all(request.form.get("category"))
    post = db.session.get(CommunityPost, post_id)
    if post is None:
        flash("That community post no longer exists.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    reason = normalize_text(request.form.get("reason")) or "User report"
    reporter = community_display_name(g.user)
    post.is_flagged = True
    post.flag_reason = f"{reason} (reported by {reporter})"
    db.session.add(post)
    db.session.commit()
    flash("Post reported for review.", "success")
    return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")


@bp.post("/community/comment/<int:comment_id>/report")
@login_required
def community_report_comment(comment_id: int):
    return_category = get_community_filter_or_all(request.form.get("category"))
    comment = db.session.get(CommunityComment, comment_id)
    if comment is None:
        flash("That comment no longer exists.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    reason = normalize_text(request.form.get("reason")) or "User report"
    reporter = community_display_name(g.user)
    comment.is_flagged = True
    comment.flag_reason = f"{reason} (reported by {reporter})"
    db.session.add(comment)
    db.session.commit()
    flash("Comment reported for review.", "success")
    return redirect(url_for("main.community_page", category=return_category) + f"#post-{comment.post_id}")


@bp.post("/community/<int:post_id>/like")
@login_required
def community_toggle_like(post_id: int):
    return_category = get_community_filter_or_all(request.form.get("category"))
    post = db.session.get(CommunityPost, post_id)
    if post is None:
        flash("That community post no longer exists.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    existing = CommunityLike.query.filter_by(post_id=post.id, user_id=g.user.id).first()
    if existing:
        db.session.delete(existing)
    else:
        db.session.add(CommunityLike(post_id=post.id, user_id=g.user.id))
    db.session.commit()
    return redirect(url_for("main.community_page", category=return_category) + f"#post-{post.id}")


@bp.post("/community/<int:post_id>/delete")
@login_required
def community_delete_post(post_id: int):
    return_category = get_community_filter_or_all(request.form.get("category"))
    post = CommunityPost.query.filter_by(id=post_id, user_id=g.user.id).first()
    if post is None:
        flash("You can only delete your own community posts.", "error")
        return redirect(url_for("main.community_page", category=return_category))

    db.session.delete(post)
    db.session.commit()
    flash("Community post deleted.", "success")
    return redirect(url_for("main.community_page", category=return_category))


@bp.get("/substance")
@login_required
@profile_required
def substance_form():
    local_today = get_user_local_today(g.user)
    day_str = request.args.get("day")
    if day_str:
        try:
            selected_day = date.fromisoformat(day_str)
        except ValueError:
            selected_day = local_today
            flash("Invalid day format. Showing current local day.", "error")
    else:
        selected_day = local_today

    if selected_day > local_today:
        selected_day = local_today
        flash("Future entries are disabled. Showing current local day.", "error")

    default_time = datetime.now(get_user_zoneinfo(g.user)).strftime("%H:%M")
    default_taken_at = f"{selected_day.isoformat()}T{default_time}"

    return render_template(
        "substance.html",
        selected_day=selected_day.isoformat(),
        selected_day_weekday=selected_day.strftime("%A"),
        selected_day_pretty=selected_day.strftime("%B %d, %Y"),
        default_taken_at=default_taken_at,
    )


@bp.post("/substance")
@login_required
@profile_required
def substance_save():
    day_str = request.form.get("day")
    selected_day = None
    if day_str:
        try:
            selected_day = date.fromisoformat(day_str)
        except ValueError:
            selected_day = None

    taken_at_raw = request.form.get("taken_at") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M")
    try:
        taken_at = datetime.fromisoformat(taken_at_raw)
    except ValueError:
        flash("Invalid date/time for substance entry.", "error")
        if selected_day:
            return redirect(url_for("main.substance_form", day=selected_day.isoformat()))
        return redirect(url_for("main.substance_form"))

    kind = request.form.get("kind")
    if not kind:
        flash("Substance kind is required.", "error")
        if selected_day:
            return redirect(url_for("main.substance_form", day=selected_day.isoformat()))
        return redirect(url_for("main.substance_form"))

    entry = Substance(
        user_id=g.user.id,
        taken_at=taken_at,
        kind=kind,
        amount=request.form.get("amount") or None,
        notes=request.form.get("notes") or None,
    )
    persist_substance_secure_fields(g.user, entry)
    db.session.add(entry)
    db.session.commit()
    flash("Substance entry logged.", "success")
    if selected_day:
        return redirect(url_for("main.checkin_form", day=selected_day.isoformat()))
    return redirect(url_for("main.timeline"))


@bp.get("/timeline")
@login_required
@profile_required
def timeline():
    meals = Meal.query.filter_by(user_id=g.user.id).order_by(Meal.eaten_at.desc()).limit(80).all()
    checkins = DailyCheckIn.query.filter_by(user_id=g.user.id).order_by(DailyCheckIn.day.desc()).limit(45).all()
    substances = Substance.query.filter_by(user_id=g.user.id).order_by(Substance.taken_at.desc()).limit(45).all()
    for meal in meals:
        hydrate_meal_secure_fields(g.user, meal)
    for checkin in checkins:
        hydrate_checkin_secure_fields(g.user, checkin)
    for substance in substances:
        hydrate_substance_secure_fields(g.user, substance)

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
    for checkin in checkins:
        hydrate_checkin_secure_fields(g.user, checkin)
    for meal in meals:
        hydrate_meal_secure_fields(g.user, meal)

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
