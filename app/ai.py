import base64
import json
import os
import re
from html import unescape
from ipaddress import ip_address
from socket import gaierror, getaddrinfo
from datetime import date, datetime, timedelta
from urllib.parse import urlparse
from typing import Sequence

import httpx
from openai import OpenAI

from app.models import DailyCheckIn, Meal


def coach_prompt_missing_fields(user, meals: Sequence[Meal], checkins: Sequence[DailyCheckIn]):
    prompts = []
    now = datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)

    checkins_by_day = {c.day: c for c in checkins}
    today_checkin = checkins_by_day.get(today)

    if now.hour >= 10 and not today_checkin:
        prompts.append("It is after 10am and today has no check-in. Log sleep, energy, focus, mood, and stress.")

    today_meals = [meal for meal in meals if meal.eaten_at.date() == today]
    if now.hour >= 14 and not today_meals:
        prompts.append("It is after 2pm and no meals are logged. Add at least one meal entry for data completeness.")

    if today_checkin and today_checkin.symptoms:
        headache = today_checkin.symptoms.get("headache")
        if headache and not today_checkin.sleep_hours:
            prompts.append("Headache was logged but sleep hours are missing. Add sleep to improve signal quality.")

    yday_checkin = checkins_by_day.get(yesterday)
    if yday_checkin and yday_checkin.alcohol_drinks and yday_checkin.alcohol_drinks > 0:
        if not today_checkin:
            prompts.append("Alcohol was logged yesterday. Add today's check-in to capture sleep quality and anxiety.")
        else:
            if today_checkin.sleep_quality is None:
                prompts.append("Yesterday included alcohol. Add today's sleep quality score (1-10).")
            if today_checkin.anxiety is None:
                prompts.append("Yesterday included alcohol. Add today's anxiety score (1-10).")

    if not prompts:
        prompts.append("Data quality looks good today. Keep logging meals and check-ins for cleaner trend detection.")

    return prompts


def ai_reflection(summary_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Set OPENAI_API_KEY to enable AI reflection."

    model = os.getenv("OPENAI_REFLECTION_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=f"You're a helpful performance coach. Give a concise reflection and 1-2 next questions.\n\n{summary_text}",
    )
    return resp.output_text or "No reflection generated."


def ask_mim_general_chat(
    *,
    first_name: str,
    question: str | None,
    history: Sequence[dict] | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "MIM chat needs OPENAI_API_KEY configured. "
            "I can still save your message, but AI response is disabled right now."
        )

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are MIM, a practical health-performance assistant. "
        "Provide general educational guidance only. "
        "Do not diagnose conditions, and do not prescribe medication. "
        "If user asks medical diagnosis or urgent symptoms, advise seeing a licensed clinician. "
        "Keep answers concise, useful, and action-oriented. "
        f"Address the user as {first_name} when natural."
    )

    chat_input = [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}]

    for entry in list(history or [])[-16:]:
        role = str(entry.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(entry.get("content") or "").strip()
        if not content:
            continue
        chat_input.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": content[:5000]}],
            }
        )

    user_text = (question or "").strip() or "Can you help me understand this?"
    user_content = [{"type": "input_text", "text": user_text[:6000]}]

    if image_bytes:
        mime = (image_mime_type or "image/jpeg").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_content.append({"type": "input_image", "image_url": f"data:{mime};base64,{image_b64}"})

    chat_input.append({"role": "user", "content": user_content})
    response = client.responses.create(model=model, input=chat_input)
    answer = (response.output_text or "").strip()
    return answer or "I couldn't generate a response. Please try rephrasing your question."


MEAL_PARSE_ALLOWED_UNITS = {"serving", "g", "ml", "oz", "lb", "cup", "tbsp", "tsp", "item"}
MEAL_PARSE_UNIT_ALIASES = {
    "serving": "serving",
    "servings": "serving",
    "portion": "serving",
    "portions": "serving",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tbs": "tbsp",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "item": "item",
    "items": "item",
    "piece": "item",
    "pieces": "item",
    "slice": "item",
    "slices": "item",
    "scoop": "item",
    "scoops": "item",
    "clove": "item",
    "cloves": "item",
}
MEAL_PARSE_NUMBER_WORDS = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "half": 0.5,
    "quarter": 0.25,
}
MEAL_PARSE_GENERIC_DISH_NAMES = {
    "hot choc",
    "hot chocolate",
    "hot cocoa",
    "protein shake",
    "shake",
    "smoothie",
    "drink",
}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _normalize_meal_unit(value) -> str:
    raw = str(value or "").strip().lower().replace(".", "")
    unit = MEAL_PARSE_UNIT_ALIASES.get(raw, raw)
    return unit if unit in MEAL_PARSE_ALLOWED_UNITS else "serving"


def _normalize_quantity_unit(quantity: float, unit_value) -> tuple[float, str]:
    raw = str(unit_value or "").strip().lower().replace(".", "")
    if raw in {"kg", "kilogram", "kilograms"}:
        return (quantity * 1000.0, "g")
    if raw in {"l", "liter", "liters"}:
        return (quantity * 1000.0, "ml")
    return (quantity, _normalize_meal_unit(raw))


def _parse_quantity_value(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if float(value) > 0:
            return float(value)
        return None

    text = str(value).strip().lower()
    text = text.replace("-", " ")
    text = text.replace("½", " 1/2 ")
    text = text.replace("¼", " 1/4 ")
    text = text.replace("¾", " 3/4 ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    if text in MEAL_PARSE_NUMBER_WORDS:
        return MEAL_PARSE_NUMBER_WORDS[text]

    mixed_match = re.fullmatch(r"(\d+)\s+(\d+)\/(\d+)", text)
    if mixed_match:
        whole = float(mixed_match.group(1))
        numerator = float(mixed_match.group(2))
        denominator = float(mixed_match.group(3))
        if denominator != 0:
            return whole + (numerator / denominator)

    fraction_match = re.fullmatch(r"(\d+)\/(\d+)", text)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator != 0:
            return numerator / denominator

    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        try:
            parsed = float(number_match.group(0))
            return parsed if parsed > 0 else None
        except ValueError:
            return None
    return None


def _normalize_parsed_ingredients(raw_ingredients) -> list[dict]:
    if not isinstance(raw_ingredients, list):
        return []

    normalized = []
    for item in raw_ingredients[:24]:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or item.get("ingredient") or item.get("item") or "").strip()
        if not name:
            continue

        quantity = _parse_quantity_value(item.get("quantity"))
        if quantity is None:
            quantity = 1.0
        quantity, unit = _normalize_quantity_unit(float(quantity), item.get("unit"))

        normalized.append(
            {
                "name": name[:255],
                "quantity": float(quantity),
                "unit": unit,
                "calories": _as_float(item.get("calories")),
                "protein_g": _as_float(item.get("protein_g")),
                "carbs_g": _as_float(item.get("carbs_g")),
                "fat_g": _as_float(item.get("fat_g")),
                "sugar_g": _as_float(item.get("sugar_g")),
                "sodium_mg": _as_float(item.get("sodium_mg")),
                "caffeine_mg": _as_float(item.get("caffeine_mg")),
            }
        )

    return normalized


def _looks_like_beverage(text: str) -> bool:
    lowered = (text or "").lower()
    beverage_tokens = {
        "shake",
        "smoothie",
        "coffee",
        "tea",
        "juice",
        "drink",
        "latte",
        "cocoa",
        "hot chocolate",
        "hot choc",
        "protein drink",
        "milk",
    }
    return any(token in lowered for token in beverage_tokens)


def _derive_meal_title_from_text(text: str) -> str | None:
    lowered = (text or "").lower()
    if "hot chocolate" in lowered or "hot choc" in lowered or "hot cocoa" in lowered:
        return "Hot Chocolate"
    if "protein shake" in lowered or "shake" in lowered:
        return "Protein Shake"
    if "smoothie" in lowered:
        return "Smoothie"
    if "salad" in lowered:
        return "Salad"
    return None


def _fallback_parse_meal_sentence(text: str) -> dict:
    sentence = str(text or "").strip()
    if not sentence:
        return {"meal_title": None, "meal_label": None, "is_beverage": False, "ingredients": [], "source": "regex"}

    working = sentence
    working = re.sub(r"\btable\s*spoons?\b", "tablespoon", working, flags=re.IGNORECASE)
    working = re.sub(r"\btea\s*spoons?\b", "teaspoon", working, flags=re.IGNORECASE)
    working = re.sub(r"\b(?:using|utilizing|with|plus)\b", ",", working, flags=re.IGNORECASE)
    working = re.sub(r"\s+and\s+", ", ", working, flags=re.IGNORECASE)
    working = re.sub(r"\s+", " ", working).strip(" .")

    segments = [part.strip(" .") for part in working.split(",") if part.strip(" .")]
    ingredients: list[dict] = []

    lead_strip = re.compile(r"^(?:i\s+)?(?:make|made|had|having|drink|drank|ate)\s+(?:a|an|my)?\s*", flags=re.IGNORECASE)
    item_pattern = re.compile(
        r"^(?:(?P<qty>\d+\s+\d+\/\d+|\d+\/\d+|\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter)\s+)?"
        r"(?:(?P<unit>tablespoons?|tbsp|teaspoons?|tsp|cups?|oz|ounces?|pounds?|lbs?|grams?|g|ml|milliliters?|liters?|servings?|items?|pieces?|slices?|scoops?)\s+)?"
        r"(?:(?:of)\s+)?(?P<name>.+)$",
        flags=re.IGNORECASE,
    )

    for segment in segments[:24]:
        cleaned = lead_strip.sub("", segment).strip()
        if not cleaned:
            continue

        match = item_pattern.match(cleaned)
        if not match:
            continue

        name = str(match.group("name") or "").strip(" .")
        if not name:
            continue

        raw_qty = match.group("qty")
        raw_unit = match.group("unit")
        normalized_name = re.sub(r"\s+", " ", name.lower()).strip()
        if not raw_qty and not raw_unit and normalized_name in MEAL_PARSE_GENERIC_DISH_NAMES:
            continue

        quantity = _parse_quantity_value(raw_qty) or 1.0
        quantity, unit = _normalize_quantity_unit(float(quantity), match.group("unit"))
        ingredients.append({"name": name[:255], "quantity": float(quantity), "unit": unit})

    if not ingredients:
        ingredients = [{"name": sentence[:255], "quantity": 1.0, "unit": "serving"}]

    return {
        "meal_title": _derive_meal_title_from_text(sentence),
        "meal_label": "Drink" if _looks_like_beverage(sentence) else None,
        "is_beverage": _looks_like_beverage(sentence),
        "ingredients": ingredients,
        "source": "regex",
    }


def _ai_parse_meal_sentence(text: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    model = os.getenv("OPENAI_MEAL_PARSE_MODEL", "gpt-4.1-mini")
    prompt = (
        "Parse this one-sentence meal description into structured JSON.\n"
        "Return JSON only using this schema:\n"
        "{\n"
        '  "meal_title": string or null,\n'
        '  "meal_label": string or null,\n'
        '  "is_beverage": boolean,\n'
        '  "ingredients": [\n'
        "    {\n"
        '      "name": string,\n'
        '      "quantity": number,\n'
        '      "unit": "serving" | "g" | "ml" | "oz" | "lb" | "cup" | "tbsp" | "tsp" | "item"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Do not include markdown.\n"
        "- Use numbers for quantity (convert fractions like 1/4 to 0.25).\n"
        "- Use unit='serving' when unit is unknown.\n"
        "- Keep ingredient names concise.\n"
        f"Sentence: {text}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    parsed = _extract_json_object(response.output_text or "")
    ingredients = _normalize_parsed_ingredients(parsed.get("ingredients"))
    if not ingredients:
        return {}

    return {
        "meal_title": str(parsed.get("meal_title") or "").strip()[:120] or None,
        "meal_label": str(parsed.get("meal_label") or "").strip()[:120] or None,
        "is_beverage": _as_bool(parsed.get("is_beverage")),
        "ingredients": ingredients,
        "source": "ai",
    }


def parse_meal_sentence(sentence_text: str) -> dict:
    text = str(sentence_text or "").strip()
    if len(text) < 6:
        raise RuntimeError("Provide a longer sentence so MIM can parse ingredients.")

    ai_result = {}
    try:
        ai_result = _ai_parse_meal_sentence(text)
    except Exception:
        ai_result = {}

    if ai_result.get("ingredients"):
        if not ai_result.get("meal_title"):
            ai_result["meal_title"] = _derive_meal_title_from_text(text)
        if not ai_result.get("meal_label") and ai_result.get("is_beverage"):
            ai_result["meal_label"] = "Drink"
        if not ai_result.get("is_beverage"):
            ai_result["is_beverage"] = _looks_like_beverage(text)
        return ai_result

    fallback = _fallback_parse_meal_sentence(text)
    if not fallback.get("ingredients"):
        raise RuntimeError("MIM could not parse ingredients from that sentence.")
    return fallback


def _normalize_substance_kind(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"alcohol", "caffeine", "nicotine", "other"} else "other"


def _normalize_med_kind(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"medication", "supplement"} else "medication"


def _as_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fallback_day_manager_assist(context: str, text: str, first_name: str) -> dict:
    cleaned = str(text or "").strip()
    lowered = cleaned.lower()
    if len(cleaned) < 5:
        target = "entry"
        if context == "substance":
            target = "substance"
        elif context == "activity":
            target = "activity"
        elif context == "medications":
            target = "medication or supplement"
        return {
            "needs_more": True,
            "reply": f"Hi {first_name}, tell me a bit more about this {target} and I can prefill the form.",
            "follow_up_prompt": "Include what it was and amount/time details.",
            "suggested_fields": {},
        }

    if context == "substance":
        kind = "other"
        if any(token in lowered for token in {"beer", "wine", "whiskey", "vodka", "alcohol", "drink", "cocktail", "shot"}):
            kind = "alcohol"
        elif any(token in lowered for token in {"coffee", "espresso", "tea", "caffeine", "energy drink"}):
            kind = "caffeine"
        elif any(token in lowered for token in {"nicotine", "cigarette", "vape", "pouch", "zyn"}):
            kind = "nicotine"

        amount_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(beers?|drinks?|shots?|cups?|oz|ml|mg|cans?|cigarettes?|pouches?|pieces?|gums?)",
            cleaned,
            flags=re.IGNORECASE,
        )
        amount = amount_match.group(0) if amount_match else cleaned[:120]

        return {
            "needs_more": False,
            "reply": f"Hi {first_name}, I filled the substance type and amount. Check it before saving.",
            "follow_up_prompt": None,
            "suggested_fields": {
                "kind": kind,
                "amount": amount,
                "notes": cleaned[:500],
            },
        }

    if context == "activity":
        activity_type = None
        for token in [
            "run",
            "walk",
            "lift",
            "strength",
            "cycling",
            "bike",
            "swim",
            "yoga",
            "pilates",
            "hiit",
            "hike",
            "jog",
        ]:
            if token in lowered:
                activity_type = token
                break
        if activity_type is None:
            activity_type = cleaned[:80]

        duration_min = None
        hours_match = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr)\b", lowered)
        if hours_match:
            duration_min = int(round(float(hours_match.group(1)) * 60))
        else:
            min_match = re.search(r"(\d{1,3})\s*(minutes?|mins?|min)\b", lowered)
            if min_match:
                duration_min = int(min_match.group(1))

        intensity = None
        intensity_match = re.search(r"(\d{1,2})\s*/\s*10", lowered)
        if intensity_match:
            intensity = max(1, min(10, int(intensity_match.group(1))))
        else:
            intensity_match = re.search(r"intensity\s*(\d{1,2})", lowered)
            if intensity_match:
                intensity = max(1, min(10, int(intensity_match.group(1))))

        return {
            "needs_more": duration_min is None,
            "reply": (
                f"Hi {first_name}, I drafted your activity entry."
                if duration_min is not None
                else f"Hi {first_name}, I captured activity type. Add duration for better tracking."
            ),
            "follow_up_prompt": None if duration_min is not None else "How many minutes did you do it?",
            "suggested_fields": {
                "activity_type": activity_type,
                "duration_min": duration_min,
                "intensity": intensity,
                "notes": cleaned[:500],
            },
        }

    kind = "supplement" if any(
        token in lowered
        for token in {"supplement", "vitamin", "omega", "magnesium", "creatine", "electrolyte", "zinc"}
    ) else "medication"
    dose_match = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?|tabs?|tablets?|capsules?|pills?)", lowered)
    dose = dose_match.group(0) if dose_match else None

    med_name = cleaned
    if dose_match:
        med_name = cleaned[: dose_match.start()].strip(" ,.-")
    med_name = med_name[:120]

    return {
        "needs_more": False,
        "reply": f"Hi {first_name}, I filled medication/supplement details. Verify and save.",
        "follow_up_prompt": None,
        "suggested_fields": {
            "kind": kind,
            "med_name": med_name,
            "dose": dose,
            "notes": cleaned[:500],
        },
    }


def _ai_day_manager_assist(context: str, text: str, first_name: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    model = os.getenv("OPENAI_DAY_MANAGER_MODEL", "gpt-4.1-mini")
    context_schema = {
        "substance": (
            '{ "kind": "alcohol|caffeine|nicotine|other", "amount": string|null, "notes": string|null }',
            "substance",
        ),
        "activity": (
            '{ "activity_type": string|null, "duration_min": number|null, "intensity": number|null, "notes": string|null }',
            "activity",
        ),
        "medications": (
            '{ "kind": "medication|supplement", "med_name": string|null, "dose": string|null, "notes": string|null }',
            "medication or supplement",
        ),
    }
    field_schema, entry_label = context_schema.get(
        context,
        (
            '{ "kind": "other", "amount": string|null, "notes": string|null }',
            "entry",
        ),
    )

    prompt = (
        "You are MIM, a concise behavior/performance logging assistant.\n"
        "Given one user sentence, return JSON only with this schema:\n"
        "{\n"
        '  "needs_more": boolean,\n'
        '  "reply": string,\n'
        '  "follow_up_prompt": string|null,\n'
        f'  "suggested_fields": {field_schema}\n'
        "}\n"
        "Rules:\n"
        "- Keep reply short and practical.\n"
        "- Use first name naturally.\n"
        "- Do not provide diagnosis or treatment advice.\n"
        "- If details are missing, set needs_more=true and ask one specific follow-up question.\n"
        f"User first name: {first_name}\n"
        f"Context: {entry_label}\n"
        f"User text: {text}\n"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    parsed = _extract_json_object(response.output_text or "")
    if not isinstance(parsed, dict):
        return {}

    suggested = parsed.get("suggested_fields")
    if not isinstance(suggested, dict):
        suggested = {}

    result = {
        "needs_more": _as_bool(parsed.get("needs_more")),
        "reply": str(parsed.get("reply") or "").strip(),
        "follow_up_prompt": str(parsed.get("follow_up_prompt") or "").strip() or None,
        "suggested_fields": suggested,
    }

    if context == "substance":
        result["suggested_fields"]["kind"] = _normalize_substance_kind(result["suggested_fields"].get("kind"))
        result["suggested_fields"]["amount"] = str(result["suggested_fields"].get("amount") or "").strip() or None
        result["suggested_fields"]["notes"] = str(result["suggested_fields"].get("notes") or "").strip() or None
    elif context == "activity":
        result["suggested_fields"]["activity_type"] = str(result["suggested_fields"].get("activity_type") or "").strip() or None
        duration = _as_int(result["suggested_fields"].get("duration_min"))
        result["suggested_fields"]["duration_min"] = duration if duration and duration > 0 else None
        intensity = _as_int(result["suggested_fields"].get("intensity"))
        result["suggested_fields"]["intensity"] = intensity if intensity and 1 <= intensity <= 10 else None
        result["suggested_fields"]["notes"] = str(result["suggested_fields"].get("notes") or "").strip() or None
    elif context == "medications":
        result["suggested_fields"]["kind"] = _normalize_med_kind(result["suggested_fields"].get("kind"))
        result["suggested_fields"]["med_name"] = str(result["suggested_fields"].get("med_name") or "").strip() or None
        result["suggested_fields"]["dose"] = str(result["suggested_fields"].get("dose") or "").strip() or None
        result["suggested_fields"]["notes"] = str(result["suggested_fields"].get("notes") or "").strip() or None

    return result


def parse_day_manager_context_assist(context: str, text: str, first_name: str = "there") -> dict:
    normalized_context = str(context or "").strip().lower()
    if normalized_context not in {"substance", "activity", "medications"}:
        raise RuntimeError("Unsupported context for day-manager assist.")

    cleaned_text = str(text or "").strip()
    if len(cleaned_text) < 2:
        return _fallback_day_manager_assist(normalized_context, cleaned_text, first_name)

    try:
        ai_result = _ai_day_manager_assist(normalized_context, cleaned_text, first_name)
    except Exception:
        ai_result = {}

    if isinstance(ai_result, dict) and isinstance(ai_result.get("suggested_fields"), dict) and (
        ai_result.get("reply") or ai_result.get("suggested_fields")
    ):
        if not ai_result.get("reply"):
            ai_result["reply"] = f"Hi {first_name}, I prefilled this form. Review before saving."
        ai_result["needs_more"] = bool(ai_result.get("needs_more"))
        ai_result["follow_up_prompt"] = ai_result.get("follow_up_prompt")
        return ai_result

    return _fallback_day_manager_assist(normalized_context, cleaned_text, first_name)


def _extract_json_object(raw_text: str) -> dict:
    text = (raw_text or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def parse_nutrition_label_image(image_bytes: bytes, mime_type: str | None, hint_name: str | None = None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    if not image_bytes:
        raise RuntimeError("No image data was provided.")

    model = os.getenv("OPENAI_OCR_MODEL", "gpt-4.1-mini")
    mime = (mime_type or "image/jpeg").split(";")[0].strip()
    if not mime.startswith("image/"):
        mime = "image/jpeg"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime};base64,{image_b64}"
    hint_line = f"Possible product name: {hint_name}" if hint_name else "No product-name hint provided."

    prompt = (
        "Read the nutrition label from this image and return JSON only with these keys:\n"
        "name, serving_size_value, serving_size_unit, calories, protein_g, carbs_g, fat_g, sugar_g, sodium_mg, caffeine_mg, confidence.\n"
        "Use numeric values when possible. Use null if unknown. Confidence should be 0.0 to 1.0.\n"
        f"{hint_line}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )

    parsed = _extract_json_object(response.output_text or "")

    def as_float(value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    normalized = {
        "name": str(parsed.get("name") or hint_name or "").strip() or None,
        "serving_size_value": as_float(parsed.get("serving_size_value")),
        "serving_size_unit": str(parsed.get("serving_size_unit") or "").strip()[:32] or None,
        "calories": as_float(parsed.get("calories")),
        "protein_g": as_float(parsed.get("protein_g")),
        "carbs_g": as_float(parsed.get("carbs_g")),
        "fat_g": as_float(parsed.get("fat_g")),
        "sugar_g": as_float(parsed.get("sugar_g")),
        "sodium_mg": as_float(parsed.get("sodium_mg")),
        "caffeine_mg": as_float(parsed.get("caffeine_mg")),
        "confidence": as_float(parsed.get("confidence")),
    }
    return normalized


def _private_or_local_ip(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if not host:
        return True
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True

    try:
        parsed = ip_address(host)
        return (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_reserved
            or parsed.is_multicast
        )
    except ValueError:
        pass

    try:
        infos = getaddrinfo(host, None)
    except gaierror:
        return False
    except Exception:
        return True

    for info in infos:
        addr = info[4][0]
        try:
            parsed = ip_address(addr)
        except ValueError:
            continue
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local
            or parsed.is_reserved
            or parsed.is_multicast
        ):
            return True
    return False


def _validate_product_url(product_url: str) -> str:
    candidate = (product_url or "").strip()
    if not candidate:
        raise RuntimeError("Product URL is required.")

    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise RuntimeError("Only http/https product URLs are supported.")
    if not parsed.netloc:
        raise RuntimeError("Invalid product URL.")
    if _private_or_local_ip(parsed.hostname or ""):
        raise RuntimeError("Private or local network URLs are not allowed.")

    return parsed.geturl()


def _clean_text(html_text: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _first_pattern(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _as_float(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_serving_size(value) -> tuple[float | None, str | None]:
    if value in (None, ""):
        return (None, None)
    text = str(value).strip()
    # Prefer grams/ml when present in parentheses: "1 tbsp (6 g)"
    paren_match = re.search(r"\((\d+(?:\.\d+)?)\s*([a-zA-Z]+)\)", text)
    if paren_match:
        return (_as_float(paren_match.group(1)), paren_match.group(2).lower()[:32])

    match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", text)
    if match:
        return (_as_float(match.group(1)), match.group(2).lower()[:32])
    return (_as_float(text), None)


def _parse_sodium_mg(value) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).lower()
    number = _as_float(text)
    if number is None:
        return None
    if "mg" in text:
        return number
    if " g" in text or text.endswith("g"):
        return number * 1000
    return number


def _extract_title(html_text: str) -> str | None:
    og = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if og:
        return unescape(og.group(1)).strip()[:255] or None

    title = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if title:
        return unescape(title.group(1)).strip()[:255] or None
    return None


def _iter_nodes(data):
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _iter_nodes(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_nodes(item)


def _node_has_type(node: dict, target: str) -> bool:
    raw_type = node.get("@type")
    if isinstance(raw_type, str):
        return raw_type.lower() == target.lower() or target.lower() in raw_type.lower()
    if isinstance(raw_type, list):
        return any(isinstance(t, str) and target.lower() in t.lower() for t in raw_type)
    return False


def _extract_from_jsonld(html_text: str) -> dict:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_script in scripts:
        raw_script = raw_script.strip()
        if not raw_script:
            continue
        try:
            parsed = json.loads(raw_script)
        except json.JSONDecodeError:
            continue

        for node in _iter_nodes(parsed):
            if not isinstance(node, dict) or not _node_has_type(node, "Product"):
                continue

            nutrition = node.get("nutrition") if isinstance(node.get("nutrition"), dict) else {}
            serving_value, serving_unit = _parse_serving_size(nutrition.get("servingSize"))
            return {
                "name": str(node.get("name") or "").strip()[:255] or None,
                "serving_size_value": serving_value,
                "serving_size_unit": serving_unit,
                "calories": _as_float(nutrition.get("calories") or nutrition.get("energy")),
                "protein_g": _as_float(nutrition.get("proteinContent")),
                "carbs_g": _as_float(nutrition.get("carbohydrateContent")),
                "fat_g": _as_float(nutrition.get("fatContent")),
                "sugar_g": _as_float(nutrition.get("sugarContent")),
                "sodium_mg": _parse_sodium_mg(nutrition.get("sodiumContent")),
                "caffeine_mg": _as_float(nutrition.get("caffeineContent")),
                "confidence": 0.9,
                "source_method": "jsonld",
            }
    return {}


def _extract_from_text(text: str) -> dict:
    serving_raw = _first_pattern(
        text,
        [
            r"serving size\s*[:\-]?\s*([0-9][^,.;]{0,24})",
            r"per serving\s*[:\-]?\s*([0-9][^,.;]{0,24})",
        ],
    )
    serving_value, serving_unit = _parse_serving_size(serving_raw)

    calories = _first_pattern(text, [r"calories?\s*[:\-]?\s*(\d+(?:\.\d+)?)"])
    protein = _first_pattern(text, [r"protein\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?"])
    carbs = _first_pattern(
        text,
        [
            r"(?:total\s+)?carbohydrates?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?",
            r"carbs?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?",
        ],
    )
    fat = _first_pattern(text, [r"(?:total\s+)?fat\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?"])
    sugar = _first_pattern(text, [r"(?:total\s+)?sugars?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*g?"])
    sodium_match = _first_pattern(text, [r"sodium\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g)?"])
    caffeine_match = _first_pattern(text, [r"caffeine\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g)?"])

    sodium_mg = None
    if sodium_match:
        sodium_search = re.search(r"sodium\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g)?", text, flags=re.IGNORECASE)
        if sodium_search:
            sodium_val = _as_float(sodium_search.group(1))
            sodium_unit = (sodium_search.group(2) or "mg").lower()
            if sodium_val is not None:
                sodium_mg = sodium_val * (1000 if sodium_unit == "g" else 1)

    caffeine_mg = None
    if caffeine_match:
        caffeine_search = re.search(r"caffeine\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g)?", text, flags=re.IGNORECASE)
        if caffeine_search:
            caffeine_val = _as_float(caffeine_search.group(1))
            caffeine_unit = (caffeine_search.group(2) or "mg").lower()
            if caffeine_val is not None:
                caffeine_mg = caffeine_val * (1000 if caffeine_unit == "g" else 1)

    return {
        "serving_size_value": serving_value,
        "serving_size_unit": serving_unit,
        "calories": _as_float(calories),
        "protein_g": _as_float(protein),
        "carbs_g": _as_float(carbs),
        "fat_g": _as_float(fat),
        "sugar_g": _as_float(sugar),
        "sodium_mg": sodium_mg,
        "caffeine_mg": caffeine_mg,
        "confidence": 0.55,
        "source_method": "text_pattern",
    }


def _merge_nutrition(primary: dict, fallback: dict) -> dict:
    merged = dict(primary or {})
    for key in [
        "name",
        "serving_size_value",
        "serving_size_unit",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "sugar_g",
        "sodium_mg",
        "caffeine_mg",
        "confidence",
        "source_method",
    ]:
        if merged.get(key) in (None, "") and fallback.get(key) not in (None, ""):
            merged[key] = fallback.get(key)
    return merged


def _parse_product_with_ai(page_title: str | None, page_text: str, product_url: str, hint_name: str | None) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}

    model = os.getenv("OPENAI_PRODUCT_PARSE_MODEL", "gpt-4.1-mini")
    excerpt = (page_text or "")[:14000]
    prompt = (
        "Extract product nutrition from this page text and return JSON only with keys:\n"
        "name, serving_size_value, serving_size_unit, calories, protein_g, carbs_g, fat_g, sugar_g, sodium_mg, caffeine_mg, confidence.\n"
        "Use numeric values where possible, null when unknown.\n"
        f"URL: {product_url}\n"
        f"Title: {page_title or ''}\n"
        f"Hint Name: {hint_name or ''}\n\n"
        f"Page text excerpt:\n{excerpt}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=model, input=prompt)
    parsed = _extract_json_object(response.output_text or "")

    return {
        "name": str(parsed.get("name") or "").strip()[:255] or None,
        "serving_size_value": _as_float(parsed.get("serving_size_value")),
        "serving_size_unit": str(parsed.get("serving_size_unit") or "").strip()[:32] or None,
        "calories": _as_float(parsed.get("calories")),
        "protein_g": _as_float(parsed.get("protein_g")),
        "carbs_g": _as_float(parsed.get("carbs_g")),
        "fat_g": _as_float(parsed.get("fat_g")),
        "sugar_g": _as_float(parsed.get("sugar_g")),
        "sodium_mg": _as_float(parsed.get("sodium_mg")),
        "caffeine_mg": _as_float(parsed.get("caffeine_mg")),
        "confidence": _as_float(parsed.get("confidence")) or 0.65,
        "source_method": "ai_text_parse",
    }


def parse_product_page_url(product_url: str, hint_name: str | None = None) -> dict:
    safe_url = _validate_product_url(product_url)

    try:
        response = httpx.get(
            safe_url,
            follow_redirects=True,
            timeout=12.0,
            headers={"User-Agent": "CoachMIM/1.0 (+nutrition-parser)"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not fetch product page: {exc}") from exc

    html_text = response.text[:2_000_000]
    page_title = _extract_title(html_text)
    text = _clean_text(html_text)

    from_jsonld = _extract_from_jsonld(html_text)
    from_text = _extract_from_text(text)
    merged = _merge_nutrition(from_jsonld, from_text)

    found_fields = sum(
        merged.get(key) not in (None, "")
        for key in ["calories", "protein_g", "carbs_g", "fat_g", "sugar_g", "sodium_mg", "caffeine_mg"]
    )

    if found_fields < 2:
        ai_guess = _parse_product_with_ai(page_title, text, safe_url, hint_name)
        merged = _merge_nutrition(merged, ai_guess)

    if not merged.get("name"):
        merged["name"] = (hint_name or page_title or "").strip()[:255] or "Product from link"
    merged["source_url"] = safe_url
    if not merged.get("source_method"):
        merged["source_method"] = "title_only"
    if merged.get("confidence") is None:
        merged["confidence"] = 0.4

    return merged
