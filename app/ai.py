import os
import base64
import json
import re
from datetime import date, datetime, timedelta
from typing import Sequence

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
        "name, serving_size_value, serving_size_unit, calories, protein_g, carbs_g, fat_g, sugar_g, sodium_mg, confidence.\n"
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
        "confidence": as_float(parsed.get("confidence")),
    }
    return normalized
