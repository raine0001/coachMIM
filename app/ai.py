import os
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
