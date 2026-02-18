import os
from typing import Any

import httpx

from app import db
from app.models import FoodItem

_COMMON_SEED_SYNCED = False

COMMON_FOODS = [
    {"name": "Egg, whole", "serving_size": 1, "serving_unit": "large", "calories": 72, "protein_g": 6.3, "carbs_g": 0.4, "fat_g": 4.8},
    {"name": "Egg white", "serving_size": 1, "serving_unit": "large", "calories": 17, "protein_g": 3.6, "carbs_g": 0.2, "fat_g": 0.1},
    {"name": "Chicken breast, cooked", "serving_size": 100, "serving_unit": "g", "calories": 165, "protein_g": 31, "carbs_g": 0, "fat_g": 3.6},
    {"name": "Ground beef, 90% lean", "serving_size": 100, "serving_unit": "g", "calories": 217, "protein_g": 26, "carbs_g": 0, "fat_g": 12},
    {"name": "Salmon, cooked", "serving_size": 100, "serving_unit": "g", "calories": 206, "protein_g": 22, "carbs_g": 0, "fat_g": 12},
    {"name": "Tuna, canned in water", "serving_size": 100, "serving_unit": "g", "calories": 116, "protein_g": 26, "carbs_g": 0, "fat_g": 1},
    {"name": "Greek yogurt, plain nonfat", "serving_size": 170, "serving_unit": "g", "calories": 100, "protein_g": 17, "carbs_g": 6, "fat_g": 0},
    {"name": "Cottage cheese, low-fat", "serving_size": 113, "serving_unit": "g", "calories": 90, "protein_g": 12, "carbs_g": 4, "fat_g": 2.5},
    {"name": "Milk, 2%", "serving_size": 240, "serving_unit": "ml", "calories": 122, "protein_g": 8.1, "carbs_g": 12, "fat_g": 4.8},
    {"name": "Cheddar cheese", "serving_size": 28, "serving_unit": "g", "calories": 113, "protein_g": 7, "carbs_g": 0.4, "fat_g": 9},
    {"name": "Whey protein powder", "serving_size": 30, "serving_unit": "g", "calories": 120, "protein_g": 24, "carbs_g": 3, "fat_g": 1.5},
    {"name": "Tofu, firm", "serving_size": 100, "serving_unit": "g", "calories": 144, "protein_g": 17, "carbs_g": 3, "fat_g": 9},
    {"name": "Lentils, cooked", "serving_size": 100, "serving_unit": "g", "calories": 116, "protein_g": 9, "carbs_g": 20, "fat_g": 0.4},
    {"name": "Black beans, cooked", "serving_size": 100, "serving_unit": "g", "calories": 132, "protein_g": 8.9, "carbs_g": 24, "fat_g": 0.5},
    {"name": "Brown rice, cooked", "serving_size": 100, "serving_unit": "g", "calories": 112, "protein_g": 2.6, "carbs_g": 23, "fat_g": 0.8},
    {"name": "White rice, cooked", "serving_size": 100, "serving_unit": "g", "calories": 130, "protein_g": 2.4, "carbs_g": 28, "fat_g": 0.3},
    {"name": "Quinoa, cooked", "serving_size": 100, "serving_unit": "g", "calories": 120, "protein_g": 4.4, "carbs_g": 21.3, "fat_g": 1.9},
    {"name": "Oats, dry", "serving_size": 40, "serving_unit": "g", "calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 3},
    {"name": "Pancakes, plain", "serving_size": 2, "serving_unit": "medium", "calories": 175, "protein_g": 4.7, "carbs_g": 28.3, "fat_g": 4.5},
    {"name": "Waffles, plain", "serving_size": 2, "serving_unit": "small", "calories": 218, "protein_g": 5.9, "carbs_g": 29.7, "fat_g": 8.3},
    {"name": "French toast", "serving_size": 2, "serving_unit": "slices", "calories": 290, "protein_g": 10, "carbs_g": 35, "fat_g": 12},
    {"name": "Bacon", "serving_size": 2, "serving_unit": "slices", "calories": 86, "protein_g": 6, "carbs_g": 0.3, "fat_g": 6.7},
    {"name": "Sausage link", "serving_size": 2, "serving_unit": "links", "calories": 170, "protein_g": 7, "carbs_g": 1.5, "fat_g": 15},
    {"name": "Hash browns", "serving_size": 85, "serving_unit": "g", "calories": 150, "protein_g": 2, "carbs_g": 18, "fat_g": 8},
    {"name": "Bagel, plain", "serving_size": 1, "serving_unit": "item", "calories": 277, "protein_g": 10.5, "carbs_g": 55, "fat_g": 1.7},
    {"name": "Cream cheese", "serving_size": 2, "serving_unit": "tbsp", "calories": 99, "protein_g": 1.8, "carbs_g": 1.6, "fat_g": 9.8},
    {"name": "Jam", "serving_size": 1, "serving_unit": "tbsp", "calories": 56, "protein_g": 0.1, "carbs_g": 14, "fat_g": 0},
    {"name": "Turkey sandwich", "serving_size": 1, "serving_unit": "item", "calories": 320, "protein_g": 21, "carbs_g": 33, "fat_g": 12},
    {"name": "Ham sandwich", "serving_size": 1, "serving_unit": "item", "calories": 360, "protein_g": 19, "carbs_g": 35, "fat_g": 15},
    {"name": "Chicken salad sandwich", "serving_size": 1, "serving_unit": "item", "calories": 410, "protein_g": 20, "carbs_g": 33, "fat_g": 22},
    {"name": "Whole wheat bread", "serving_size": 1, "serving_unit": "slice", "calories": 80, "protein_g": 4, "carbs_g": 14, "fat_g": 1},
    {"name": "Sourdough bread", "serving_size": 1, "serving_unit": "slice", "calories": 93, "protein_g": 3.4, "carbs_g": 18, "fat_g": 0.7},
    {"name": "Tortilla, flour", "serving_size": 1, "serving_unit": "medium", "calories": 140, "protein_g": 4, "carbs_g": 24, "fat_g": 3.5},
    {"name": "Pasta, cooked", "serving_size": 100, "serving_unit": "g", "calories": 157, "protein_g": 5.8, "carbs_g": 31, "fat_g": 0.9},
    {"name": "Potato, baked", "serving_size": 1, "serving_unit": "medium", "calories": 161, "protein_g": 4.3, "carbs_g": 37, "fat_g": 0.2},
    {"name": "Sweet potato, baked", "serving_size": 1, "serving_unit": "medium", "calories": 112, "protein_g": 2, "carbs_g": 26, "fat_g": 0.1},
    {"name": "Banana", "serving_size": 1, "serving_unit": "medium", "calories": 105, "protein_g": 1.3, "carbs_g": 27, "fat_g": 0.4},
    {"name": "Apple", "serving_size": 1, "serving_unit": "medium", "calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3},
    {"name": "Blueberries", "serving_size": 100, "serving_unit": "g", "calories": 57, "protein_g": 0.7, "carbs_g": 14, "fat_g": 0.3},
    {"name": "Avocado", "serving_size": 100, "serving_unit": "g", "calories": 160, "protein_g": 2, "carbs_g": 8.5, "fat_g": 14.7},
    {"name": "Spinach", "serving_size": 100, "serving_unit": "g", "calories": 23, "protein_g": 2.9, "carbs_g": 3.6, "fat_g": 0.4},
    {"name": "Broccoli", "serving_size": 100, "serving_unit": "g", "calories": 35, "protein_g": 2.4, "carbs_g": 7.2, "fat_g": 0.4},
    {"name": "Mixed salad greens", "serving_size": 85, "serving_unit": "g", "calories": 15, "protein_g": 1.4, "carbs_g": 2.9, "fat_g": 0.2},
    {"name": "Olive oil", "serving_size": 1, "serving_unit": "tbsp", "calories": 119, "protein_g": 0, "carbs_g": 0, "fat_g": 13.5},
    {"name": "Butter", "serving_size": 1, "serving_unit": "tbsp", "calories": 102, "protein_g": 0.1, "carbs_g": 0, "fat_g": 11.5},
    {"name": "Almonds", "serving_size": 28, "serving_unit": "g", "calories": 164, "protein_g": 6, "carbs_g": 6.1, "fat_g": 14.2},
    {"name": "Peanut butter", "serving_size": 2, "serving_unit": "tbsp", "calories": 190, "protein_g": 7, "carbs_g": 8, "fat_g": 16},
    {"name": "Hummus", "serving_size": 2, "serving_unit": "tbsp", "calories": 70, "protein_g": 2, "carbs_g": 5, "fat_g": 5},
    {"name": "Pizza, cheese slice", "serving_size": 1, "serving_unit": "slice", "calories": 285, "protein_g": 12, "carbs_g": 36, "fat_g": 10},
    {"name": "Burger, single patty", "serving_size": 1, "serving_unit": "item", "calories": 354, "protein_g": 17, "carbs_g": 29, "fat_g": 19},
    {"name": "French fries", "serving_size": 100, "serving_unit": "g", "calories": 312, "protein_g": 3.4, "carbs_g": 41, "fat_g": 15},
    {"name": "Chicken burrito", "serving_size": 1, "serving_unit": "item", "calories": 520, "protein_g": 29, "carbs_g": 49, "fat_g": 22},
    {"name": "Caesar salad with chicken", "serving_size": 1, "serving_unit": "bowl", "calories": 420, "protein_g": 27, "carbs_g": 15, "fat_g": 28},
    {"name": "Sushi roll, California", "serving_size": 1, "serving_unit": "roll", "calories": 255, "protein_g": 9, "carbs_g": 38, "fat_g": 7},
    {"name": "Sushi roll, spicy tuna", "serving_size": 1, "serving_unit": "roll", "calories": 290, "protein_g": 12, "carbs_g": 33, "fat_g": 10},
    {"name": "Ramen noodles", "serving_size": 1, "serving_unit": "bowl", "calories": 380, "protein_g": 10, "carbs_g": 52, "fat_g": 14},
    {"name": "Pho with beef", "serving_size": 1, "serving_unit": "bowl", "calories": 450, "protein_g": 26, "carbs_g": 55, "fat_g": 12},
    {"name": "Taco, beef", "serving_size": 1, "serving_unit": "item", "calories": 180, "protein_g": 8, "carbs_g": 14, "fat_g": 10},
    {"name": "Burrito bowl, chicken", "serving_size": 1, "serving_unit": "bowl", "calories": 610, "protein_g": 37, "carbs_g": 64, "fat_g": 22},
    {"name": "Quesadilla, cheese", "serving_size": 1, "serving_unit": "item", "calories": 510, "protein_g": 20, "carbs_g": 42, "fat_g": 28},
    {"name": "Fried chicken breast", "serving_size": 1, "serving_unit": "piece", "calories": 320, "protein_g": 29, "carbs_g": 12, "fat_g": 18},
    {"name": "Mashed potatoes", "serving_size": 1, "serving_unit": "cup", "calories": 214, "protein_g": 3.9, "carbs_g": 35, "fat_g": 7.2},
    {"name": "Mac and cheese", "serving_size": 1, "serving_unit": "cup", "calories": 310, "protein_g": 12, "carbs_g": 33, "fat_g": 14},
    {"name": "Protein bar", "serving_size": 1, "serving_unit": "bar", "calories": 210, "protein_g": 20, "carbs_g": 23, "fat_g": 7},
    {"name": "Granola", "serving_size": 60, "serving_unit": "g", "calories": 270, "protein_g": 6, "carbs_g": 40, "fat_g": 9},
    {"name": "Trail mix", "serving_size": 40, "serving_unit": "g", "calories": 200, "protein_g": 5, "carbs_g": 16, "fat_g": 14},
    {"name": "Pretzels", "serving_size": 28, "serving_unit": "g", "calories": 108, "protein_g": 2.8, "carbs_g": 22.5, "fat_g": 0.8},
    {"name": "Chips, potato", "serving_size": 28, "serving_unit": "g", "calories": 152, "protein_g": 2, "carbs_g": 15, "fat_g": 10},
    {"name": "Cookie, chocolate chip", "serving_size": 1, "serving_unit": "large", "calories": 78, "protein_g": 1, "carbs_g": 11, "fat_g": 3.3},
    {"name": "Ice cream, vanilla", "serving_size": 1, "serving_unit": "cup", "calories": 273, "protein_g": 4.6, "carbs_g": 31, "fat_g": 14.5},
    {"name": "Coffee, black", "serving_size": 240, "serving_unit": "ml", "calories": 2, "protein_g": 0.3, "carbs_g": 0, "fat_g": 0, "sodium_mg": 5, "caffeine_mg": 95},
    {"name": "Coffee with milk", "serving_size": 240, "serving_unit": "ml", "calories": 35, "protein_g": 2, "carbs_g": 3, "fat_g": 1.5, "caffeine_mg": 80},
    {"name": "Tea, unsweetened", "serving_size": 240, "serving_unit": "ml", "calories": 2, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "caffeine_mg": 47},
    {"name": "Orange juice", "serving_size": 240, "serving_unit": "ml", "calories": 110, "protein_g": 2, "carbs_g": 26, "fat_g": 0.5},
    {"name": "Soda", "serving_size": 355, "serving_unit": "ml", "calories": 140, "protein_g": 0, "carbs_g": 39, "fat_g": 0, "caffeine_mg": 34},
    {"name": "Diet soda", "serving_size": 355, "serving_unit": "ml", "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "caffeine_mg": 46},
    {"name": "Sports drink", "serving_size": 591, "serving_unit": "ml", "calories": 140, "protein_g": 0, "carbs_g": 34, "fat_g": 0},
    {"name": "Energy drink", "serving_size": 473, "serving_unit": "ml", "calories": 210, "protein_g": 0, "carbs_g": 54, "fat_g": 0, "caffeine_mg": 150},
    {"name": "Protein shake", "serving_size": 1, "serving_unit": "bottle", "calories": 160, "protein_g": 30, "carbs_g": 5, "fat_g": 3},
    {"name": "Smoothie, fruit", "serving_size": 350, "serving_unit": "ml", "calories": 220, "protein_g": 4, "carbs_g": 48, "fat_g": 1},
    {"name": "Sparkling water", "serving_size": 355, "serving_unit": "ml", "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
    {"name": "Beer, regular", "serving_size": 355, "serving_unit": "ml", "calories": 153, "protein_g": 1.6, "carbs_g": 13, "fat_g": 0},
    {"name": "Wine, red", "serving_size": 150, "serving_unit": "ml", "calories": 125, "protein_g": 0.1, "carbs_g": 4, "fat_g": 0},
    {"name": "Whiskey", "serving_size": 44, "serving_unit": "ml", "calories": 97, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
]

NUTRIENT_NUMBER_MAP = {
    "208": "calories",
    "203": "protein_g",
    "205": "carbs_g",
    "204": "fat_g",
    "269": "sugar_g",
    "307": "sodium_mg",
    "262": "caffeine_mg",
}


def safe_str(value, max_len: int):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def seed_common_foods_if_needed() -> None:
    global _COMMON_SEED_SYNCED
    if _COMMON_SEED_SYNCED:
        return

    changed = False
    for row in COMMON_FOODS:
        name = row["name"]
        brand = row.get("brand")
        existing = FoodItem.query.filter_by(name=name, brand=brand, source="seed").first()

        if existing:
            # Keep seeded defaults fresh in case values are tuned in code.
            fields_to_sync = {
                "serving_size": row.get("serving_size"),
                "serving_unit": row.get("serving_unit"),
                "calories": row.get("calories"),
                "protein_g": row.get("protein_g"),
                "carbs_g": row.get("carbs_g"),
                "fat_g": row.get("fat_g"),
                "sugar_g": row.get("sugar_g"),
                "sodium_mg": row.get("sodium_mg"),
                "caffeine_mg": row.get("caffeine_mg"),
            }
            existing_changed = False
            for field_name, field_value in fields_to_sync.items():
                if getattr(existing, field_name) != field_value:
                    setattr(existing, field_name, field_value)
                    existing_changed = True

            if existing_changed:
                db.session.add(existing)
                changed = True
            continue

        food = FoodItem(
            external_id=None,
            name=name,
            brand=brand,
            serving_size=row.get("serving_size"),
            serving_unit=row.get("serving_unit"),
            calories=row.get("calories"),
            protein_g=row.get("protein_g"),
            carbs_g=row.get("carbs_g"),
            fat_g=row.get("fat_g"),
            sugar_g=row.get("sugar_g"),
            sodium_mg=row.get("sodium_mg"),
            caffeine_mg=row.get("caffeine_mg"),
            source="seed",
        )
        db.session.add(food)
        changed = True

    if changed:
        db.session.commit()
    _COMMON_SEED_SYNCED = True


def parse_usda_nutrients(food_row: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}

    label_nutrients = food_row.get("labelNutrients") or {}
    if label_nutrients:
        calories = (((label_nutrients.get("calories") or {}).get("value")))
        carbs = (((label_nutrients.get("carbohydrates") or {}).get("value")))
        fat = (((label_nutrients.get("fat") or {}).get("value")))
        protein = (((label_nutrients.get("protein") or {}).get("value")))
        sugars = (((label_nutrients.get("sugars") or {}).get("value")))
        sodium = (((label_nutrients.get("sodium") or {}).get("value")))
        caffeine = (((label_nutrients.get("caffeine") or {}).get("value")))

        if calories is not None:
            parsed["calories"] = int(round(float(calories)))
        if carbs is not None:
            parsed["carbs_g"] = float(carbs)
        if fat is not None:
            parsed["fat_g"] = float(fat)
        if protein is not None:
            parsed["protein_g"] = float(protein)
        if sugars is not None:
            parsed["sugar_g"] = float(sugars)
        if sodium is not None:
            parsed["sodium_mg"] = float(sodium)
        if caffeine is not None:
            parsed["caffeine_mg"] = float(caffeine)

    for nutrient in (food_row.get("foodNutrients") or []):
        nutrient_number = str(
            nutrient.get("nutrientNumber")
            or nutrient.get("number")
            or ((nutrient.get("nutrient") or {}).get("number") or "")
        )
        field_name = NUTRIENT_NUMBER_MAP.get(nutrient_number)
        if not field_name:
            continue
        value = nutrient.get("value")
        if value is None:
            value = nutrient.get("amount")
        if value is None:
            continue

        if field_name == "calories":
            parsed[field_name] = int(round(float(value)))
        else:
            parsed[field_name] = float(value)

    return parsed


def import_foods_from_usda(query: str, max_results: int = 12) -> int:
    api_key = os.getenv("USDA_API_KEY") or os.getenv("FDC_API_KEY") or "DEMO_KEY"

    query = query.strip()
    if len(query) < 2:
        return 0

    endpoint = "https://api.nal.usda.gov/fdc/v1/foods/search"
    payload = {
        "query": query,
        "pageSize": max(1, min(max_results, 25)),
        "dataType": ["Foundation", "Survey (FNDDS)", "SR Legacy", "Branded"],
        "sortBy": "dataType.keyword",
        "sortOrder": "asc",
    }

    try:
        response = httpx.post(endpoint, params={"api_key": api_key}, json=payload, timeout=8.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return 0

    data = response.json()
    foods = data.get("foods") if isinstance(data, dict) else None
    if not foods:
        return 0

    imported = 0
    for row in foods:
        fdc_id = row.get("fdcId")
        if not fdc_id:
            continue

        external_id = safe_str(f"usda:{fdc_id}", 64)
        name = safe_str(row.get("description"), 255) or "Unnamed USDA item"
        brand = safe_str(row.get("brandOwner"), 255)
        existing = FoodItem.query.filter_by(external_id=external_id).first()
        if not existing:
            existing = FoodItem.query.filter_by(name=name, brand=brand, source="usda").first()
            if existing and not existing.external_id:
                existing.external_id = external_id

        nutrients = parse_usda_nutrients(row)
        if existing:
            # Refresh nutrient values as USDA updates records over time.
            existing.name = name or existing.name
            existing.brand = brand or existing.brand
            existing.serving_size = row.get("servingSize") or existing.serving_size
            existing.serving_unit = safe_str(row.get("servingSizeUnit"), 32) or existing.serving_unit
            existing.calories = nutrients.get("calories", existing.calories)
            existing.protein_g = nutrients.get("protein_g", existing.protein_g)
            existing.carbs_g = nutrients.get("carbs_g", existing.carbs_g)
            existing.fat_g = nutrients.get("fat_g", existing.fat_g)
            existing.sugar_g = nutrients.get("sugar_g", existing.sugar_g)
            existing.sodium_mg = nutrients.get("sodium_mg", existing.sodium_mg)
            existing.caffeine_mg = nutrients.get("caffeine_mg", existing.caffeine_mg)
            db.session.add(existing)
            continue

        food = FoodItem(
            external_id=external_id,
            name=name,
            brand=brand,
            serving_size=row.get("servingSize"),
            serving_unit=safe_str(row.get("servingSizeUnit"), 32),
            calories=nutrients.get("calories"),
            protein_g=nutrients.get("protein_g"),
            carbs_g=nutrients.get("carbs_g"),
            fat_g=nutrients.get("fat_g"),
            sugar_g=nutrients.get("sugar_g"),
            sodium_mg=nutrients.get("sodium_mg"),
            caffeine_mg=nutrients.get("caffeine_mg"),
            source="usda",
        )
        db.session.add(food)
        imported += 1

    db.session.commit()
    return imported
