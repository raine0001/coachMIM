import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app, db
from app.models import FoodItem

TARGET_NUTRIENTS = {
    "208": "calories",
    "203": "protein_g",
    "205": "carbs_g",
    "204": "fat_g",
    "269": "sugar_g",
    "307": "sodium_mg",
}


def safe_str(value, max_len: int):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def parse_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def find_required_csv(data_dir: Path, filename: str) -> Path:
    direct = data_dir / filename
    if direct.exists():
        return direct

    matches = list(data_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Missing required USDA CSV: {filename}")
    return matches[0]


def find_optional_csv(data_dir: Path, filename: str) -> Path | None:
    direct = data_dir / filename
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(filename))
    return matches[0] if matches else None


def load_measure_units(measure_unit_csv: Path) -> dict[int, str]:
    units = {}
    with measure_unit_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            unit_id = parse_int(row.get("id"))
            name = (row.get("name") or "").strip()
            if unit_id is not None and name:
                units[unit_id] = name
    return units


def load_target_nutrient_ids(nutrient_csv: Path) -> dict[int, str]:
    nutrient_ids = {}
    with nutrient_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nutrient_id = parse_int(row.get("id"))
            nutrient_number = (row.get("number") or "").strip()
            mapped = TARGET_NUTRIENTS.get(nutrient_number)
            if nutrient_id is not None and mapped:
                nutrient_ids[nutrient_id] = mapped
    return nutrient_ids


def should_include_food(description: str, data_type: str, include_branded: bool, keywords: list[str]) -> bool:
    if not description:
        return False

    if not include_branded and data_type.lower() == "branded":
        return False

    if not keywords:
        return True

    lowered = description.lower()
    return any(keyword in lowered for keyword in keywords)


def load_food_rows(
    food_csv: Path,
    include_branded: bool,
    keywords: list[str],
    max_foods: int | None,
) -> dict[int, dict]:
    selected = {}
    with food_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fdc_id = parse_int(row.get("fdc_id"))
            if fdc_id is None:
                continue

            description = safe_str(row.get("description"), 255) or ""
            data_type = (row.get("data_type") or "").strip()
            if not should_include_food(description, data_type, include_branded, keywords):
                continue

            selected[fdc_id] = {
                "fdc_id": fdc_id,
                "name": description,
                "data_type": data_type,
                "brand": None,
                "serving_size": None,
                "serving_unit": None,
                "calories": None,
                "protein_g": None,
                "carbs_g": None,
                "fat_g": None,
                "sugar_g": None,
                "sodium_mg": None,
            }

            if max_foods and len(selected) >= max_foods:
                break
    return selected


def load_branded_rows(branded_csv: Path | None, selected: dict[int, dict]) -> None:
    if not branded_csv:
        return

    with branded_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fdc_id = parse_int(row.get("fdc_id"))
            if fdc_id is None or fdc_id not in selected:
                continue

            selected_row = selected[fdc_id]
            brand = safe_str(row.get("brand_owner") or row.get("brand_name"), 255)
            serving_size = parse_float(row.get("serving_size"))
            serving_unit = safe_str(row.get("serving_size_unit"), 32)

            if brand:
                selected_row["brand"] = brand
            if serving_size is not None:
                selected_row["serving_size"] = serving_size
            if serving_unit:
                selected_row["serving_unit"] = serving_unit


def load_portion_rows(food_portion_csv: Path | None, measure_units: dict[int, str], selected: dict[int, dict]) -> None:
    if not food_portion_csv:
        return

    with food_portion_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fdc_id = parse_int(row.get("fdc_id"))
            if fdc_id is None or fdc_id not in selected:
                continue

            selected_row = selected[fdc_id]
            if selected_row["serving_size"] is not None and selected_row["serving_unit"] is not None:
                continue

            amount = parse_float(row.get("amount"))
            unit_id = parse_int(row.get("measure_unit_id"))
            modifier = safe_str(row.get("modifier"), 32)
            unit_name = measure_units.get(unit_id) if unit_id is not None else None

            if amount is not None and selected_row["serving_size"] is None:
                selected_row["serving_size"] = amount

            if selected_row["serving_unit"] is None:
                if unit_name:
                    selected_row["serving_unit"] = safe_str(unit_name, 32)
                elif modifier:
                    selected_row["serving_unit"] = modifier


def load_food_nutrients(food_nutrient_csv: Path, nutrient_ids: dict[int, str], selected: dict[int, dict]) -> None:
    tracked_ids = set(selected.keys())
    target_ids = set(nutrient_ids.keys())

    with food_nutrient_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fdc_id = parse_int(row.get("fdc_id"))
            if fdc_id is None or fdc_id not in tracked_ids:
                continue

            nutrient_id = parse_int(row.get("nutrient_id"))
            if nutrient_id is None or nutrient_id not in target_ids:
                continue

            field_name = nutrient_ids[nutrient_id]
            amount = parse_float(row.get("amount"))
            if amount is None:
                continue

            if field_name == "calories":
                selected[fdc_id][field_name] = int(round(amount))
            else:
                selected[fdc_id][field_name] = amount


def load_existing_usda_cache() -> tuple[dict[str, FoodItem], dict[tuple[str, str], FoodItem]]:
    by_external = {}
    by_key = {}

    existing = FoodItem.query.filter_by(source="usda").all()
    for item in existing:
        if item.external_id:
            by_external[item.external_id] = item
        key = ((item.name or "").strip().lower(), (item.brand or "").strip().lower())
        by_key[key] = item

    return by_external, by_key


def upsert_selected_foods(selected: dict[int, dict], batch_size: int = 1000) -> tuple[int, int]:
    by_external, by_key = load_existing_usda_cache()
    inserted = 0
    updated = 0
    processed = 0

    # Deduplicate dump rows by display key before upsert.
    deduped = {}
    for row in selected.values():
        key = ((row["name"] or "").strip().lower(), (row["brand"] or "").strip().lower())
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
            continue

        # Prefer rows with more nutrient coverage.
        coverage_existing = sum(existing.get(field) is not None for field in TARGET_NUTRIENTS.values())
        coverage_new = sum(row.get(field) is not None for field in TARGET_NUTRIENTS.values())
        if coverage_new > coverage_existing:
            deduped[key] = row

    for row in deduped.values():
        external_id = safe_str(f"usda:{row['fdc_id']}", 64)
        key = ((row["name"] or "").strip().lower(), (row["brand"] or "").strip().lower())

        item = by_external.get(external_id) or by_key.get(key)
        if item:
            item.external_id = item.external_id or external_id
            item.name = safe_str(row["name"], 255) or item.name
            item.brand = safe_str(row["brand"], 255) or item.brand
            item.serving_size = row["serving_size"] if row["serving_size"] is not None else item.serving_size
            item.serving_unit = safe_str(row["serving_unit"], 32) or item.serving_unit
            item.calories = row["calories"] if row["calories"] is not None else item.calories
            item.protein_g = row["protein_g"] if row["protein_g"] is not None else item.protein_g
            item.carbs_g = row["carbs_g"] if row["carbs_g"] is not None else item.carbs_g
            item.fat_g = row["fat_g"] if row["fat_g"] is not None else item.fat_g
            item.sugar_g = row["sugar_g"] if row["sugar_g"] is not None else item.sugar_g
            item.sodium_mg = row["sodium_mg"] if row["sodium_mg"] is not None else item.sodium_mg
            db.session.add(item)
            updated += 1
        else:
            item = FoodItem(
                external_id=external_id,
                name=safe_str(row["name"], 255) or "Unnamed USDA item",
                brand=safe_str(row["brand"], 255),
                serving_size=row["serving_size"],
                serving_unit=safe_str(row["serving_unit"], 32),
                calories=row["calories"],
                protein_g=row["protein_g"],
                carbs_g=row["carbs_g"],
                fat_g=row["fat_g"],
                sugar_g=row["sugar_g"],
                sodium_mg=row["sodium_mg"],
                source="usda",
            )
            db.session.add(item)
            inserted += 1

        by_external[external_id] = item
        by_key[key] = item
        processed += 1

        if processed % batch_size == 0:
            db.session.commit()

    db.session.commit()
    return inserted, updated


def run_import(
    data_dir: Path,
    include_branded: bool,
    keywords: list[str],
    max_foods: int | None,
    batch_size: int,
):
    food_csv = find_required_csv(data_dir, "food.csv")
    nutrient_csv = find_required_csv(data_dir, "nutrient.csv")
    food_nutrient_csv = find_required_csv(data_dir, "food_nutrient.csv")
    branded_csv = find_optional_csv(data_dir, "branded_food.csv")
    food_portion_csv = find_optional_csv(data_dir, "food_portion.csv")
    measure_unit_csv = find_optional_csv(data_dir, "measure_unit.csv")

    measure_units = load_measure_units(measure_unit_csv) if measure_unit_csv else {}
    nutrient_ids = load_target_nutrient_ids(nutrient_csv)

    selected = load_food_rows(
        food_csv=food_csv,
        include_branded=include_branded,
        keywords=keywords,
        max_foods=max_foods,
    )
    load_branded_rows(branded_csv, selected)
    load_portion_rows(food_portion_csv, measure_units, selected)
    load_food_nutrients(food_nutrient_csv, nutrient_ids, selected)

    inserted, updated = upsert_selected_foods(selected, batch_size=batch_size)
    print(f"Selected rows: {len(selected)}")
    print(f"Inserted: {inserted}")
    print(f"Updated: {updated}")


def main():
    parser = argparse.ArgumentParser(
        description="Import USDA FoodData Central CSV dump into FoodItem table (deduped)."
    )
    parser.add_argument(
        "data_dir",
        help="Path to extracted USDA CSV folder (contains food.csv, nutrient.csv, food_nutrient.csv, ...).",
    )
    parser.add_argument(
        "--include-branded",
        action="store_true",
        help="Include branded foods from USDA dump (can massively increase row count).",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Comma-separated keyword filter for description (example: chicken,rice,pancake).",
    )
    parser.add_argument(
        "--max-foods",
        type=int,
        default=50000,
        help="Max number of food rows selected from food.csv before nutrient join (default: 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="DB commit batch size (default: 1000).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Invalid data directory: {data_dir}")

    keywords = [item.strip().lower() for item in args.keywords.split(",") if item.strip()]

    app = create_app()
    with app.app_context():
        run_import(
            data_dir=data_dir,
            include_branded=args.include_branded,
            keywords=keywords,
            max_foods=args.max_foods if args.max_foods > 0 else None,
            batch_size=max(100, args.batch_size),
        )


if __name__ == "__main__":
    main()
