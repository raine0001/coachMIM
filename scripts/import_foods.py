import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.food_catalog import import_foods_from_usda, seed_common_foods_if_needed


def main():
    parser = argparse.ArgumentParser(
        description="Import foods into local DB from USDA FoodData Central search results."
    )
    parser.add_argument(
        "queries",
        nargs="+",
        help="Search query terms to import (example: chicken rice yogurt coffee).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=25,
        help="Max USDA results per query (default: 25).",
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        seed_common_foods_if_needed()
        total = 0
        for query in args.queries:
            imported = import_foods_from_usda(query, max_results=args.max_results)
            total += imported
            print(f"{query}: imported {imported}")
        print(f"Total imported: {total}")


if __name__ == "__main__":
    main()
