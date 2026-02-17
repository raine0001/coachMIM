# CoachMIM

CoachMIM is a structured self-report system for longitudinal behavioral pattern detection.

Phase I (MVP) focuses on 30-day baseline capture:

- Basic account login per user
- Core profile baseline (required fields) + optional profile enrichment
- Daily check-ins with timezone-aware day locking and segmented tabs (sleep, morning, midday, evening, overall)
- Prev/next day navigation and check-in history table for consistency tracking
- Meal logging with searchable food catalog + USDA-backed import/caching
- Ingredient-based meal builder (multi-item recipes) with reusable favorites
- Custom ingredient entry (manual nutrition facts) + optional nutrition-label photo parsing
- Product link parsing (fetch nutrition from product pages, with manual/USDA/photo fallback)
- Favorite meals (save/reuse defaults)
- Daily meal table with edit/delete
- Optional meal photo upload
- Pre-emptive prompts for missing or inconsistent data
- Lightweight weekly insights

## Stack

- Flask
- Flask-SQLAlchemy
- Flask-Migrate (Alembic)
- PostgreSQL (Render)
- OpenAI (optional reflection layer)

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
$env:FLASK_APP="wsgi.py"
flask db upgrade
flask run
```

Open http://127.0.0.1:5000

## Quickstart (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export FLASK_APP=wsgi.py
flask db upgrade
flask run
```

## Core Routes

- `/` dashboard
- `/register` create account
- `/login` login
- `/logout` logout
- `/profile` profile setup/update
- `/checkin` daily check-in form
- `/meal` meal logger
- `/meal/<id>/edit` update meal entry
- `/meal/<id>/delete` delete meal entry
- `/foods/search` autocomplete endpoint for local + USDA food catalog lookup
- `/nutrition/product/parse` parse nutrition from product URL
- `/nutrition/label/parse` parse nutrition from label photo
- `/substance` substance logger
- `/timeline` recent logs + pre-emptive prompts
- `/insights` weekly summary and reflection

## Data Model

- `User`
- `UserProfile`
- `DailyCheckIn`
- `Meal`
- `FoodItem`
- `FavoriteMeal`
- `Substance`

## Render Deployment

`render.yaml` provisions:

- One web service (`coachmim-web`)
- `DATABASE_URL` wiring from existing managed Postgres (`coachMIMdb`)
- Build migration step (`flask db upgrade`)

## Notes

- OpenAI is optional for MVP. If `OPENAI_API_KEY` is unset, weekly AI reflection shows a fallback message.
- Nutrition-label photo parsing also requires `OPENAI_API_KEY`.
- Product link parsing works with direct page extraction; `OPENAI_API_KEY` enables AI fallback parsing for harder pages.
- USDA food import is optional. Add `USDA_API_KEY` to enable search-time USDA imports into your local DB cache.
- If `USDA_API_KEY` is unset, app falls back to USDA `DEMO_KEY` (lower rate limits).
- Optional bulk import command: `python -m scripts.import_foods chicken rice yogurt coffee --max-results 50`
- Optional USDA CSV dump import (very large catalogs) from extracted folder:
  - Windows PowerShell example:
    - `.\.venv\Scripts\python.exe scripts\import_usda_dump.py "C:\Users\dave\Desktop\FoodData_Central_csv_2025-12-18\FoodData_Central_csv_2025-12-18" --max-foods 100000 --batch-size 2000`
  - Start without `--include-branded` to keep results cleaner and import size manageable.
  - If you want more coverage, run additional passes with higher `--max-foods` or targeted `--keywords`.
- Uploads are stored in `app/static/uploads` for local/dev. Production object storage can replace this later.
- For production cookie security, set `SESSION_COOKIE_SECURE=true`.
