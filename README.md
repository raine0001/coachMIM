# CoachMIM

CoachMIM is a structured self-report system for longitudinal behavioral pattern detection.

Phase I (MVP) focuses on 30-day baseline capture:

- Daily check-ins (sleep, mood, focus, energy, stress, productivity, symptoms, digestion, workout, alcohol)
- Meal logging (time, composition, tags, notes)
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
- `/checkin` daily check-in form
- `/meal` meal logger
- `/substance` substance logger
- `/timeline` recent logs + pre-emptive prompts
- `/insights` weekly summary and reflection

## Data Model

- `User`
- `DailyCheckIn`
- `Meal`
- `Substance`

## Render Deployment

`render.yaml` provisions:

- One web service (`coachmim-web`)
- One PostgreSQL database (`coachmim-db`)
- `DATABASE_URL` wiring from managed Postgres
- Build migration step (`flask db upgrade`)

## Notes

- OpenAI is optional for MVP. If `OPENAI_API_KEY` is unset, weekly AI reflection shows a fallback message.
- Uploads are stored in `app/static/uploads` for local/dev. Production object storage can replace this later.
