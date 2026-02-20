import os


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


# Force small-memory runtime defaults for Render Starter.
# This file is auto-loaded by gunicorn when present in project root.
requested_workers = _as_int("GUNICORN_WORKERS", _as_int("WEB_CONCURRENCY", 1))
requested_threads = _as_int("GUNICORN_THREADS", 2)
workers = max(1, min(requested_workers, 2))
threads = max(1, min(requested_threads, 4))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")

timeout = _as_int("GUNICORN_TIMEOUT", 120)
graceful_timeout = _as_int("GUNICORN_GRACEFUL_TIMEOUT", 30)
keepalive = _as_int("GUNICORN_KEEPALIVE", 5)

max_requests = _as_int("GUNICORN_MAX_REQUESTS", 80)
max_requests_jitter = _as_int("GUNICORN_MAX_REQUESTS_JITTER", 20)

preload_app = os.getenv("GUNICORN_PRELOAD", "false").lower() == "true"

# Stream logs to platform collector.
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
