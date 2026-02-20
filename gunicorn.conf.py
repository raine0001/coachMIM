import os

# Force small-memory runtime defaults for Render Starter.
# This file is auto-loaded by gunicorn when present in project root.
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")

timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "80"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "20"))

preload_app = os.getenv("GUNICORN_PRELOAD", "false").lower() == "true"

# Stream logs to platform collector.
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
