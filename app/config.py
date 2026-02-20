import os
from datetime import timedelta


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = normalize_database_url(
        os.getenv("DATABASE_URL", "sqlite:///dev.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEFAULT_UPLOAD_FOLDER = "/var/data/uploads" if os.path.isdir("/var/data") else "app/static/uploads"
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", DEFAULT_UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "coachmim_session")
    PERMANENT_SESSION_LIFETIME = timedelta(
        hours=int(os.getenv("SESSION_LIFETIME_HOURS", "24"))
    )

    ENCRYPTION_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY")
    ENCRYPTION_REQUIRED = os.getenv("ENCRYPTION_REQUIRED", "false").lower() == "true"
    LOGIN_EMAIL_CODE_ENABLED = (
        os.getenv("LOGIN_EMAIL_CODE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    LOGIN_EMAIL_CODE_TTL_MINUTES = int(os.getenv("LOGIN_EMAIL_CODE_TTL_MINUTES", "10"))
    LOGIN_EMAIL_CODE_RESEND_SECONDS = int(os.getenv("LOGIN_EMAIL_CODE_RESEND_SECONDS", "30"))

    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
    SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").strip().lower() in {"1", "true", "yes", "on"}
    MAIL_FROM = os.getenv("MAIL_FROM", "no-reply@coachmim.com")

    GOOGLE_OAUTH_CLIENT_ID = (
        os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        or os.getenv("GOOGLE_CLIENT_ID")
        or os.getenv("google_oauth_client_id")
    )
    GOOGLE_OAUTH_SECRET = (
        os.getenv("GOOGLE_OAUTH_SECRET")
        or os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        or os.getenv("GOOGLE_CLIENT_SECRET")
        or os.getenv("google_oauth_secret")
    )
    GOOGLE_OAUTH_REDIRECT_URI = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    PUSH_VAPID_PUBLIC_KEY = os.getenv("PUSH_VAPID_PUBLIC_KEY")
    PUSH_VAPID_PRIVATE_KEY = os.getenv("PUSH_VAPID_PRIVATE_KEY")
    PUSH_VAPID_CLAIMS_SUB = os.getenv("PUSH_VAPID_CLAIMS_SUB", "mailto:support@coachmim.com")
