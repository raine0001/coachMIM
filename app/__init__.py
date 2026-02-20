import os

from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from app.security import EncryptionConfigError, validate_encryption_configuration

db = SQLAlchemy()
migrate = Migrate()


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__)
    app.config.from_object("app.config.Config")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    try:
        validate_encryption_configuration(
            app.config.get("ENCRYPTION_MASTER_KEY"),
            bool(app.config.get("ENCRYPTION_REQUIRED")),
        )
    except EncryptionConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import bp

    app.register_blueprint(bp)

    # Ensure model metadata is registered for migrations.
    from app import models  # noqa: F401

    @app.after_request
    def apply_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if app.config.get("SESSION_COOKIE_SECURE"):
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    return app
