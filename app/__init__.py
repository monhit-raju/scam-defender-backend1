import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from app.services.anomaly_service import AnomalyService
from app.services.cloud_verification import CloudVerificationService
from app.services.gating_policy import ConfidenceThresholds
from app.services.mongo_store import MongoStore
from app.services.model_service import ModelService

load_dotenv()

jwt = JWTManager()


class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent

    SECRET_KEY = os.getenv("SECRET_KEY", "4c1f4cb047eb93d72412a37a5b592d8427c2d2e86170f222528d953da60d16fb")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "charity")

    MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))
    FRAUD_ISO_ALERT_THRESHOLD = float(os.getenv("FRAUD_ISO_ALERT_THRESHOLD", "-0.02"))

    POLICY_ALLOW_THRESHOLD = float(os.getenv("POLICY_ALLOW_THRESHOLD", "0.90"))
    POLICY_UNCERTAIN_THRESHOLD = float(os.getenv("POLICY_UNCERTAIN_THRESHOLD", "0.70"))
    POLICY_BLOCK_THRESHOLD = float(os.getenv("POLICY_BLOCK_THRESHOLD", "0.95"))

    CLOUD_VERIFICATION_URL = os.getenv("CLOUD_VERIFICATION_URL", "")
    CLOUD_VERIFICATION_API_KEY = os.getenv("CLOUD_VERIFICATION_API_KEY", "")
    CLOUD_VERIFICATION_TIMEOUT_SECONDS = int(os.getenv("CLOUD_VERIFICATION_TIMEOUT_SECONDS", "8"))
    CLOUD_MAX_INLINE_FILE_BYTES = int(os.getenv("CLOUD_MAX_INLINE_FILE_BYTES", str(5 * 1024 * 1024)))

    # Integration security
    INTEGRATION_SHARED_TOKEN = os.getenv("INTEGRATION_SHARED_TOKEN", "")
    BROWSER_EXTENSION_SHARED_TOKEN = os.getenv("BROWSER_EXTENSION_SHARED_TOKEN", INTEGRATION_SHARED_TOKEN)
    FILE_WATCHER_SHARED_TOKEN = os.getenv("FILE_WATCHER_SHARED_TOKEN", INTEGRATION_SHARED_TOKEN)
    ANDROID_INGEST_TOKEN = os.getenv("ANDROID_INGEST_TOKEN", INTEGRATION_SHARED_TOKEN)

    # WhatsApp Cloud API
    WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_FETCH_MEDIA = os.getenv("WHATSAPP_FETCH_MEDIA", "false").lower() in {"1", "true", "yes"}

    # Gmail / Google APIs
    GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID", "")
    GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "")
    GMAIL_REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN", "")
    GMAIL_USER_EMAIL = os.getenv("GMAIL_USER_EMAIL", "me")
    GMAIL_PUSH_WEBHOOK_TOKEN = os.getenv("GMAIL_PUSH_WEBHOOK_TOKEN", "")

    # Microsoft Graph / Outlook
    MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "")
    MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
    MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
    MICROSOFT_USER_ID = os.getenv("MICROSOFT_USER_ID", "")
    MICROSOFT_GRAPH_WEBHOOK_SECRET = os.getenv("MICROSOFT_GRAPH_WEBHOOK_SECRET", "")

    # IMAP polling connector
    IMAP_HOST = os.getenv("IMAP_HOST", "")
    IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
    IMAP_USERNAME = os.getenv("IMAP_USERNAME", "")
    IMAP_PASSWORD = os.getenv("IMAP_PASSWORD", "")
    IMAP_USE_SSL = os.getenv("IMAP_USE_SSL", "true").lower() in {"1", "true", "yes"}

    # Artifact download paths
    ANDROID_RELEASE_DIR = os.getenv("ANDROID_RELEASE_DIR", str(PROJECT_ROOT / "mobile" / "releases"))
    BROWSER_EXTENSION_DIR = os.getenv("BROWSER_EXTENSION_DIR", str(PROJECT_ROOT / "integrations" / "browser_extension"))

    # Operational hardening
    API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120"))
    API_UPLOAD_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_UPLOAD_RATE_LIMIT_PER_MINUTE", "40"))

    raw_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:4173,http://127.0.0.1:5173",
    )
    CORS_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(AppConfig)

    if config_overrides:
        app.config.update(config_overrides)

    jwt.init_app(app)

    CORS(
        app,
        resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}},
        supports_credentials=True,
    )

    app.extensions["model_service"] = ModelService(app.config["MODEL_DIR"])
    app.extensions["anomaly_service"] = AnomalyService(
        app.config["MODEL_DIR"],
        fraud_iso_threshold=app.config["FRAUD_ISO_ALERT_THRESHOLD"],
    )
    app.extensions["store"] = MongoStore(
        mongo_uri=app.config["MONGO_URI"],
        database_name=app.config["MONGO_DB_NAME"],
    )
    app.extensions["cloud_verifier"] = CloudVerificationService(
        endpoint_url=app.config["CLOUD_VERIFICATION_URL"],
        api_key=app.config["CLOUD_VERIFICATION_API_KEY"],
        timeout_seconds=app.config["CLOUD_VERIFICATION_TIMEOUT_SECONDS"],
    )
    app.extensions["confidence_thresholds"] = ConfidenceThresholds(
        allow_confidence=app.config["POLICY_ALLOW_THRESHOLD"],
        uncertain_confidence=app.config["POLICY_UNCERTAIN_THRESHOLD"],
        block_malicious_confidence=app.config["POLICY_BLOCK_THRESHOLD"],
    )

    from app.api import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    return app
