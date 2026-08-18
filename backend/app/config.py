import os
from datetime import timedelta


class Config:
    SECRET_KEY = os.environ["SECRET_KEY"]
    JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRES_MINUTES", "15")))
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRES_DAYS", "30")))
    JWT_BLACKLIST_ENABLED = True
    JWT_BLACKLIST_TOKEN_CHECKS = ["access", "refresh"]

    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 300}

    ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]

    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))

    FRAUD_MEDIUM_THRESHOLD = float(os.environ.get("FRAUD_MEDIUM_THRESHOLD", "0.40"))
    FRAUD_HIGH_THRESHOLD = float(os.environ.get("FRAUD_HIGH_THRESHOLD", "0.70"))

    MODEL_PATH = os.environ.get("MODEL_PATH", "model/weights/best_efficientnetv2_s.pth")

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")

    RATELIMIT_STORAGE_URL = os.environ.get("REDIS_URL", "memory://")
    RATELIMIT_DEFAULT = "100 per minute"

    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
    SECRET_KEY = "test-secret-key-do-not-use-in-prod"
    JWT_SECRET_KEY = "test-jwt-secret-do-not-use-in-prod"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    WTF_CSRF_ENABLED = False
    FRAUD_MEDIUM_THRESHOLD = 0.40
    FRAUD_HIGH_THRESHOLD = 0.70
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    pass


config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config():
    env = os.environ.get("FLASK_ENV", "production")
    return config_map.get(env, ProductionConfig)
