import os
import logging
from flask import Flask
from flask_swagger_ui import get_swaggerui_blueprint
from app.config import get_config
from app.extensions import db, migrate, jwt, cors, limiter
from app.middleware.security import register_security_middleware
from app.middleware.error_handler import register_error_handlers
from app.routes.auth import auth_bp, is_token_revoked
from app.routes.claims import claims_bp
from app.routes.predictions import predictions_bp
from app.routes.admin import admin_bp
from app.routes.health import health_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app(config_class=None):
    app = Flask(__name__, instance_relative_config=False)

    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    _init_extensions(app)
    register_security_middleware(app)
    register_error_handlers(app)
    _register_blueprints(app)
    _register_swagger(app)

    if not app.config.get("TESTING"):
        with app.app_context():
            _load_ml_model(app)

    return app


def _init_extensions(app):
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    limiter.init_app(app)

    cors.init_app(app, resources={
        r"/api/*": {
            "origins": app.config["ALLOWED_ORIGINS"],
            "methods": ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Request-ID"],
            "expose_headers": ["X-Request-ID"],
            "supports_credentials": True,
        }
    })

    from app.routes.auth import _REVOKED_TOKENS

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        return is_token_revoked(jwt_payload)

    @jwt.revoked_token_loader
    def revoked_token_response(jwt_header, jwt_payload):
        from app.utils.responses import error_response
        return error_response("TOKEN_REVOKED", "Token has been revoked.", 401)

    @jwt.expired_token_loader
    def expired_token_response(jwt_header, jwt_payload):
        from app.utils.responses import error_response
        return error_response("TOKEN_EXPIRED", "Token has expired.", 401)

    @jwt.invalid_token_loader
    def invalid_token_response(reason):
        from app.utils.responses import error_response
        return error_response("INVALID_TOKEN", "Invalid token.", 401)

    @jwt.unauthorized_loader
    def missing_token_response(reason):
        from app.utils.responses import error_response
        return error_response("UNAUTHORIZED", "Authentication required.", 401)


def _register_blueprints(app):
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth")
    app.register_blueprint(claims_bp, url_prefix="/api/v1/claims")
    app.register_blueprint(predictions_bp, url_prefix="/api/v1/predictions")
    app.register_blueprint(admin_bp, url_prefix="/api/v1/admin")
    app.register_blueprint(health_bp, url_prefix="/api/v1/health")

    import os
    from flask import send_from_directory
    from flask_jwt_extended import jwt_required

    @app.route("/api/v1/uploads/<path:filename>")
    def serve_upload(filename):
        upload_dir = os.path.abspath(app.config.get("UPLOAD_FOLDER", "uploads"))
        return send_from_directory(upload_dir, filename)


def _register_swagger(app):
    from app.swagger import get_openapi_spec
    import json

    @app.route("/api/v1/openapi.json")
    def openapi_spec():
        from flask import jsonify
        return jsonify(get_openapi_spec())

    swaggerui_bp = get_swaggerui_blueprint(
        "/api/v1/docs",
        "/api/v1/openapi.json",
        config={"app_name": "Vehicle Insurance Fraud Detection API"},
    )
    app.register_blueprint(swaggerui_bp, url_prefix="/api/v1/docs")


def _load_ml_model(app):
    from app.services import model_service
    model_path = app.config.get("MODEL_PATH", "model/weights/best_efficientnetv2_s.pth")
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
    if os.path.exists(model_path):
        try:
            model_service.load_model(model_path)
        except Exception as exc:
            app.logger.error("Failed to load ML model: %s", exc)
    else:
        app.logger.warning("Model weights not found at %s. Predictions will be unavailable.", model_path)
