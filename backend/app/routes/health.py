from flask import Blueprint
from app.extensions import db
from app.services import model_service
from app.utils.responses import success_response, error_response

health_bp = Blueprint("health", __name__)


@health_bp.get("")
def health():
    return success_response({"status": "ok"})


@health_bp.get("/ready")
def readiness():
    checks = {}
    all_ok = True

    try:
        db.session.execute(db.text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"
        all_ok = False

    checks["model"] = "ok" if model_service.is_model_loaded() else "unavailable"
    if not model_service.is_model_loaded():
        all_ok = False

    if all_ok:
        return success_response({"status": "ready", "checks": checks})
    return error_response("NOT_READY", "Service not ready.", 503)
