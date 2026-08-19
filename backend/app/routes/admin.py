from flask import Blueprint, request, g, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import ValidationError
from app.extensions import db, limiter
from app.models.user import User, UserRole
from app.models.claim import Claim, ClaimStatus
from app.models.audit_log import AuditLog, AuditEventType
from app.schemas.claim_schema import AdminStatusUpdateSchema
from app.services import audit_service, claim_service
from app.utils.security import require_role
from app.utils.responses import success_response, error_response

admin_bp = Blueprint("admin", __name__)
_status_schema = AdminStatusUpdateSchema()


@admin_bp.get("/users")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def list_users():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    pagination = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return success_response({
        "users": [u.to_dict() for u in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
    })


@admin_bp.get("/claims")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def list_all_claims():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    risk = request.args.get("risk_level")
    status = request.args.get("status")

    query = Claim.query
    if risk:
        query = query.filter_by(risk_level=risk.upper())
    if status:
        query = query.filter_by(status=status.upper())

    pagination = query.order_by(Claim.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return success_response({
        "claims": [c.to_dict() for c in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
    })


@admin_bp.get("/claims/<claim_id>")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def get_claim(claim_id: str):
    claim = db.session.get(Claim, claim_id)
    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)
    return success_response({"claim": claim.to_dict()})


@admin_bp.patch("/claims/<claim_id>/status")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def update_claim_status(claim_id: str):
    admin_id = get_jwt_identity()

    try:
        data = _status_schema.load(request.get_json(force=True) or {})
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Input validation failed.", 422, details=e.messages)

    claim = db.session.get(Claim, claim_id)
    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)

    new_status = data["status"]
    if new_status not in ClaimStatus.ALL:
        return error_response("INVALID_STATUS", "Invalid claim status.", 422)

    claim_service.update_claim_status(claim, new_status)

    audit_service.log_event(
        AuditEventType.CLAIM_STATUS_UPDATED, user_id=admin_id,
        resource_id=claim_id, ip_address=request.remote_addr,
        request_id=g.request_id, status=new_status
    )
    audit_service.log_event(
        AuditEventType.ADMIN_ACTION, user_id=admin_id,
        resource_id=claim_id, ip_address=request.remote_addr,
        request_id=g.request_id, status=f"STATUS_UPDATED_TO_{new_status}"
    )

    return success_response({"claim": claim.to_dict()})


@admin_bp.get("/audit-logs")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def list_audit_logs():
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 50, type=int), 200)
    event_type = request.args.get("event_type")
    user_id = request.args.get("user_id")

    query = AuditLog.query
    if event_type:
        query = query.filter_by(event_type=event_type.upper())
    if user_id:
        query = query.filter_by(user_id=user_id)

    pagination = query.order_by(AuditLog.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return success_response({
        "logs": [log.to_dict() for log in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
    })


def _resolve_image_path(raw_path: str, config) -> str:
    import os
    if not raw_path:
        return None
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    upload_dir = config.get("UPLOAD_FOLDER", "uploads")
    filename = os.path.basename(raw_path)
    candidates = [
        raw_path,
        os.path.join(upload_dir, filename),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads", filename),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", filename),
        os.path.join("backend", "uploads", filename),
        os.path.join("uploads", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


@admin_bp.post("/claims/<claim_id>/analyze")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def analyze_claim(claim_id: str):
    from app.services import image_service, model_service
    from PIL import Image

    claim = db.session.get(Claim, claim_id)
    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)

    image_path = _resolve_image_path(claim.image_path, current_app.config)
    if not image_path:
        return error_response("NO_IMAGE", "No image available for this claim.", 422)

    if not model_service.is_model_loaded():
        return error_response("MODEL_UNAVAILABLE", "Prediction service is temporarily unavailable.", 503)

    try:
        pil_image = image_service.open_image_for_inference(image_path)
        result = model_service.predict(pil_image, current_app.config)
    except Exception as e:
        current_app.logger.error("Inference error on claim %s: %s", claim_id, e)
        return error_response("PREDICTION_FAILED", "Prediction service is temporarily unavailable.", 503)

    from app.services import claim_service as cs
    cs.attach_prediction(claim, result)

    admin_id = get_jwt_identity()
    audit_service.log_event(
        AuditEventType.PREDICTION_COMPLETED, user_id=admin_id,
        resource_id=claim_id, ip_address=request.remote_addr,
        request_id=g.request_id, status="SUCCESS"
    )

    return success_response({"prediction": result, "claim": claim.to_dict()})


@admin_bp.get("/statistics")
@require_role(UserRole.ADMIN)
@limiter.limit("100 per minute")
def statistics():
    total_users = User.query.count()
    total_claims = Claim.query.count()
    high_risk = Claim.query.filter_by(risk_level="HIGH").count()
    medium_risk = Claim.query.filter_by(risk_level="MEDIUM").count()
    low_risk = Claim.query.filter_by(risk_level="LOW").count()
    fraud_predictions = Claim.query.filter_by(prediction="Fraud").count()
    non_fraud_predictions = Claim.query.filter_by(prediction="Non-Fraud").count()

    status_counts = {}
    from app.models.claim import ClaimStatus
    for s in ClaimStatus.ALL:
        status_counts[s] = Claim.query.filter_by(status=s).count()

    return success_response({
        "users": {"total": total_users},
        "claims": {
            "total": total_claims,
            "by_risk": {"HIGH": high_risk, "MEDIUM": medium_risk, "LOW": low_risk},
            "by_prediction": {"Fraud": fraud_predictions, "Non-Fraud": non_fraud_predictions},
            "by_status": status_counts,
        },
    })


@admin_bp.get("/claims/<claim_id>/gradcam")
@require_role(UserRole.ADMIN)
@limiter.limit("30 per minute")
def get_gradcam(claim_id: str):
    """
    Generate (or return cached) Grad-CAM heatmap for a claim image.

    Returns both the original image URL and the heatmap overlay URL so the
    frontend can display them side-by-side.
    """
    import os
    from app.services import gradcam_service

    claim = db.session.get(Claim, claim_id)
    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)

    image_path = _resolve_image_path(claim.image_path, current_app.config)
    if not image_path:
        return error_response("NO_IMAGE", "No image available for this claim.", 422)

    upload_dir = current_app.config.get("UPLOAD_FOLDER", "uploads")
    if not os.path.isabs(upload_dir):
        upload_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), upload_dir
        )

    # Return cached heatmap if it already exists (avoid recomputation)
    cached_path = os.path.join(upload_dir, "gradcam", f"{claim_id}.png")
    if os.path.exists(cached_path):
        gradcam_url = f"/api/v1/uploads/gradcam/{claim_id}.png"
    else:
        from app.services.model_service import is_model_loaded
        if not is_model_loaded():
            return error_response(
                "MODEL_UNAVAILABLE",
                "Prediction model is not loaded. Cannot generate Grad-CAM.",
                503,
            )
        try:
            gradcam_url = gradcam_service.generate_gradcam(
                image_path=image_path,
                claim_id=claim_id,
                upload_dir=upload_dir,
            )
        except Exception as exc:
            current_app.logger.error(
                "GradCAM generation failed for claim %s: %s", claim_id, exc
            )
            return error_response(
                "GRADCAM_FAILED",
                "Failed to generate Grad-CAM heatmap.",
                503,
            )

    # Build original image URL
    original_filename = os.path.basename(claim.image_path)
    original_url = f"/api/v1/uploads/{original_filename}"

    return success_response({
        "original_url": original_url,
        "gradcam_url": gradcam_url,
    })


@admin_bp.delete("/claims/<claim_id>/gradcam/cache")
@require_role(UserRole.ADMIN)
@limiter.limit("30 per minute")
def clear_gradcam_cache(claim_id: str):
    """Delete the cached GradCAM heatmap so the next request regenerates it."""
    import os

    claim = db.session.get(Claim, claim_id)
    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)

    upload_dir = current_app.config.get("UPLOAD_FOLDER", "uploads")
    if not os.path.isabs(upload_dir):
        upload_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), upload_dir
        )

    cached_path = os.path.join(upload_dir, "gradcam", f"{claim_id}.png")
    if os.path.exists(cached_path):
        os.remove(cached_path)

    return success_response({"message": "GradCAM cache cleared."})

