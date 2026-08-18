from flask import Blueprint, request, g, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db, limiter
from app.models.audit_log import AuditEventType
from app.services import audit_service, image_service, model_service, claim_service
from app.models.claim import Claim
from app.utils.responses import success_response, error_response

predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.post("")
@jwt_required()
@limiter.limit("20 per minute")
def predict():
    user_id = get_jwt_identity()

    if "vehicle_image" not in request.files:
        return error_response("MISSING_IMAGE", "vehicle_image field is required.", 422)

    file = request.files["vehicle_image"]
    if not file or not file.filename:
        return error_response("MISSING_IMAGE", "No image file provided.", 422)

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    try:
        image_path = image_service.validate_and_save_image(file, upload_dir)
    except ValueError as e:
        return error_response("INVALID_IMAGE", str(e), 422)

    if not model_service.is_model_loaded():
        return error_response("MODEL_UNAVAILABLE", "Prediction service is temporarily unavailable.", 503)

    audit_service.log_event(
        AuditEventType.PREDICTION_REQUESTED, user_id=user_id,
        ip_address=request.remote_addr, request_id=g.request_id, status="STARTED"
    )

    try:
        pil_image = image_service.open_image_for_inference(image_path)
        result = model_service.predict(pil_image, current_app.config)
    except Exception as e:
        current_app.logger.error("Inference error: %s", e)
        return error_response("PREDICTION_FAILED", "Prediction service is temporarily unavailable.", 503)

    audit_service.log_event(
        AuditEventType.PREDICTION_COMPLETED, user_id=user_id,
        ip_address=request.remote_addr, request_id=g.request_id, status="SUCCESS"
    )

    claim_id = request.form.get("claim_id")
    if claim_id:
        claim = db.session.get(Claim, claim_id)
        if claim and claim.user_id == user_id:
            claim_service.attach_prediction(claim, result)

    return success_response({"prediction": result})
