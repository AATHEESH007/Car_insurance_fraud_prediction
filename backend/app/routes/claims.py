import os
from flask import Blueprint, request, g, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import ValidationError
from app.extensions import db, limiter
from app.models.claim import Claim
from app.models.audit_log import AuditEventType
from app.schemas.claim_schema import ClaimCreateSchema
from app.services import audit_service, claim_service, image_service
from app.utils.responses import success_response, error_response

claims_bp = Blueprint("claims", __name__)
_claim_schema = ClaimCreateSchema()


@claims_bp.post("")
@jwt_required()
@limiter.limit("100 per minute")
def create_claim():
    user_id = get_jwt_identity()

    try:
        form_data = {k: v for k, v in request.form.items()}
        data = _claim_schema.load(form_data)
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Input validation failed.", 422, details=e.messages)

    if Claim.query.filter_by(claim_reference=data["claim_reference"]).first():
        return error_response("DUPLICATE_REFERENCE", "Claim reference already exists.", 409)

    image_path = None
    if "vehicle_image" in request.files:
        file = request.files["vehicle_image"]
        if file and file.filename:
            upload_dir = current_app.config["UPLOAD_FOLDER"]
            try:
                image_path = image_service.validate_and_save_image(file, upload_dir)
            except ValueError as e:
                return error_response("INVALID_IMAGE", str(e), 422)

            audit_service.log_event(
                AuditEventType.IMAGE_UPLOADED, user_id=user_id,
                ip_address=request.remote_addr, request_id=g.request_id, status="SUCCESS"
            )

    claim = claim_service.create_claim(user_id, data, image_path)

    audit_service.log_event(
        AuditEventType.CLAIM_CREATED, user_id=user_id,
        resource_id=claim.id, ip_address=request.remote_addr,
        request_id=g.request_id, status="SUCCESS"
    )

    return success_response({"claim": claim.to_dict()}, status_code=201)


@claims_bp.get("")
@jwt_required()
@limiter.limit("100 per minute")
def list_claims():
    user_id = get_jwt_identity()
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)

    pagination = (
        Claim.query
        .filter_by(user_id=user_id)
        .order_by(Claim.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return success_response({
        "claims": [c.to_dict() for c in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
        "per_page": pagination.per_page,
    })


@claims_bp.get("/<claim_id>")
@jwt_required()
@limiter.limit("100 per minute")
def get_claim(claim_id: str):
    user_id = get_jwt_identity()
    claim = db.session.get(Claim, claim_id)

    if not claim:
        return error_response("NOT_FOUND", "Claim not found.", 404)

    if claim.user_id != user_id:
        audit_service.log_event(
            AuditEventType.UNAUTHORIZED_ACCESS, user_id=user_id,
            resource_id=claim_id, ip_address=request.remote_addr,
            request_id=g.request_id, status="DENIED"
        )
        return error_response("FORBIDDEN", "Access denied.", 403)

    audit_service.log_event(
        AuditEventType.CLAIM_VIEWED, user_id=user_id,
        resource_id=claim_id, ip_address=request.remote_addr,
        request_id=g.request_id, status="SUCCESS"
    )

    return success_response({"claim": claim.to_dict()})
