from flask import Blueprint, request, g, current_app
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt,
)
from marshmallow import ValidationError
from app.extensions import db, limiter
from app.models.user import User, UserRole
from app.models.audit_log import AuditEventType
from app.schemas.auth_schema import RegisterSchema, LoginSchema
from app.services.auth_service import hash_password, verify_password, needs_rehash
from app.services.audit_service import log_event
from app.utils.responses import success_response, error_response

auth_bp = Blueprint("auth", __name__)
_register_schema = RegisterSchema()
_login_schema = LoginSchema()

_REVOKED_TOKENS: set = set()


def is_token_revoked(jwt_payload):
    return jwt_payload.get("jti") in _REVOKED_TOKENS


@auth_bp.post("/register")
@limiter.limit("3 per minute")
def register():
    try:
        data = _register_schema.load(request.get_json(force=True) or {})
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Input validation failed.", 422, details=e.messages)

    if User.query.filter_by(email=data["email"].lower()).first():
        log_event(AuditEventType.LOGIN_FAILURE, ip_address=request.remote_addr,
                  request_id=g.request_id, status="DUPLICATE_EMAIL")
        return error_response("EMAIL_TAKEN", "An account with this email already exists.", 409)

    user = User(
        name=data["name"].strip(),
        email=data["email"].lower().strip(),
        password_hash=hash_password(data["password"]),
        role=UserRole.USER,
    )
    db.session.add(user)
    db.session.commit()

    log_event(AuditEventType.LOGIN_SUCCESS, user_id=user.id,
              ip_address=request.remote_addr, request_id=g.request_id, status="REGISTERED")

    return success_response({"user": user.to_dict()}, status_code=201)


@auth_bp.post("/login")
@limiter.limit("5 per minute")
def login():
    try:
        data = _login_schema.load(request.get_json(force=True) or {})
    except ValidationError as e:
        return error_response("VALIDATION_ERROR", "Input validation failed.", 422, details=e.messages)

    user = User.query.filter_by(email=data["email"].lower().strip()).first()

    if not user or not verify_password(data["password"], user.password_hash):
        log_event(AuditEventType.LOGIN_FAILURE, ip_address=request.remote_addr,
                  request_id=g.request_id, status="INVALID_CREDENTIALS")
        return error_response("INVALID_CREDENTIALS", "Invalid email or password.", 401)

    if not user.is_active:
        return error_response("ACCOUNT_INACTIVE", "Account is deactivated.", 403)

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(data["password"])
        db.session.commit()

    additional_claims = {"role": user.role}
    access_token = create_access_token(identity=user.id, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=user.id, additional_claims=additional_claims)

    log_event(AuditEventType.LOGIN_SUCCESS, user_id=user.id,
              ip_address=request.remote_addr, request_id=g.request_id, status="SUCCESS")

    return success_response({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "user": user.to_dict(),
    })


@auth_bp.post("/refresh")
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    claims = get_jwt()
    jti = claims.get("jti")
    _REVOKED_TOKENS.add(jti)

    user = db.session.get(User, identity)
    if not user or not user.is_active:
        return error_response("ACCOUNT_INACTIVE", "Account is deactivated.", 403)

    additional_claims = {"role": user.role}
    new_access = create_access_token(identity=identity, additional_claims=additional_claims)
    new_refresh = create_refresh_token(identity=identity, additional_claims=additional_claims)

    return success_response({
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
    })


@auth_bp.post("/logout")
@jwt_required()
def logout():
    jti = get_jwt().get("jti")
    _REVOKED_TOKENS.add(jti)
    user_id = get_jwt_identity()
    log_event(AuditEventType.LOGOUT, user_id=user_id,
              ip_address=request.remote_addr, request_id=g.request_id, status="SUCCESS")
    return success_response({"message": "Successfully logged out."})
