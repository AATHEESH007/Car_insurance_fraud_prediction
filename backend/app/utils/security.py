import uuid
from functools import wraps
from flask import request, g
from flask_jwt_extended import get_jwt, verify_jwt_in_request
from app.utils.responses import error_response


def generate_request_id() -> str:
    return str(uuid.uuid4())


def require_role(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
            except Exception:
                return error_response("UNAUTHORIZED", "Authentication required.", 401)
            claims = get_jwt()
            if claims.get("role") not in roles:
                _log_unauthorized_access()
                return error_response("FORBIDDEN", "Insufficient permissions.", 403)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def _log_unauthorized_access():
    try:
        from app.services.audit_service import log_event
        from app.models.audit_log import AuditEventType
        log_event(
            event_type=AuditEventType.UNAUTHORIZED_ACCESS,
            ip_address=request.remote_addr,
            request_id=getattr(g, "request_id", None),
            status="DENIED",
        )
    except Exception:
        pass
