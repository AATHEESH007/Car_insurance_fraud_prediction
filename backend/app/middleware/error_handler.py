import logging
from flask import jsonify, g
from flask_jwt_extended.exceptions import (
    NoAuthorizationError,
    InvalidHeaderError,
    JWTDecodeError,
    RevokedTokenError,
    WrongTokenError,
)
from marshmallow import ValidationError as MarshmallowValidationError
from werkzeug.exceptions import RequestEntityTooLarge

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return _err("BAD_REQUEST", str(e), 400)

    @app.errorhandler(401)
    def unauthorized(e):
        return _err("UNAUTHORIZED", "Authentication required.", 401)

    @app.errorhandler(403)
    def forbidden(e):
        return _err("FORBIDDEN", "Insufficient permissions.", 403)

    @app.errorhandler(404)
    def not_found(e):
        return _err("NOT_FOUND", "Resource not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return _err("METHOD_NOT_ALLOWED", "Method not allowed.", 405)

    @app.errorhandler(409)
    def conflict(e):
        return _err("CONFLICT", str(e), 409)

    @app.errorhandler(413)
    def too_large(e):
        return _err("FILE_TOO_LARGE", "File size exceeds the 10 MB limit.", 413)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(e):
        return _err("FILE_TOO_LARGE", "File size exceeds the 10 MB limit.", 413)

    @app.errorhandler(415)
    def unsupported_media(e):
        return _err("UNSUPPORTED_MEDIA_TYPE", "Unsupported media type.", 415)

    @app.errorhandler(422)
    def unprocessable(e):
        return _err("UNPROCESSABLE_ENTITY", str(e), 422)

    @app.errorhandler(429)
    def rate_limit(e):
        return _err("RATE_LIMIT_EXCEEDED", "Too many requests. Please try again later.", 429)

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error: %s", e, exc_info=True)
        return _err("INTERNAL_SERVER_ERROR", "An unexpected error occurred.", 500)

    @app.errorhandler(503)
    def service_unavailable(e):
        return _err("SERVICE_UNAVAILABLE", "Service temporarily unavailable.", 503)

    @app.errorhandler(NoAuthorizationError)
    @app.errorhandler(InvalidHeaderError)
    def handle_no_auth(e):
        return _err("UNAUTHORIZED", "Authentication required.", 401)

    @app.errorhandler(JWTDecodeError)
    @app.errorhandler(WrongTokenError)
    def handle_invalid_token(e):
        return _err("INVALID_TOKEN", "Invalid or malformed token.", 401)

    @app.errorhandler(RevokedTokenError)
    def handle_revoked(e):
        return _err("TOKEN_REVOKED", "Token has been revoked.", 401)

    @app.errorhandler(MarshmallowValidationError)
    def handle_validation(e):
        return _err("VALIDATION_ERROR", "Input validation failed.", 422, details=e.messages)


def _err(code, message, status, details=None):
    body = {"success": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    resp = jsonify(body)
    resp.status_code = status
    request_id = getattr(g, "request_id", None)
    if request_id:
        resp.headers["X-Request-ID"] = request_id
    return resp
