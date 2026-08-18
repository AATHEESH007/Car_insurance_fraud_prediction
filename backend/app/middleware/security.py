import uuid
import logging
from flask import request, g

logger = logging.getLogger(__name__)


def register_security_middleware(app):
    @app.before_request
    def assign_request_id():
        g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    @app.after_request
    def apply_security_headers(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none';"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        return response
