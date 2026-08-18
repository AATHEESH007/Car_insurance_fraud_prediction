import logging
from app.extensions import db
from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


def log_event(event_type: str, user_id=None, resource_id=None,
              ip_address=None, request_id=None, status=None):
    try:
        entry = AuditLog(
            user_id=user_id,
            event_type=event_type,
            resource_id=resource_id,
            ip_address=ip_address,
            request_id=request_id,
            status=status,
        )
        db.session.add(entry)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        logger.error("Failed to write audit log: %s", exc)
