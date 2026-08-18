import uuid
from datetime import datetime, timezone
from app.extensions import db


class AuditEventType:
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    CLAIM_CREATED = "CLAIM_CREATED"
    IMAGE_UPLOADED = "IMAGE_UPLOADED"
    PREDICTION_REQUESTED = "PREDICTION_REQUESTED"
    PREDICTION_COMPLETED = "PREDICTION_COMPLETED"
    CLAIM_VIEWED = "CLAIM_VIEWED"
    CLAIM_STATUS_UPDATED = "CLAIM_STATUS_UPDATED"
    ADMIN_ACTION = "ADMIN_ACTION"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    RATE_LIMIT_TRIGGERED = "RATE_LIMIT_TRIGGERED"


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = db.Column(db.String(50), nullable=False, index=True)
    resource_id = db.Column(db.String(36), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    request_id = db.Column(db.String(36), nullable=True)
    status = db.Column(db.String(20), nullable=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)

    user = db.relationship("User", back_populates="audit_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "request_id": self.request_id,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
        }
