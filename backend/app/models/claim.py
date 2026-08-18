import uuid
from datetime import datetime, timezone
from app.extensions import db


class ClaimStatus:
    SUBMITTED = "SUBMITTED"
    ANALYZED = "ANALYZED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"

    ALL = [SUBMITTED, ANALYZED, UNDER_REVIEW, APPROVED, REJECTED, ESCALATED]


class Claim(db.Model):
    __tablename__ = "claims"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    claim_reference = db.Column(db.String(100), unique=True, nullable=False, index=True)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    vehicle_number = db.Column(db.String(50), nullable=False)
    vehicle_model = db.Column(db.String(100), nullable=False)
    vehicle_year = db.Column(db.Integer, nullable=False)
    claim_amount = db.Column(db.Numeric(12, 2), nullable=False)
    incident_date = db.Column(db.Date, nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(512), nullable=True)

    prediction = db.Column(db.String(20), nullable=True)
    fraud_probability = db.Column(db.Float, nullable=True)
    non_fraud_probability = db.Column(db.Float, nullable=True)
    risk_level = db.Column(db.String(10), nullable=True)
    recommendation = db.Column(db.String(255), nullable=True)

    status = db.Column(db.String(20), nullable=False, default=ClaimStatus.SUBMITTED)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False,
                           default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    user = db.relationship("User", back_populates="claims")

    def to_dict(self, include_image_path=False):
        import os
        image_url = None
        if self.image_path:
            filename = os.path.basename(self.image_path)
            image_url = f"/api/v1/uploads/{filename}"
        data = {
            "id": self.id,
            "claim_reference": self.claim_reference,
            "user_id": self.user_id,
            "vehicle_number": self.vehicle_number,
            "vehicle_model": self.vehicle_model,
            "vehicle_year": self.vehicle_year,
            "claim_amount": float(self.claim_amount) if self.claim_amount is not None else None,
            "incident_date": self.incident_date.isoformat() if self.incident_date else None,
            "description": self.description,
            "image_url": image_url,
            "prediction": self.prediction,
            "fraud_probability": self.fraud_probability,
            "non_fraud_probability": self.non_fraud_probability,
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        return data
