import uuid
from datetime import date, datetime, timezone
from app.extensions import db
from app.models.claim import Claim, ClaimStatus


def create_claim(user_id: str, data: dict, image_path: str = None) -> Claim:
    claim = Claim(
        id=str(uuid.uuid4()),
        claim_reference=data["claim_reference"],
        user_id=user_id,
        vehicle_number=data["vehicle_number"].strip().upper(),
        vehicle_model=data["vehicle_model"].strip(),
        vehicle_year=int(data["vehicle_year"]),
        claim_amount=float(data["claim_amount"]),
        incident_date=date.fromisoformat(data["incident_date"]),
        description=data["description"].strip(),
        image_path=image_path,
        status=ClaimStatus.SUBMITTED,
    )
    db.session.add(claim)
    db.session.commit()
    return claim


def attach_prediction(claim: Claim, prediction_result: dict) -> Claim:
    claim.prediction = prediction_result["prediction"]
    claim.fraud_probability = prediction_result["fraud_probability"]
    claim.non_fraud_probability = prediction_result["non_fraud_probability"]
    claim.risk_level = prediction_result["risk_level"]
    claim.recommendation = prediction_result["recommendation"]
    claim.status = ClaimStatus.ANALYZED
    claim.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return claim


def update_claim_status(claim: Claim, new_status: str) -> Claim:
    claim.status = new_status
    claim.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return claim
