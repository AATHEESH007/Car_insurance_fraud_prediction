from marshmallow import Schema, fields, validate, validates, ValidationError
from app.utils.validators import (
    validate_vehicle_year,
    validate_claim_amount,
    validate_incident_date,
    validate_vehicle_number,
)


class ClaimCreateSchema(Schema):
    claim_reference = fields.String(required=True, validate=validate.Length(min=3, max=100))
    vehicle_number = fields.String(required=True, validate=validate.Length(min=2, max=50))
    vehicle_model = fields.String(required=True, validate=validate.Length(min=1, max=100))
    vehicle_year = fields.Integer(required=True)
    claim_amount = fields.Float(required=True)
    incident_date = fields.String(required=True)
    description = fields.String(required=True, validate=validate.Length(min=3, max=5000))

    @validates("vehicle_number")
    def check_vehicle_number(self, value, **kwargs):
        ok, msg = validate_vehicle_number(value)
        if not ok:
            raise ValidationError(msg)

    @validates("vehicle_year")
    def check_year(self, value, **kwargs):
        ok, msg = validate_vehicle_year(value)
        if not ok:
            raise ValidationError(msg)

    @validates("claim_amount")
    def check_amount(self, value, **kwargs):
        ok, msg = validate_claim_amount(value)
        if not ok:
            raise ValidationError(msg)

    @validates("incident_date")
    def check_date(self, value, **kwargs):
        ok, msg = validate_incident_date(value)
        if not ok:
            raise ValidationError(msg)


class AdminStatusUpdateSchema(Schema):
    status = fields.String(required=True, validate=validate.OneOf(
        ["UNDER_REVIEW", "APPROVED", "REJECTED", "ESCALATED"]
    ))
    note = fields.String(load_default=None, validate=validate.Length(max=500))
