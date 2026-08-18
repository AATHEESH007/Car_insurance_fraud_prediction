from marshmallow import Schema, fields, validate, validates, ValidationError


class RegisterSchema(Schema):
    name = fields.String(required=True, validate=validate.Length(min=1, max=255))
    email = fields.String(required=True, validate=validate.Length(min=1, max=255))
    password = fields.String(required=True, load_only=True, validate=validate.Length(min=1))


class LoginSchema(Schema):
    email = fields.String(required=True)
    password = fields.String(required=True, load_only=True)


class RefreshSchema(Schema):
    pass
