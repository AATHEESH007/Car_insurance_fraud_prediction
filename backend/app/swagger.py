def get_openapi_spec():
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Vehicle Insurance Fraud Detection API",
            "version": "1.0.0",
            "description": (
                "AI-based vehicle insurance fraud detection backend. "
                "Uses EfficientNetV2-S to classify damage images as Fraud or Non-Fraud."
            ),
        },
        "servers": [{"url": "/api/v1"}],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean", "example": False},
                        "error": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
                "UserOut": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "role": {"type": "string", "enum": ["USER", "ADMIN"]},
                        "is_active": {"type": "boolean"},
                        "created_at": {"type": "string", "format": "date-time"},
                    },
                },
                "ClaimOut": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "claim_reference": {"type": "string"},
                        "vehicle_number": {"type": "string"},
                        "vehicle_model": {"type": "string"},
                        "vehicle_year": {"type": "integer"},
                        "claim_amount": {"type": "number"},
                        "incident_date": {"type": "string", "format": "date"},
                        "description": {"type": "string"},
                        "prediction": {"type": "string", "nullable": True},
                        "fraud_probability": {"type": "number", "nullable": True},
                        "non_fraud_probability": {"type": "number", "nullable": True},
                        "risk_level": {"type": "string", "nullable": True},
                        "recommendation": {"type": "string", "nullable": True},
                        "status": {"type": "string"},
                        "created_at": {"type": "string", "format": "date-time"},
                    },
                },
                "PredictionResult": {
                    "type": "object",
                    "properties": {
                        "prediction": {"type": "string", "enum": ["Fraud", "Non-Fraud"]},
                        "fraud_probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "non_fraud_probability": {"type": "number", "minimum": 0, "maximum": 1},
                        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                        "recommendation": {"type": "string"},
                    },
                },
            },
        },
        "paths": {
            "/auth/register": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Register a new user",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["name", "email", "password"],
                                    "properties": {
                                        "name": {"type": "string", "minLength": 2},
                                        "email": {"type": "string", "format": "email"},
                                        "password": {"type": "string", "minLength": 8},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "User registered successfully"},
                        "409": {"description": "Email already taken"},
                        "422": {"description": "Validation error"},
                    },
                }
            },
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Login and receive JWT tokens",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["email", "password"],
                                    "properties": {
                                        "email": {"type": "string", "format": "email"},
                                        "password": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Login successful, returns access and refresh tokens"},
                        "401": {"description": "Invalid credentials"},
                    },
                }
            },
            "/auth/refresh": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Refresh access token",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"description": "New access and refresh tokens"},
                        "401": {"description": "Invalid or expired refresh token"},
                    },
                }
            },
            "/auth/logout": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Logout and revoke token",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {"description": "Logged out successfully"},
                    },
                }
            },
            "/claims": {
                "post": {
                    "tags": ["Claims"],
                    "summary": "Submit a new insurance claim",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["claim_reference", "vehicle_number", "vehicle_model",
                                                 "vehicle_year", "claim_amount", "incident_date", "description"],
                                    "properties": {
                                        "claim_reference": {"type": "string"},
                                        "vehicle_number": {"type": "string"},
                                        "vehicle_model": {"type": "string"},
                                        "vehicle_year": {"type": "integer"},
                                        "claim_amount": {"type": "number"},
                                        "incident_date": {"type": "string", "format": "date"},
                                        "description": {"type": "string"},
                                        "vehicle_image": {"type": "string", "format": "binary"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {"description": "Claim created"},
                        "409": {"description": "Duplicate claim reference"},
                        "422": {"description": "Validation error"},
                    },
                },
                "get": {
                    "tags": ["Claims"],
                    "summary": "List own claims",
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                        {"name": "per_page", "in": "query", "schema": {"type": "integer", "default": 20}},
                    ],
                    "responses": {"200": {"description": "List of claims"}},
                },
            },
            "/claims/{id}": {
                "get": {
                    "tags": ["Claims"],
                    "summary": "Get a specific claim",
                    "security": [{"BearerAuth": []}],
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {
                        "200": {"description": "Claim details"},
                        "403": {"description": "Access denied (IDOR protection)"},
                        "404": {"description": "Claim not found"},
                    },
                }
            },
            "/predictions": {
                "post": {
                    "tags": ["Predictions"],
                    "summary": "Run AI fraud prediction on a vehicle image",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "required": ["vehicle_image"],
                                    "properties": {
                                        "vehicle_image": {"type": "string", "format": "binary"},
                                        "claim_id": {"type": "string", "description": "Optional: attach prediction to a claim"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Prediction result"},
                        "422": {"description": "Invalid image"},
                        "503": {"description": "Model unavailable"},
                    },
                }
            },
            "/admin/users": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "List all users (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "User list"}, "403": {"description": "Forbidden"}},
                }
            },
            "/admin/claims": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "List all claims (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"name": "risk_level", "in": "query", "schema": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}},
                        {"name": "status", "in": "query", "schema": {"type": "string"}},
                    ],
                    "responses": {"200": {"description": "All claims"}},
                }
            },
            "/admin/claims/{id}": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "Get any claim by ID (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "responses": {"200": {"description": "Claim detail"}, "404": {"description": "Not found"}},
                }
            },
            "/admin/claims/{id}/status": {
                "patch": {
                    "tags": ["Admin"],
                    "summary": "Update claim status (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["status"],
                                    "properties": {
                                        "status": {"type": "string", "enum": ["UNDER_REVIEW", "APPROVED", "REJECTED", "ESCALATED"]},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Updated claim"}},
                }
            },
            "/admin/audit-logs": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "View audit logs (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "Audit log entries"}},
                }
            },
            "/admin/statistics": {
                "get": {
                    "tags": ["Admin"],
                    "summary": "System statistics (ADMIN only)",
                    "security": [{"BearerAuth": []}],
                    "responses": {"200": {"description": "System statistics"}},
                }
            },
            "/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Basic health check",
                    "responses": {"200": {"description": "API is up"}},
                }
            },
            "/health/ready": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Readiness check (DB + ML model)",
                    "responses": {
                        "200": {"description": "All systems ready"},
                        "503": {"description": "Not ready"},
                    },
                }
            },
        },
    }
