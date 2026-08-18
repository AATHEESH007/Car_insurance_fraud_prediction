import pytest
from tests.conftest import register_user, login_user, auth_header


class TestCORS:
    def test_cors_blocked_from_unknown_origin(self, client, db):
        resp = client.get("/api/v1/health", headers={"Origin": "https://evil.com"})
        assert "Access-Control-Allow-Origin" not in resp.headers or \
               resp.headers.get("Access-Control-Allow-Origin") != "https://evil.com"


class TestSecurityHeaders:
    def test_x_content_type_options(self, client):
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        resp = client.get("/api/v1/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_request_id_returned(self, client):
        resp = client.get("/api/v1/health")
        assert "X-Request-ID" in resp.headers

    def test_custom_request_id_propagated(self, client):
        resp = client.get("/api/v1/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers.get("X-Request-ID") == "my-custom-id"


class TestIDOR:
    def test_user_cannot_access_another_users_claim(self, client, db):
        register_user(client, email="victim@example.com")
        register_user(client, name="Attacker", email="attacker@example.com")

        victim_token = login_user(client, email="victim@example.com")
        attacker_token = login_user(client, email="attacker@example.com")

        claim_data = {
            "claim_reference": "CLM-SEC-001",
            "vehicle_number": "SEC001",
            "vehicle_model": "BMW 3 Series",
            "vehicle_year": "2022",
            "claim_amount": "30000",
            "incident_date": "2024-06-01",
            "description": "Parking lot collision with significant door damage.",
        }
        resp = client.post(
            "/api/v1/claims",
            data=claim_data,
            content_type="multipart/form-data",
            headers=auth_header(victim_token),
        )
        claim_id = resp.get_json()["data"]["claim"]["id"]

        resp = client.get(f"/api/v1/claims/{claim_id}", headers=auth_header(attacker_token))
        assert resp.status_code == 403


class TestMissingAuth:
    def test_claims_requires_auth(self, client):
        assert client.get("/api/v1/claims").status_code == 401
        assert client.post("/api/v1/claims", data={}).status_code == 401

    def test_predictions_requires_auth(self, client):
        assert client.post("/api/v1/predictions", data={}).status_code == 401

    def test_admin_requires_auth(self, client):
        assert client.get("/api/v1/admin/users").status_code == 401
        assert client.get("/api/v1/admin/claims").status_code == 401
        assert client.get("/api/v1/admin/audit-logs").status_code == 401
        assert client.get("/api/v1/admin/statistics").status_code == 401


class TestPrivilegeEscalation:
    def test_user_cannot_reach_admin_endpoints(self, client, db):
        register_user(client)
        token = login_user(client)
        assert client.get("/api/v1/admin/users", headers=auth_header(token)).status_code == 403
        assert client.get("/api/v1/admin/claims", headers=auth_header(token)).status_code == 403
        assert client.get("/api/v1/admin/audit-logs", headers=auth_header(token)).status_code == 403
        assert client.get("/api/v1/admin/statistics", headers=auth_header(token)).status_code == 403


class TestSQLInjection:
    def test_sql_injection_in_email(self, client, db):
        resp = client.post("/api/v1/auth/login", json={
            "email": "' OR '1'='1", "password": "anything"
        })
        assert resp.status_code in (401, 422)

    def test_sql_injection_in_query_param(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = client.get("/api/v1/claims/'; DROP TABLE claims; --", headers=auth_header(token))
        assert resp.status_code == 404
