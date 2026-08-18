import io
import pytest
from app.models.user import User, UserRole
from app.extensions import db as _db
from app.services.auth_service import hash_password
from tests.conftest import register_user, login_user, auth_header, make_jpeg_bytes

CLAIM_DATA = {
    "claim_reference": "CLM-ADMIN-001",
    "vehicle_number": "ADMIN001",
    "vehicle_model": "Ford F-150",
    "vehicle_year": "2021",
    "claim_amount": "20000",
    "incident_date": "2024-05-01",
    "description": "Side-impact collision on highway requiring major body repairs.",
}


def create_admin(app):
    with app.app_context():
        admin = User(
            name="Admin User",
            email="admin@example.com",
            password_hash=hash_password("AdminPass1!"),
            role=UserRole.ADMIN,
        )
        _db.session.add(admin)
        _db.session.commit()


def login_admin(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "admin@example.com", "password": "AdminPass1!"
    })
    return resp.get_json()["data"]["access_token"]


class TestAdminAccess:
    def test_user_cannot_access_admin_users(self, client, db, app):
        register_user(client)
        token = login_user(client)
        resp = client.get("/api/v1/admin/users", headers=auth_header(token))
        assert resp.status_code == 403

    def test_unauthenticated_cannot_access_admin(self, client, db):
        resp = client.get("/api/v1/admin/users")
        assert resp.status_code == 401

    def test_admin_can_list_users(self, client, db, app):
        create_admin(app)
        register_user(client)
        token = login_admin(client)
        resp = client.get("/api/v1/admin/users", headers=auth_header(token))
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert "users" in data
        assert data["total"] >= 1

    def test_admin_can_list_all_claims(self, client, db, app):
        create_admin(app)
        register_user(client)
        user_token = login_user(client)
        client.post(
            "/api/v1/claims",
            data=CLAIM_DATA,
            content_type="multipart/form-data",
            headers=auth_header(user_token),
        )
        admin_token = login_admin(client)
        resp = client.get("/api/v1/admin/claims", headers=auth_header(admin_token))
        assert resp.status_code == 200
        assert resp.get_json()["data"]["total"] >= 1

    def test_admin_can_update_claim_status(self, client, db, app):
        create_admin(app)
        register_user(client)
        user_token = login_user(client)
        claim_resp = client.post(
            "/api/v1/claims",
            data=CLAIM_DATA,
            content_type="multipart/form-data",
            headers=auth_header(user_token),
        )
        claim_id = claim_resp.get_json()["data"]["claim"]["id"]

        admin_token = login_admin(client)
        resp = client.patch(
            f"/api/v1/admin/claims/{claim_id}/status",
            json={"status": "APPROVED"},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.get_json()["data"]["claim"]["status"] == "APPROVED"

    def test_user_cannot_update_claim_status(self, client, db, app):
        create_admin(app)
        register_user(client)
        user_token = login_user(client)
        claim_resp = client.post(
            "/api/v1/claims",
            data=CLAIM_DATA,
            content_type="multipart/form-data",
            headers=auth_header(user_token),
        )
        claim_id = claim_resp.get_json()["data"]["claim"]["id"]
        resp = client.patch(
            f"/api/v1/admin/claims/{claim_id}/status",
            json={"status": "APPROVED"},
            headers=auth_header(user_token),
        )
        assert resp.status_code == 403

    def test_admin_can_view_audit_logs(self, client, db, app):
        create_admin(app)
        token = login_admin(client)
        resp = client.get("/api/v1/admin/audit-logs", headers=auth_header(token))
        assert resp.status_code == 200

    def test_admin_can_view_statistics(self, client, db, app):
        create_admin(app)
        token = login_admin(client)
        resp = client.get("/api/v1/admin/statistics", headers=auth_header(token))
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert "users" in data
        assert "claims" in data
