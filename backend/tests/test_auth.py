import pytest
from tests.conftest import register_user, login_user, auth_header


class TestRegistration:
    def test_register_success(self, client, db):
        resp = register_user(client)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["user"]["email"] == "test@example.com"
        assert "password" not in data["data"]["user"]
        assert "password_hash" not in data["data"]["user"]

    def test_register_duplicate_email(self, client, db):
        register_user(client)
        resp = register_user(client)
        assert resp.status_code == 409
        assert resp.get_json()["error"]["code"] == "EMAIL_TAKEN"

    def test_register_weak_password(self, client, db):
        resp = client.post("/api/v1/auth/register", json={
            "name": "Test", "email": "weak@example.com", "password": "short"
        })
        assert resp.status_code == 422

    def test_register_invalid_email(self, client, db):
        resp = client.post("/api/v1/auth/register", json={
            "name": "Test", "email": "not-an-email", "password": "Password1!"
        })
        assert resp.status_code == 422

    def test_register_missing_fields(self, client, db):
        resp = client.post("/api/v1/auth/register", json={"email": "a@b.com"})
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client, db):
        register_user(client)
        resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com", "password": "Password1!"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]

    def test_login_wrong_password(self, client, db):
        register_user(client)
        resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com", "password": "WrongPass1!"
        })
        assert resp.status_code == 401
        assert resp.get_json()["error"]["code"] == "INVALID_CREDENTIALS"

    def test_login_unknown_email(self, client, db):
        resp = client.post("/api/v1/auth/login", json={
            "email": "nobody@example.com", "password": "Password1!"
        })
        assert resp.status_code == 401

    def test_login_returns_no_sensitive_data(self, client, db):
        register_user(client)
        resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com", "password": "Password1!"
        })
        body = resp.get_json()
        assert "password" not in str(body)
        assert "password_hash" not in str(body)


class TestJWT:
    def test_access_protected_without_token(self, client, db):
        resp = client.get("/api/v1/claims")
        assert resp.status_code == 401

    def test_access_with_invalid_token(self, client, db):
        resp = client.get("/api/v1/claims", headers=auth_header("invalid.token.here"))
        assert resp.status_code == 401

    def test_logout_revokes_token(self, client, db):
        register_user(client)
        token = login_user(client)
        assert token is not None

        resp = client.post("/api/v1/auth/logout", headers=auth_header(token))
        assert resp.status_code == 200

        resp2 = client.get("/api/v1/claims", headers=auth_header(token))
        assert resp2.status_code == 401

    def test_refresh_token_rotation(self, client, db):
        register_user(client)
        login_resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com", "password": "Password1!"
        })
        refresh_token = login_resp.get_json()["data"]["refresh_token"]

        resp = client.post("/api/v1/auth/refresh", headers=auth_header(refresh_token))
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert "access_token" in data
        assert "refresh_token" in data

        resp2 = client.post("/api/v1/auth/refresh", headers=auth_header(refresh_token))
        assert resp2.status_code == 401
