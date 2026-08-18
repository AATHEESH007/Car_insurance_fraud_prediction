import io
import pytest
from tests.conftest import register_user, login_user, auth_header, make_jpeg_bytes

CLAIM_DATA = {
    "claim_reference": "CLM-2024-001",
    "vehicle_number": "ABC123",
    "vehicle_model": "Toyota Camry",
    "vehicle_year": "2020",
    "claim_amount": "15000.00",
    "incident_date": "2024-01-15",
    "description": "Vehicle collision at intersection causing significant front-end damage.",
}


def post_claim(client, token, data=None, image=None):
    form = data or CLAIM_DATA.copy()
    if image is not None:
        form["vehicle_image"] = (io.BytesIO(image), "damage.jpg", "image/jpeg")
    return client.post(
        "/api/v1/claims",
        data=form,
        content_type="multipart/form-data",
        headers=auth_header(token),
    )


class TestClaimCreation:
    def test_create_claim_no_image(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = post_claim(client, token)
        assert resp.status_code == 201
        data = resp.get_json()["data"]["claim"]
        assert data["claim_reference"] == "CLM-2024-001"
        assert data["status"] == "SUBMITTED"

    def test_create_claim_with_jpeg(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = post_claim(client, token, image=make_jpeg_bytes())
        assert resp.status_code == 201

    def test_duplicate_claim_reference(self, client, db):
        register_user(client)
        token = login_user(client)
        post_claim(client, token)
        resp = post_claim(client, token)
        assert resp.status_code == 409

    def test_invalid_year(self, client, db):
        register_user(client)
        token = login_user(client)
        bad = CLAIM_DATA.copy()
        bad["vehicle_year"] = "1800"
        resp = post_claim(client, token, data=bad)
        assert resp.status_code == 422

    def test_negative_amount(self, client, db):
        register_user(client)
        token = login_user(client)
        bad = CLAIM_DATA.copy()
        bad["claim_amount"] = "-500"
        resp = post_claim(client, token, data=bad)
        assert resp.status_code == 422

    def test_future_incident_date(self, client, db):
        register_user(client)
        token = login_user(client)
        bad = CLAIM_DATA.copy()
        bad["incident_date"] = "2099-01-01"
        resp = post_claim(client, token, data=bad)
        assert resp.status_code == 422

    def test_unauthenticated_rejected(self, client, db):
        resp = client.post("/api/v1/claims", data=CLAIM_DATA, content_type="multipart/form-data")
        assert resp.status_code == 401


class TestClaimAccess:
    def test_user_sees_own_claims(self, client, db):
        register_user(client)
        token = login_user(client)
        post_claim(client, token)
        resp = client.get("/api/v1/claims", headers=auth_header(token))
        assert resp.status_code == 200
        assert len(resp.get_json()["data"]["claims"]) == 1

    def test_user_can_get_own_claim(self, client, db):
        register_user(client)
        token = login_user(client)
        claim_id = post_claim(client, token).get_json()["data"]["claim"]["id"]
        resp = client.get(f"/api/v1/claims/{claim_id}", headers=auth_header(token))
        assert resp.status_code == 200

    def test_idor_protection(self, client, db):
        register_user(client, email="user1@example.com")
        register_user(client, name="User2", email="user2@example.com")
        token1 = login_user(client, email="user1@example.com")
        token2 = login_user(client, email="user2@example.com")

        claim_id = post_claim(client, token1).get_json()["data"]["claim"]["id"]

        resp = client.get(f"/api/v1/claims/{claim_id}", headers=auth_header(token2))
        assert resp.status_code == 403

    def test_claim_not_found(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = client.get("/api/v1/claims/nonexistent-id", headers=auth_header(token))
        assert resp.status_code == 404
