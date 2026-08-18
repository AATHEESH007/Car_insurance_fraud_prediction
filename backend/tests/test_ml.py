import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from tests.conftest import register_user, login_user, auth_header, make_jpeg_bytes


MOCK_PREDICTION = {
    "prediction": "Fraud",
    "fraud_probability": 0.87,
    "non_fraud_probability": 0.13,
    "risk_level": "HIGH",
    "recommendation": "Manual review recommended. High fraud probability detected.",
}


class TestPredictionAPI:
    def _upload(self, client, token, image_bytes=None, filename="car.jpg", ct="image/jpeg"):
        img = image_bytes or make_jpeg_bytes()
        return client.post(
            "/api/v1/predictions",
            data={"vehicle_image": (io.BytesIO(img), filename, ct)},
            content_type="multipart/form-data",
            headers=auth_header(token),
        )

    @patch("app.services.model_service.is_model_loaded", return_value=True)
    @patch("app.services.model_service.predict", return_value=MOCK_PREDICTION)
    def test_valid_prediction(self, mock_predict, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        assert resp.status_code == 200
        data = resp.get_json()["data"]["prediction"]
        assert data["prediction"] in ("Fraud", "Non-Fraud")
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["non_fraud_probability"] <= 1.0
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    @patch("app.services.model_service.is_model_loaded", return_value=False)
    def test_model_unavailable(self, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        assert resp.status_code == 503
        assert resp.get_json()["error"]["code"] == "MODEL_UNAVAILABLE"

    def test_no_image_field(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = client.post(
            "/api/v1/predictions",
            data={},
            content_type="multipart/form-data",
            headers=auth_header(token),
        )
        assert resp.status_code == 422

    def test_invalid_image_rejected(self, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token, image_bytes=b"not an image")
        assert resp.status_code == 422

    def test_unauthenticated_rejected(self, client, db):
        resp = client.post(
            "/api/v1/predictions",
            data={"vehicle_image": (io.BytesIO(make_jpeg_bytes()), "car.jpg", "image/jpeg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 401

    @patch("app.services.model_service.is_model_loaded", return_value=True)
    @patch("app.services.model_service.predict", return_value=MOCK_PREDICTION)
    def test_probability_bounds(self, mock_predict, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        data = resp.get_json()["data"]["prediction"]
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert 0.0 <= data["non_fraud_probability"] <= 1.0

    @patch("app.services.model_service.is_model_loaded", return_value=True)
    @patch("app.services.model_service.predict", return_value={**MOCK_PREDICTION, "fraud_probability": 0.87})
    def test_high_risk_classification(self, mock_predict, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        assert resp.get_json()["data"]["prediction"]["risk_level"] == "HIGH"

    @patch("app.services.model_service.is_model_loaded", return_value=True)
    @patch("app.services.model_service.predict", return_value={**MOCK_PREDICTION, "prediction": "Fraud", "fraud_probability": 0.87})
    def test_class_mapping_fraud(self, mock_predict, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        assert resp.get_json()["data"]["prediction"]["prediction"] == "Fraud"

    @patch("app.services.model_service.is_model_loaded", return_value=True)
    @patch("app.services.model_service.predict", return_value={
        "prediction": "Non-Fraud", "fraud_probability": 0.10,
        "non_fraud_probability": 0.90, "risk_level": "LOW",
        "recommendation": "Claim appears legitimate. Standard processing."
    })
    def test_class_mapping_non_fraud(self, mock_predict, mock_loaded, client, db):
        register_user(client)
        token = login_user(client)
        resp = self._upload(client, token)
        assert resp.get_json()["data"]["prediction"]["prediction"] == "Non-Fraud"
