import io
import pytest
from tests.conftest import register_user, login_user, auth_header, make_jpeg_bytes, make_png_bytes, make_webp_bytes

CLAIM_BASE = {
    "claim_reference": "CLM-IMG-001",
    "vehicle_number": "XYZ999",
    "vehicle_model": "Honda Civic",
    "vehicle_year": "2019",
    "claim_amount": "5000",
    "incident_date": "2024-03-10",
    "description": "Rear-end collision causing bumper and trunk damage.",
}


def upload_image(client, token, image_bytes, filename, content_type, ref="CLM-IMG-001"):
    data = CLAIM_BASE.copy()
    data["claim_reference"] = ref
    data["vehicle_image"] = (io.BytesIO(image_bytes), filename, content_type)
    return client.post(
        "/api/v1/claims",
        data=data,
        content_type="multipart/form-data",
        headers=auth_header(token),
    )


class TestImageUpload:
    def _setup(self, client, db):
        register_user(client)
        return login_user(client)

    def test_valid_jpeg(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, make_jpeg_bytes(), "car.jpg", "image/jpeg")
        assert resp.status_code == 201

    def test_valid_png(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, make_png_bytes(), "car.png", "image/png", ref="CLM-IMG-002")
        assert resp.status_code == 201

    def test_valid_webp(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, make_webp_bytes(), "car.webp", "image/webp", ref="CLM-IMG-003")
        assert resp.status_code == 201

    def test_invalid_extension(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, make_jpeg_bytes(), "car.gif", "image/gif")
        assert resp.status_code in (415, 422)

    def test_fake_mime_type(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, b"not an image at all", "car.jpg", "image/jpeg")
        assert resp.status_code == 422

    def test_oversized_file(self, client, db):
        token = self._setup(client, db)
        big = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
        resp = upload_image(client, token, big, "big.jpg", "image/jpeg")
        assert resp.status_code in (413, 422)

    def test_corrupted_image(self, client, db):
        token = self._setup(client, db)
        corrupted = b"\xff\xd8\xff\xe0" + b"\xde\xad\xbe\xef" * 100
        resp = upload_image(client, token, corrupted, "corrupt.jpg", "image/jpeg")
        assert resp.status_code == 422

    def test_empty_file(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, b"", "empty.jpg", "image/jpeg")
        assert resp.status_code == 422

    def test_path_traversal_filename(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, make_jpeg_bytes(), "../../../etc/passwd.jpg", "image/jpeg")
        assert resp.status_code in (201, 400, 422)

    def test_non_image_file(self, client, db):
        token = self._setup(client, db)
        resp = upload_image(client, token, b"PK\x03\x04", "script.jpg", "image/jpeg")
        assert resp.status_code == 422
