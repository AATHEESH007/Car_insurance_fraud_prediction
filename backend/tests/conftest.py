import os
import io
import pytest
from PIL import Image

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-for-pytest")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FLASK_ENV", "testing")

from app import create_app
from app.config import TestingConfig
from app.extensions import db as _db


@pytest.fixture(scope="session")
def app():
    application = create_app(TestingConfig)
    with application.app_context():
        _db.create_all()
        yield application
        _db.drop_all()


@pytest.fixture(autouse=True)
def db(app):
    with app.app_context():
        yield _db
        _db.session.remove()
        for table in reversed(_db.metadata.sorted_tables):
            _db.session.execute(table.delete())
        _db.session.commit()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


def make_jpeg_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_webp_bytes(width=64, height=64) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(50, 200, 100))
    img.save(buf, format="WEBP")
    return buf.getvalue()


def register_user(client, name="Test User", email="test@example.com", password="Password1!"):
    return client.post("/api/v1/auth/register", json={"name": name, "email": email, "password": password})


def login_user(client, email="test@example.com", password="Password1!"):
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    data = resp.get_json()
    if resp.status_code == 200 and data and "data" in data:
        return data["data"]["access_token"]
    return None


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
