import os
import uuid
import logging
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"RIFF": "webp",
}
MAX_SIZE_BYTES = 10 * 1024 * 1024


def validate_and_save_image(file: FileStorage, upload_dir: str) -> str:
    """
    Validates the uploaded file against multiple security checks,
    saves it with a UUID filename, and returns the saved path.
    Raises ValueError with a safe message on any validation failure.
    """
    raw = file.read(MAX_SIZE_BYTES + 1)
    if len(raw) > MAX_SIZE_BYTES:
        raise ValueError("File size exceeds 10 MB limit.")
    if len(raw) == 0:
        raise ValueError("Uploaded file is empty.")

    _check_magic_bytes(raw)
    _check_mime_type(file.content_type)
    _check_filename_extension(file.filename)
    _decode_with_pillow(raw)

    ext = _safe_extension(raw)
    filename = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(upload_dir, filename)
    os.makedirs(upload_dir, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(raw)

    return save_path


def _check_magic_bytes(raw: bytes):
    for sig, fmt in MAGIC_SIGNATURES.items():
        if raw[:len(sig)] == sig:
            if fmt == "webp" and raw[8:12] != b"WEBP":
                raise ValueError("Invalid image file.")
            return
    raise ValueError("Invalid image file format.")


def _check_mime_type(content_type: str):
    if not content_type:
        raise ValueError("Missing content type.")
    ct = content_type.split(";")[0].strip().lower()
    if ct not in ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported image type. Allowed: JPEG, PNG, WEBP.")


def _check_filename_extension(filename: str):
    if not filename:
        raise ValueError("Missing filename.")
    name = filename.lower()
    if "\x00" in name:
        raise ValueError("Invalid filename.")
    parts = name.rsplit(".", 1)
    if len(parts) < 2:
        raise ValueError("File has no extension.")
    ext = parts[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file extension.")
    if name.count(".") > 1:
        base = parts[0]
        if any(base.endswith(f".{e}") for e in ["php", "js", "exe", "sh", "py", "rb"]):
            raise ValueError("Double extension detected.")


def _decode_with_pillow(raw: bytes):
    try:
        img = Image.open(BytesIO(raw))
        img.verify()
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError("Cannot decode image. File may be corrupted.")


def _safe_extension(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return "jpg"


def open_image_for_inference(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        img.load()
        return img.convert("RGB")
