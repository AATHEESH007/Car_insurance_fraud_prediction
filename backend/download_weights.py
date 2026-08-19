"""
Downloads model weights from WEIGHTS_URL at build time on Render.
Set WEIGHTS_URL in the Render dashboard to a direct download link
(e.g. a Hugging Face Hub file URL or a GitHub Release asset URL).
"""
import os
import urllib.request
from pathlib import Path

url = os.environ.get("WEIGHTS_URL", "")
dest = Path(os.environ.get("MODEL_PATH", "model/weights/best_efficientnetv2_s.pth"))

if not url:
    print("WEIGHTS_URL not set — skipping weight download.")
elif dest.exists():
    print(f"Weights already present at {dest} — skipping download.")
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading weights from {url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest.stat().st_size / 1e6:.1f} MB")
