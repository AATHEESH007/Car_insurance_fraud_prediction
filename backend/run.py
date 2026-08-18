import os
from dotenv import load_dotenv

load_dotenv()

from app import create_app

app = create_app()

if __name__ == "__main__":
    if os.environ.get("FLASK_ENV") == "development":
        app.run(host="0.0.0.0", port=8000, debug=True)
    else:
        raise RuntimeError("Use gunicorn in production. Run: gunicorn -c gunicorn.conf.py run:app")
