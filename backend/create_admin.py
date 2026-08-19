import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables before importing app
load_dotenv()

from app import create_app
from app.extensions import db
from app.models.user import User, UserRole
from app.services.auth_service import hash_password

app = create_app()

with app.app_context():
    # Check if admin already exists
    admin = User.query.filter_by(email="admin").first()
    if admin:
        print("Admin user already exists!")
        print("Email: admin")
        print("Password: admin")
        sys.exit(0)

    # Create admin user
    admin_user = User(
        name="Admin",
        email="admin",
        password_hash=hash_password("admin"),
        role=UserRole.ADMIN,
        is_active=True,
    )

    db.session.add(admin_user)
    db.session.commit()

    print("✅ Admin created successfully!")
    print("Email: admin")
    print("Password: admin")
