# Vehicle Insurance Fraud Detection — Backend API

Production-ready Flask REST API with EfficientNetV2-S fraud detection.

## Quick Start (Local Development)

### 1. Set up environment
```bash
cp .env.example .env
# Edit .env with your secrets (uses SQLite by default)
```

### 2. Install dependencies
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Place model weights
Ensure the model file exists:
```
model/weights/best_efficientnetv2_s.pth
```

### 4. Initialize database (first time only)
```bash
flask db init
flask db migrate -m "initial"
flask db upgrade
```

### 5. Start development server
```bash
python run.py
```
The API will be available at `http://localhost:8000`

### 6. Start production server
```bash
gunicorn -c gunicorn.conf.py run:app
```

## Database
By default, the project uses **SQLite** for local development (no external database needed).

To use PostgreSQL instead, update `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_detection
```

## API Documentation
Visit `http://localhost:8000/api/v1/docs` for Swagger UI.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /api/v1/auth/register | — | Register user |
| POST | /api/v1/auth/login | — | Login |
| POST | /api/v1/auth/refresh | Refresh JWT | Rotate tokens |
| POST | /api/v1/auth/logout | JWT | Revoke token |
| POST | /api/v1/claims | JWT | Submit claim |
| GET | /api/v1/claims | JWT | List own claims |
| GET | /api/v1/claims/:id | JWT | Get own claim |
| POST | /api/v1/predictions | JWT | AI fraud prediction |
| GET | /api/v1/admin/users | ADMIN | List users |
| GET | /api/v1/admin/claims | ADMIN | List all claims |
| GET | /api/v1/admin/claims/:id | ADMIN | Get any claim |
| PATCH | /api/v1/admin/claims/:id/status | ADMIN | Update status |
| GET | /api/v1/admin/audit-logs | ADMIN | Audit logs |
| GET | /api/v1/admin/statistics | ADMIN | Statistics |
| GET | /api/v1/health | — | Health check |
| GET | /api/v1/health/ready | — | Readiness check |

## Run Tests
```bash
pytest tests/ -v --cov=app
```

## Example API Usage

### Register
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Jane Doe","email":"jane@example.com","password":"SecurePass1!"}'
```

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"jane@example.com","password":"SecurePass1!"}'
```

### Submit Claim with Prediction
```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "vehicle_image=@/path/to/damage.jpg"
```

### Submit Claim
```bash
curl -X POST http://localhost:8000/api/v1/claims \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -F "claim_reference=CLM-2024-001" \
  -F "vehicle_number=ABC123" \
  -F "vehicle_model=Toyota Camry" \
  -F "vehicle_year=2020" \
  -F "claim_amount=15000" \
  -F "incident_date=2024-01-15" \
  -F "description=Front-end collision damage." \
  -F "vehicle_image=@/path/to/damage.jpg"
```

## Security Features
- Argon2id password hashing
- Short-lived JWT access tokens (15 min) + refresh tokens (30 days)
- Token revocation on logout with refresh token rotation
- IDOR/BOLA protection on all claim endpoints
- Rate limiting: 3/min register, 5/min login, 20/min prediction
- Strict CORS from environment-configured origins
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- File upload validation: extension + MIME + magic bytes + Pillow decode
- UUID filenames — original filenames never trusted
- Full audit logging of security events
- SQLAlchemy ORM (no raw SQL concatenation)
