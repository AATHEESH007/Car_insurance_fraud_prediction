# Vehicle Insurance Fraud Detection — Backend API

Production-ready Flask REST API with EfficientNetV2-S fraud detection, Grad-CAM visual explainability, and enterprise-grade security.

---

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

### 3. Model Weights & Git LFS
Ensure Git LFS is installed and model weights are present:
```bash
git lfs install
git lfs pull
```
Target weight file:
```
model/weights/best_efficientnetv2_s.pth
```

### 4. Initialize database (first time only)
```bash
flask db init
flask db migrate -m "initial"
flask db upgrade
python create_admin.py
```

### 5. Start development server
```bash
python run.py
```
The API will be available at `http://localhost:8000`.

### 6. Start production server
```bash
gunicorn -c gunicorn.conf.py run:app
```

---

## Database Configuration
By default, the project uses **SQLite** for local development (zero configuration).

To use PostgreSQL in production, set in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_detection
```

---

## API Documentation
Visit `http://localhost:8000/api/v1/docs` for interactive Swagger UI documentation.

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /api/v1/auth/register | — | Register user |
| POST | /api/v1/auth/login | — | Login |
| POST | /api/v1/auth/refresh | Refresh JWT | Rotate tokens |
| POST | /api/v1/auth/logout | JWT | Revoke token |
| POST | /api/v1/claims | JWT | Submit claim with vehicle photo |
| GET | /api/v1/claims | JWT | List own claims |
| GET | /api/v1/claims/:id | JWT | Get claim details |
| POST | /api/v1/predictions | JWT | AI fraud prediction inference |
| GET | /api/v1/admin/users | ADMIN | List registered users |
| GET | /api/v1/admin/claims | ADMIN | List all claims with triage status |
| GET | /api/v1/admin/claims/:id | ADMIN | Get any claim |
| PATCH | /api/v1/admin/claims/:id/status | ADMIN | Update claim status (Approved/Rejected) |
| GET | /api/v1/admin/claims/:id/gradcam | ADMIN | Generate & fetch Grad-CAM heatmap |
| GET | /api/v1/admin/audit-logs | ADMIN | View security audit logs |
| GET | /api/v1/admin/statistics | ADMIN | View aggregated statistics |
| GET | /api/v1/health | — | Health check |
| GET | /api/v1/health/ready | — | Model & database readiness check |

---

## Model Evaluation & Explainability

### Run Model Evaluation
```bash
python eval.py --checkpoint model/weights/best_efficientnetv2_s.pth
```
Computes MCC, ROC-AUC, Balanced Accuracy, Precision, Recall, Specificity, and detailed Confusion Matrix.

### Grad-CAM Heatmap Generation (CLI)
```bash
python features/gradcam.py --image path/to/image.jpg --checkpoint model/weights/best_efficientnetv2_s.pth
```

---

## Run Tests
```bash
pytest tests/ -v --cov=app
```

---

## Security Features
- **Argon2id password hashing**
- **Dual JWT token flow**: short-lived access tokens (15 min) + refresh tokens (30 days)
- **Token revocation & rotation** on logout
- **IDOR / BOLA protection** on all claim endpoints
- **Rate limiting**: 3/min register, 5/min login, 20/min prediction
- **Strict CORS** from environment-configured origins
- **Security headers**: CSP, HSTS, X-Frame-Options, X-Content-Type-Options
- **File upload validation**: extension + MIME + magic bytes + Pillow decode
- **UUID filenames**: user-supplied filenames never trusted
- **Full audit logging** of security events & admin decisions
- **SQLAlchemy ORM** parameterization against SQL injection
