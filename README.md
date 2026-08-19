# 🛡️ AI-Powered Vehicle Insurance Fraud Detection System

An enterprise-grade, end-to-end vehicle insurance claim fraud detection platform leveraging **Deep Learning (EfficientNetV2-S)**, **Explainable AI (Grad-CAM)**, and a **Full-Stack Web Architecture (Flask REST API + React SPA)**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Machine Learning & Explainability Pipeline](#-machine-learning--explainability-pipeline)
  - [Model Architecture (EfficientNetV2-S)](#model-architecture-efficientnetv2-s)
  - [Explainable AI with Grad-CAM](#explainable-ai-with-grad-cam)
  - [Training & Evaluation Metrics](#training--evaluation-metrics)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Running the Application](#running-the-application)
- [API Documentation](#-api-documentation)
- [Security & Compliance](#-security--compliance)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Deployment](#-deployment)
- [License](#-license)

---

## 🔍 Overview

Insurance fraud costs the automotive insurance industry billions annually. This project provides an automated, explainable fraud detection pipeline that analyzes submitted vehicle damage imagery and claim details in real-time.

By combining state-of-the-art **EfficientNetV2-S** computer vision with **Grad-CAM visual heatmaps**, claims adjusters and fraud investigation teams gain immediate probabilistic predictions accompanied by visual evidence explaining *why* a claim was flagged.

---

## ✨ Key Features

- **🧠 Deep Learning Fraud Classifier**: Fine-tuned EfficientNetV2-S with transfer learning, stochastic depth, dropout, and label smoothing.
- **🔥 Grad-CAM Visual Explainability**: High-resolution heatmap overlays identifying exact vehicle damage regions and artifacts influencing predictions.
- **⚡ Real-Time Fraud Assessment**: Instant automated risk score, confidence interval, and actionable recommendation (e.g., Fast-Track Approval vs. Manual SIU Inspection).
- **👥 Role-Based Portals**:
  - **Claimant Portal**: Seamless claim filing with drag-and-drop vehicle photo uploads, instant submission tracking, and real-time status updates.
  - **Adjuster / Admin Portal**: Split-column triage view ("Fraud Flagged" vs. "Likely Genuine"), batch approval workflows, interactive Grad-CAM inspection modal, and statistics analytics.
- **🔐 Enterprise Security**: Argon2id password hashing, JWT access/refresh token rotation, IDOR protection, strict rate limiting, magic-byte image validation, and comprehensive audit trails.
- **📊 Comprehensive Evaluation**: Scripts for calculating MCC, ROC-AUC, Balanced Accuracy, Precision, Recall, Specificity, F1-scores, and Confusion Matrices.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────────────┐
                                  │      React + Vite Frontend     │
                                  │  (Claimant & Admin Dashboard)  │
                                  └───────────────┬────────────────┘
                                                  │ HTTPS / REST API
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │       Flask REST API (v1)      │
                                  │   (JWT Auth, Rate Limiter,     │
                                  │    Security Headers, CORS)     │
                                  └───────────────┬────────────────┘
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
   ┌───────────────────────────┐    ┌───────────────────────────┐    ┌───────────────────────────┐
   │    SQLAlchemy ORM DB      │    │   EfficientNetV2-S Model  │    │     Grad-CAM Service      │
   │  (Users, Claims, Audits)  │    │   (Inference & Prediction)│    │   (Visual Heatmap Gen)    │
   └───────────────────────────┘    └───────────────────────────┘    └───────────────────────────┘
```

---

## 🔬 Machine Learning & Explainability Pipeline

### Model Architecture (EfficientNetV2-S)
- **Base Architecture**: EfficientNetV2-S pre-trained on ImageNet-1K with progressive learning.
- **Custom Classifier**: Adaptive Average Pooling $\to$ Flatten $\to$ Dropout(0.4) $\to$ Linear(1280, 256) $\to$ SiLU $\to$ BatchNorm $\to$ Dropout(0.3) $\to$ Linear(256, 2).
- **Optimization**: AdamW optimizer with Cosine Annealing / ReduceLROnPlateau learning rate scheduling, Automatic Mixed Precision (AMP), and Gradient Clipping (max norm 1.0).
- **Data Augmentation**: Mix of RandomResizedCrop, RandomHorizontalFlip, RandomRotation(15°), ColorJitter, and RandomErasing for robust generalization.

### Explainable AI with Grad-CAM
- **Layer Targeting**: Gradients extracted with respect to the final convolutional stage (`features[7]`).
- **Heatmap Generation**: Computes weighted activation maps, applies Jet colormapping, and overlays transparently on original vehicle damage images.
- **Adjuster Inspection**: Heatmap visualizations are exposed through protected admin endpoints and displayed inside the Claim Review modal.

### Training & Evaluation Metrics
The model is evaluated using comprehensive multi-dimensional classification metrics:
- **Matthews Correlation Coefficient (MCC)**
- **Area Under ROC Curve (ROC-AUC)**
- **Balanced Accuracy & Macro/Weighted F1-Score**
- **Sensitivity (Recall) & Specificity**

```bash
# Run standalone evaluation on the test dataset
python backend/eval.py --checkpoint backend/model/weights/best_efficientnetv2_s.pth
```

---

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask 3.1.0
- **Database**: SQLite (Development) / PostgreSQL (Production) via Flask-SQLAlchemy 3.1 & Flask-Migrate
- **Authentication**: Flask-JWT-Extended (Short-lived Access + Refresh Rotation) & Argon2-cffi
- **Security & Utilities**: Flask-Limiter, Flask-CORS, Marshmallow schema validation
- **Deep Learning**: PyTorch 2.4.1, Torchvision 0.19.1, PyTorch-Grad-CAM 1.5.4, NumPy, Pillow, Scikit-Learn

### Frontend
- **Framework**: React 18 + Vite
- **Styling**: Modern Vanilla CSS Design System (Glassmorphism, Dark/Light palettes, Micro-animations)
- **Icons**: Lucide React
- **HTTP Client**: Native Fetch with JWT interceptor & auto-token refresh

---

## 📁 Project Directory Structure

```
Insurance-Fraud-Detection/
├── backend/
│   ├── app/
│   │   ├── middleware/          # Security headers, rate limiters, error handlers
│   │   ├── models/              # SQLAlchemy database models (User, Claim, AuditLog)
│   │   ├── routes/              # Blueprint routes (Auth, Claims, Admin, Predictions, Health)
│   │   ├── schemas/             # Marshmallow validation schemas
│   │   ├── services/            # Core business logic (Model, GradCAM, Image, Audit, Claim)
│   │   └── utils/               # Security utilities, custom responses, validators
│   ├── Data/                    # Preprocessing & augmentation scripts
│   ├── features/                # Grad-CAM visualization utilities
│   ├── model/
│   │   ├── weights/             # Trained PyTorch model checkpoint (.pth via Git LFS)
│   │   └── architecture.py      # Neural network architecture definitions
│   ├── tests/                   # PyTest unit & integration tests
│   ├── training_scripts/        # Modular EfficientNetV2 training scripts
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Backend container definition
│   └── run.py                   # Development entrypoint
├── frontend/
│   ├── src/
│   │   ├── ClaimSeal.jsx        # Complete Frontend Single-Page Application
│   │   ├── App.css / index.css  # Design system styling & typography
│   │   └── main.jsx             # React DOM entry point
│   ├── package.json             # NPM dependencies & scripts
│   └── vite.config.js           # Vite configuration & proxy settings
├── .gitattributes               # Git LFS tracking configuration
├── .gitignore                   # Ignored files (virtualenvs, cache, datasets, raw DBs)
└── README.md                    # Root project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.10+ (Recommended: 3.11)
- **Node.js**: 18+ and npm
- **Git LFS**: Required for model weight management (`git lfs install`)

### Backend Setup

1. **Navigate to the backend directory and set up a virtual environment**:
   ```bash
   cd backend
   python -m venv .venv

   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your custom JWT secret and settings
   ```

4. **Initialize Database & Seed Admin**:
   ```bash
   flask db init
   flask db migrate -m "initial"
   flask db upgrade
   python create_admin.py
   ```

5. **Start Backend Server**:
   ```bash
   python run.py
   # API will be active on http://localhost:8000
   ```

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   # Web interface will open at http://localhost:5173
   ```

---

## 📖 API Documentation

Interactive Swagger API documentation is available locally at:
👉 `http://localhost:8000/api/v1/docs`

### Primary Endpoints Summary

| Method | Endpoint | Access | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Public | Register new user account |
| `POST` | `/api/v1/auth/login` | Public | Authenticate user & return JWT tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh JWT | Rotate access & refresh tokens |
| `POST` | `/api/v1/claims` | User JWT | Submit vehicle insurance claim with photo |
| `GET` | `/api/v1/claims` | User JWT | List claims submitted by current user |
| `GET` | `/api/v1/claims/<id>` | User / Admin | Retrieve details of a specific claim |
| `POST` | `/api/v1/predictions` | User / Admin | Run instant AI fraud inference on image |
| `GET` | `/api/v1/admin/claims` | Admin JWT | List all claims with fraud triage status |
| `PATCH` | `/api/v1/admin/claims/<id>/status`| Admin JWT | Update claim status (Approved / Rejected) |
| `GET` | `/api/v1/admin/claims/<id>/gradcam`| Admin JWT | Generate / fetch Grad-CAM heatmap overlay |
| `GET` | `/api/v1/admin/statistics` | Admin JWT | Aggregated claim & fraud analytics |
| `GET` | `/api/v1/health` | Public | Service health & model readiness status |

---

## 🔒 Security & Compliance

- **Argon2id Hashing**: Memory-hard key derivation protecting user passwords against GPU-assisted brute-force attacks.
- **JWT Protection & Revocation**: Dual-token architecture with blacklist token revocation on logout.
- **Strict Input Validation**: Marshmallow schema enforcement, UUID filename rewriting, and image magic-byte inspection.
- **Rate Limiting**: Built-in Flask-Limiter protecting auth and inference endpoints from DoS/credential stuffing.
- **BOLA / IDOR Safeguards**: Tenant-isolated claim querying preventing horizontal privilege escalation.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with coverage reporting:
```bash
pytest backend/tests/ -v --cov=backend/app
```

---

## 🚢 Deployment

### Production Docker Deployment
```bash
cd backend
docker build -t insurance-fraud-backend:latest .
docker run -p 8000:8000 --env-file .env insurance-fraud-backend:latest
```

### Cloud Platforms
- **Backend**: Render / Railway / AWS ECS using `gunicorn -c gunicorn.conf.py run:app`.
- **Frontend**: Vercel / Netlify with `npm run build` targeting the `/dist` output directory.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
