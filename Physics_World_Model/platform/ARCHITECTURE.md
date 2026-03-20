# PWM Platform Architecture & Implementation Plan

**Domain:** pwm.platformai.org
**CPU Server:** 34.63.169.185 (Ubuntu, no GPU)
**GPU Workers:** Modal (on-demand)
**Storage:** GCP (GCS + BigQuery)
**Date:** 2026-02-22

---

## A) Executive Architecture Summary

PWM Platform is a three-tier system:

1. **CPU Control Plane** (34.63.169.185) — FastAPI backend + Jinja2/HTMX UI + PostgreSQL + Redis. Handles auth, job orchestration, metadata registry, UI, and API. Runs behind Caddy (auto-TLS).

2. **Modal GPU Worker Plane** — On-demand GPU functions for heavy simulation, reconstruction, calibration, and operator fitting. Invoked asynchronously from the CPU server via Modal's Python SDK.

3. **GCP Data Plane** — GCS for canonical dataset storage (simulation, real, calibration, manifests). BigQuery for analytics over manifests and run metadata. Optional Firestore for lightweight metadata if Postgres proves insufficient.

**Auth:** Reuses the CompareGPT-AIScientist pattern — OAuth 2.0 SSO redirect → JWT access tokens (HS256, 7-day expiry), stored client-side, verified per-request via FastAPI `Depends()`. Adapted for PWM with local-password fallback option and security hardening.

---

## B) Text Architecture Diagram

```
                         ┌──────────────────────────────────────┐
                         │         pwm.platformai.org           │
                         │    Caddy reverse proxy (auto-TLS)    │
                         └──────────────┬───────────────────────┘
                                        │ HTTPS :443
                         ┌──────────────▼───────────────────────┐
                         │     CPU Control Plane (FastAPI)       │
                         │  34.63.169.185                       │
                         │                                       │
                         │  ┌─────────┐  ┌──────────────────┐  │
                         │  │ Auth /  │  │  Jinja2 + HTMX   │  │
                         │  │ JWT     │  │  Simple UI        │  │
                         │  └────┬────┘  └────────┬─────────┘  │
                         │       │                │              │
                         │  ┌────▼────────────────▼─────────┐  │
                         │  │       FastAPI API Router       │  │
                         │  │  /api/v1/runs, /api/v1/auth    │  │
                         │  │  /api/v1/datasets, /api/v1/    │  │
                         │  │  bootstrap, /api/v1/modalities │  │
                         │  └──────────┬────────────────────┘  │
                         │             │                        │
                         │  ┌──────────▼────────────────────┐  │
                         │  │    Job Orchestrator (arq)      │  │
                         │  │    Redis-backed task queue     │  │
                         │  └──────┬──────────┬─────────────┘  │
                         │         │          │                 │
                         │  ┌──────▼──┐  ┌────▼──────────┐    │
                         │  │Postgres │  │   Redis       │    │
                         │  │(users,  │  │(queue,cache,  │    │
                         │  │ runs,   │  │ sessions)     │    │
                         │  │ meta)   │  └───────────────┘    │
                         │  └─────────┘                        │
                         └────────┬──────────────┬─────────────┘
                                  │              │
                    ┌─────────────▼──┐    ┌──────▼──────────────┐
                    │  Modal GPU     │    │  GCP Data Plane      │
                    │  Workers       │    │                      │
                    │                │    │  ┌────────────────┐  │
                    │ simulate()     │    │  │  GCS Buckets   │  │
                    │ reconstruct()  │    │  │  (canonical)   │  │
                    │ fit_operator() │    │  └───────┬────────┘  │
                    │ calibrate()    │    │          │            │
                    │ bootstrap_sim()│    │  ┌───────▼────────┐  │
                    └────────────────┘    │  │  BigQuery      │  │
                                          │  │  (analytics)   │  │
                                          │  └────────────────┘  │
                                          └──────────────────────┘
```

---

## C) Service Selection Table

| Component | Choice | Justification |
|---|---|---|
| **Backend API** | FastAPI 0.115+ | Async, Pydantic-native (matches PWM types), OpenAPI auto-docs |
| **UI** | Jinja2 templates + HTMX + Tailwind CSS | Server-rendered, minimal JS, fast to build, easy to maintain |
| **Database** | PostgreSQL 16 | Relational integrity for users/runs/metadata/bootstrap proposals |
| **Cache / Queue** | Redis 7 | Session cache, arq task queue, rate limiting |
| **Task Queue** | arq (async Redis queue) | Lightweight, async-native, fits FastAPI; dispatches to Modal |
| **Reverse Proxy** | Caddy 2 | Auto-TLS (Let's Encrypt), simple config, HTTP/2 |
| **Auth** | JWT (HS256) + SSO redirect (CompareGPT pattern) | Consistency across products |
| **GPU Workers** | Modal | On-demand GPUs, no infra management, Python-native SDK |
| **Object Storage** | GCS | Canonical datasets, manifests, RunBundles |
| **Analytics** | BigQuery | SQL over manifests, run metadata, benchmarks |
| **Process Manager** | Docker Compose | Postgres + Redis + FastAPI + Caddy in one stack |
| **Password Hashing** | bcrypt (passlib) | For local-password fallback (SSO primary) |

---

## D) Simple UI Design

### Pages / Screens

| # | Page | Route | Auth Required |
|---|---|---|---|
| 1 | **Login** | `/login` | No |
| 2 | **SSO Callback** | `/sso/callback` | No |
| 3 | **Signup** (local fallback) | `/signup` | No |
| 4 | **Forgot Password** | `/forgot-password` | No |
| 5 | **Dashboard / My Runs** | `/` | Yes |
| 6 | **New Run** | `/runs/new` | Yes |
| 7 | **Run Status** | `/runs/{run_id}` | Yes |
| 8 | **Run Results** (TriadReport) | `/runs/{run_id}/results` | Yes |
| 9 | **Dataset Catalog** | `/datasets` | Yes |
| 10 | **Modality Catalog** | `/modalities` | Yes |
| 11 | **New Modality Bootstrap** | `/bootstrap/new` | Yes |
| 12 | **Bootstrap Review Queue** | `/bootstrap/review` | Yes (admin) |
| 13 | **Admin Panel** | `/admin` | Yes (admin) |
| 14 | **API Docs** | `/docs` | No (FastAPI auto) |

### Core User Flows

**Flow 1: Submit a Run**
```
Login → Dashboard → "New Run" →
  Choose modality → Upload spec or enter prompt →
  Select compute (CPU/GPU/auto) → Submit →
  Redirect to Run Status (live polling via HTMX) →
  View Results (TriadReport summary, download RunBundle)
```

**Flow 2: Bootstrap New Modality**
```
Login → Modality Catalog → "Bootstrap New Modality" →
  Fill modality basics form (physics class, sensor type, geometry) →
  System shows similar existing modalities + confidence scores →
  Review auto-generated OperatorGraph template + ExperimentSpec →
  Edit/accept → Submit for review →
  Admin reviews → Approve/revise → Initial dataset pack generated
```

**Flow 3: Browse Datasets**
```
Login → Dataset Catalog → Filter by modality/type →
  View dataset card (metadata, manifest summary, splits) →
  Download link (GCS signed URL) or copy CLI command
```

### Wireframe Layouts

```
┌─────────────────────────────────────────────────┐
│  PWM Platform    [Datasets] [Modalities] [Runs] │
│                                    [User ▼]     │
├─────────────────────────────────────────────────┤
│                                                  │
│  Dashboard                                       │
│  ┌────────────────────────────────────────────┐  │
│  │  Recent Runs                    [+ New Run]│  │
│  │  ┌──────┬──────────┬────────┬──────────┐  │  │
│  │  │ ID   │ Modality │ Status │ Created  │  │  │
│  │  │ r-42 │ CASSI    │ ✓ done │ 2h ago   │  │  │
│  │  │ r-41 │ CT       │ ⟳ run  │ 3h ago   │  │  │
│  │  │ r-40 │ MRI      │ ✗ fail │ 1d ago   │  │  │
│  │  └──────┴──────────┴────────┴──────────┘  │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Quick Stats                               │  │
│  │  Runs: 42  │  Modalities: 12  │  GPU hrs: 8│  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────┐
│  New Run                                         │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input Mode: ○ Prompt  ○ Spec (JSON/YAML)       │
│                ○ Measured Data                    │
│                                                  │
│  [Prompt text area / Spec upload / y upload]     │
│                                                  │
│  Modality: [CASSI ▼]                             │
│  Task:     [simulate_recon_analyze ▼]            │
│                                                  │
│  Compute:  ○ CPU  ○ GPU  ● Auto                 │
│  Est. time: ~5 min (GPU) / ~45 min (CPU)        │
│  Est. cost: ~$0.12 (GPU)                        │
│                                                  │
│  [Submit Run]                                    │
└─────────────────────────────────────────────────┘
```

### Minimal API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/auth/validate` | SSO token → JWT exchange |
| POST | `/api/v1/auth/login` | Local email/password login |
| POST | `/api/v1/auth/signup` | Local user registration |
| POST | `/api/v1/auth/logout` | Invalidate session |
| GET  | `/api/v1/auth/me` | Current user info |
| POST | `/api/v1/runs` | Submit new run |
| GET  | `/api/v1/runs` | List user's runs |
| GET  | `/api/v1/runs/{id}` | Run status + metadata |
| GET  | `/api/v1/runs/{id}/results` | TriadReport + artifacts |
| GET  | `/api/v1/runs/{id}/bundle` | Download RunBundle |
| GET  | `/api/v1/datasets` | List datasets |
| GET  | `/api/v1/datasets/{id}` | Dataset metadata + manifest |
| GET  | `/api/v1/modalities` | List supported modalities |
| GET  | `/api/v1/modalities/{key}` | Modality details |
| POST | `/api/v1/bootstrap` | Submit bootstrap proposal |
| GET  | `/api/v1/bootstrap` | List proposals |
| PATCH| `/api/v1/bootstrap/{id}` | Review/approve proposal |
| GET  | `/api/v1/bootstrap/{id}/outputs` | Generated templates |

---

## E) Auth / Login Reuse Plan from CompareGPT-AIScientist

### What CompareGPT Uses (Exactly)

| Aspect | Implementation |
|---|---|
| **Auth type** | OAuth 2.0 SSO → JWT exchange |
| **JWT algorithm** | HS256, 7-day expiry |
| **Token storage** | localStorage (client) |
| **User DB** | SQLite (users table) |
| **Password hashing** | None (SSO-delegated) |
| **Session type** | Stateless (JWT only) |
| **Route protection** | FastAPI `Depends(get_current_user)` |
| **Frontend guard** | Vue router `beforeEach` + meta.requiresAuth |
| **Token manager** | `backend/core/token_manager.py` (PyJWT) |
| **Auth service** | `backend/services/auth_service.py` |
| **SSO provider** | `https://auth.comparegpt.io/sso/validate` |

### Key Files in CompareGPT

```
backend/
  core/token_manager.py    — JWT create/verify (HS256)
  core/config.py           — Settings (SECRET_KEY, SSO_VALIDATE_URL)
  services/auth_service.py — exchange_sso_token, validate_access_token, logout
  routers/user.py          — /api/user/validate, /api/user/logout, /api/user/me
  db/models.py             — UserModel (SQLAlchemy)
  db/schemas.py            — ValidateRequest/Response, UserInfo
  db/repo.py               — UserRepository (upsert_user, get_user, clear_user_data)
frontend/
  src/services/user.ts     — initiateLogin, handleOAuthCallback, validateToken
  src/stores/auth.ts       — Pinia store (initialize, login, logout)
  src/router/index.ts      — Navigation guards
  src/views/SSOCallbackView.vue — OAuth callback handler
```

### PWM Auth Replication Plan

Since PWM uses server-rendered Jinja2 + HTMX (not a Vue SPA), the auth flow adapts:

**Backend (reuse directly with modifications):**

```python
# platform/auth/token_manager.py — IDENTICAL to CompareGPT
# (copy token_manager.py, change import paths)

# platform/auth/service.py — ADAPTED from CompareGPT auth_service.py
# Changes:
#   1. Add local email/password login (bcrypt) as fallback
#   2. Use Postgres instead of SQLite
#   3. Add role field (user/admin/reviewer)
#   4. Add CSRF token generation for form-based UI

# platform/auth/dependencies.py — FastAPI Depends functions
# (same pattern as get_current_user_id / get_current_user)

# platform/auth/models.py — SQLAlchemy User model for Postgres
# (extend CompareGPT UserModel with password_hash, role, email)
```

**Frontend (adapt for server-rendered pages):**

```
Instead of Vue SPA + localStorage:
  - JWT stored in HttpOnly secure cookie (not localStorage — security improvement)
  - Login page is server-rendered Jinja2 form
  - SSO callback is a server-side route that sets the cookie
  - HTMX requests automatically include cookie
  - Route protection via FastAPI middleware (check cookie on every request)
```

### PWM User Model (Postgres)

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=True)       # for local login
    password_hash = Column(String(255), nullable=True)             # bcrypt, for local login
    sso_user_id = Column(Integer, unique=True, nullable=True)      # from SSO provider
    sso_token = Column(String(512), nullable=True)                 # SSO token for refresh
    username = Column(String(100), nullable=False)
    role = Column(String(20), default="user")                      # user/admin/reviewer
    api_key = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

### PWM Auth Flow (Adapted)

```
FLOW A: SSO Login (same as CompareGPT)
  1. User clicks "Login with SSO" on /login page
  2. Server redirects to SSO_URL (e.g., https://auth.comparegpt.io/...)
  3. SSO authenticates user, redirects to /sso/callback?token=<sso_token>
  4. Server-side /sso/callback:
     a. Calls SSO validate endpoint with sso_token
     b. Upserts user in Postgres
     c. Creates JWT access_token
     d. Sets HttpOnly secure cookie with JWT
     e. Redirects to /

FLOW B: Local Login (new, optional fallback)
  1. User fills email + password on /login page
  2. POST /api/v1/auth/login
  3. Server verifies bcrypt hash
  4. Creates JWT, sets HttpOnly cookie
  5. Redirects to /

FLOW C: API Access (same JWT pattern as CompareGPT)
  1. API clients use Authorization: Bearer <token> header
  2. FastAPI Depends(get_current_user) validates JWT
  3. Returns 401 with require_reauth if invalid
```

### Security Hardening (Preserving Compatibility)

| Issue in CompareGPT | PWM Improvement |
|---|---|
| localStorage for JWT (XSS risk) | HttpOnly + Secure + SameSite=Lax cookie |
| No CSRF protection | CSRF tokens for form submissions |
| No rate limiting | Rate limit on /login, /signup (10/min) |
| SECRET_KEY can default to random | Require explicit SECRET_KEY env var |
| 7-day token expiry, no refresh | Keep 7-day for compat, add optional refresh |
| No password support | Add bcrypt local login as fallback |
| SQLite | PostgreSQL for production reliability |
| No token blacklist | Redis-based token blacklist on logout |
| `allow_origins=["*"]` risk | Explicit CORS_ORIGINS whitelist |

### Example PWM Auth Routes

```python
# platform/routers/auth.py

from fastapi import APIRouter, Depends, Response, Request, HTTPException
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

@router.post("/validate")
async def validate_sso_token(req: ValidateRequest, response: Response):
    """Exchange SSO token for JWT (CompareGPT-compatible)."""
    if req.sso_token:
        result = await auth_service.exchange_sso_token(req.sso_token)
        # Set HttpOnly cookie
        response.set_cookie(
            key="access_token",
            value=result["access_token"],
            httponly=True,
            secure=True,           # HTTPS only
            samesite="lax",
            max_age=7 * 24 * 3600, # 7 days
            path="/",
        )
        return result
    # Validate existing token
    return await auth_service.validate_access_token(req)

@router.post("/login")
async def local_login(req: LoginRequest, response: Response):
    """Local email/password login (PWM addition)."""
    user = await auth_service.verify_password(req.email, req.password)
    token = token_manager.create_access_token(user.id)
    response.set_cookie(
        key="access_token", value=token,
        httponly=True, secure=True, samesite="lax",
        max_age=7 * 24 * 3600, path="/",
    )
    return {"success": True, "access_token": token, "user": user.to_dict()}

@router.post("/signup")
async def signup(req: SignupRequest, response: Response):
    """Local registration."""
    user = await auth_service.create_local_user(req.email, req.username, req.password)
    token = token_manager.create_access_token(user.id)
    response.set_cookie(...)
    return {"success": True}

@router.post("/logout")
async def logout(response: Response, user=Depends(get_current_user)):
    """Clear session."""
    response.delete_cookie("access_token", path="/")
    await auth_service.logout_user(user.id)
    return {"success": True, "message": "Logged out"}

@router.get("/me")
async def me(user=Depends(get_current_user)):
    return {"success": True, "user": user}
```

### Auth Middleware for Cookie + Bearer

```python
# platform/auth/dependencies.py

async def get_current_user(request: Request) -> User:
    """Extract user from HttpOnly cookie OR Authorization header."""
    token = None

    # 1. Try HttpOnly cookie (UI sessions)
    token = request.cookies.get("access_token")

    # 2. Fallback to Authorization header (API clients)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(status_code=401, detail={
            "error": "missing_token",
            "message": "Authentication required",
            "require_reauth": True
        })

    user_id = token_manager.verify_access_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail={
            "error": "invalid_token",
            "message": "Token invalid or expired",
            "require_reauth": True
        })

    user = await user_repo.get_by_id(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail={
            "error": "user_not_found",
            "message": "User not found",
            "require_reauth": True
        })
    return user


def require_role(role: str):
    """Dependency factory for role-based access."""
    async def _check(user: User = Depends(get_current_user)):
        if user.role not in (role, "admin"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return _check
```

### Session/Auth Middleware Config

```python
# platform/config.py (environment variables — PLACEHOLDERS ONLY)

class Settings(BaseSettings):
    # Auth
    SECRET_KEY: str                          # REQUIRED — no default
    SSO_VALIDATE_URL: str = ""               # e.g., https://auth.comparegpt.io/api/sso/validate
    SSO_REDIRECT_URL: str = ""               # e.g., https://auth.comparegpt.io/sso-redirect?redirect=...
    ACCESS_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    # CSRF
    CSRF_SECRET: str                         # REQUIRED

    # Rate limiting
    LOGIN_RATE_LIMIT: str = "10/minute"

    # Database
    DATABASE_URL: str                        # e.g., postgresql+asyncpg://user:PASS@localhost/pwm
    REDIS_URL: str = "redis://localhost:6379"

    # Modal
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""

    # GCP
    GCS_BUCKET: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    BIGQUERY_PROJECT: str = ""
    BIGQUERY_DATASET: str = ""

    # CORS
    CORS_ORIGINS: list[str] = ["https://pwm.platformai.org"]

    class Config:
        env_file = ".env"
```

---

## F) Canonical Folder Layout (GCS)

```
gs://pwm-canonical/
├── datasets/
│   ├── simulation/
│   │   ├── cassi/
│   │   │   ├── cassi_v1_10scene/
│   │   │   │   ├── manifest.json
│   │   │   │   ├── README.md
│   │   │   │   ├── splits.json
│   │   │   │   ├── checksums.sha256
│   │   │   │   ├── data/
│   │   │   │   │   ├── scene_001.npz
│   │   │   │   │   └── ...
│   │   │   │   └── metadata/
│   │   │   │       ├── experiment_spec.yaml
│   │   │   │       └── operator_graph.json
│   │   │   └── cassi_v2_kaist/
│   │   │       └── ...
│   │   ├── ct/
│   │   ├── mri/
│   │   └── ptychography/
│   ├── real/
│   │   ├── cassi/
│   │   │   └── cassi_kaist_real_v1/
│   │   │       ├── manifest.json
│   │   │       ├── data/
│   │   │       ├── calibration/
│   │   │       │   ├── mask.npz
│   │   │       │   ├── dispersion_params.json
│   │   │       │   └── dark_frame.npz
│   │   │       └── metadata/
│   │   └── ...
│   └── benchmarks/
│       ├── public/
│       │   └── cassi_benchmark_v1/
│       └── private/
│           └── cassi_benchmark_internal_v1/
├── setup/
│   ├── cassi/
│   │   ├── default_experiment_spec.yaml
│   │   ├── operator_graph_template.json
│   │   └── calibration_schema.json
│   └── ...
├── runs/
│   ├── {run_id}/
│   │   ├── runbundle.json
│   │   ├── triad_report.json
│   │   ├── artifacts/
│   │   │   ├── x_hat.npz
│   │   │   ├── residual.npz
│   │   │   └── uncertainty_map.npz
│   │   └── provenance/
│   │       ├── git_hash.txt
│   │       ├── environment.json
│   │       └── config_snapshot.yaml
│   └── ...
├── bootstrap/
│   ├── proposals/
│   │   └── {proposal_id}/
│   │       ├── proposal.json
│   │       ├── operator_graph_template.json
│   │       ├── experiment_spec_template.yaml
│   │       └── review_history.json
│   └── knowledge_base/
│       ├── modality_basics.json
│       ├── physics_classes.json
│       ├── similarity_weights.json
│       └── embeddings/
│           └── modality_embeddings.npz
└── manifests/
    └── bigquery/
        ├── datasets_manifest.jsonl
        └── runs_manifest.jsonl
```

---

## G) Core Schemas (Pydantic / JSON Examples)

### DatasetSpec

```python
class DatasetSpec(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier")
    version: str = Field("1.0.0")
    modality: str = Field(..., description="Modality key, e.g., 'cassi'")
    data_type: Literal["simulation", "real", "benchmark"] = "simulation"
    description: str = ""

    # Source / provenance
    source: str = ""               # e.g., "pwm_simulator_v0.2", "kaist_lab"
    license: str = "internal"

    # Dimensions
    num_samples: int = 0
    x_shape: list[int] = []        # e.g., [256, 256, 28]
    y_shape: list[int] = []        # e.g., [256, 310]

    # Splits
    splits: dict[str, list[int]] = {}  # {"train": [0..7], "val": [8,9]}

    # Checksums
    checksum_algo: str = "sha256"
    total_checksum: str = ""

    # Links
    gcs_prefix: str = ""           # gs://pwm-canonical/datasets/simulation/cassi/...
    experiment_spec_ref: str = ""  # path to linked ExperimentSpec
    calibration_ref: str = ""      # path to calibration files
    manifest_ref: str = ""         # path to manifest.json

    # Metadata
    tags: list[str] = []
    created_at: str = ""
    created_by: str = ""
```

### ExperimentSetupSpec Linkage

```python
class ExperimentSetupLink(BaseModel):
    """Links a dataset to its experimental setup configuration."""
    dataset_id: str
    experiment_spec_path: str      # GCS path to ExperimentSpec YAML
    operator_graph_path: str       # GCS path to OperatorGraph JSON
    calibration_paths: dict[str, str] = {}  # {"mask": "gs://...", "dispersion": "gs://..."}
    setup_version: str = "1.0"
    notes: str = ""
```

### Manifest Entry Schema

```python
class ManifestEntry(BaseModel):
    """Per-file entry in a dataset manifest."""
    file_path: str                 # relative path within dataset
    file_type: str                 # "data", "calibration", "metadata", "doc"
    size_bytes: int
    checksum: str                  # SHA-256
    shape: list[int] | None = None
    dtype: str | None = None
    sample_index: int | None = None
    split: str | None = None       # "train", "val", "test"
    tags: list[str] = []

class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    modality: str
    num_entries: int
    entries: list[ManifestEntry]
    created_at: str
    created_by: str
```

### RunBundle Metadata Record

```python
class RunBundleRecord(BaseModel):
    run_id: str
    user_id: int
    modality: str
    task_kind: str                 # "simulate_recon_analyze", "calibrate_and_reconstruct", etc.
    status: Literal["pending", "queued", "running", "completed", "failed", "cancelled"]
    compute_mode: Literal["cpu", "gpu", "auto"]

    # Input
    input_mode: str                # "prompt", "spec", "measured"
    experiment_spec: dict = {}     # resolved spec snapshot
    dataset_id: str | None = None

    # Execution
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    modal_job_id: str | None = None
    worker_type: str | None = None  # "cpu", "modal-a10g", "modal-a100"

    # Outputs
    gcs_bundle_path: str | None = None
    triad_report_summary: dict | None = None
    error_message: str | None = None

    # Provenance
    pwm_version: str = ""
    git_hash: str = ""
    config_hash: str = ""
```

### TriadReport Summary Record

```python
class TriadReportSummary(BaseModel):
    run_id: str
    modality: str

    # Metrics
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    sam: float | None = None       # spectral angle mapper (for spectral)
    custom_metrics: dict[str, float] = {}

    # Diagnosis
    diagnosis_severity: str = "info"  # info/warning/error
    diagnosis_codes: list[str] = []
    recommended_actions: list[str] = []

    # Uncertainty
    uncertainty_mean: float | None = None
    uncertainty_max: float | None = None
    confidence_interval: list[float] | None = None  # [lower, upper]

    # Operator fidelity
    operator_mismatch_detected: bool = False
    mismatch_type: str | None = None
    theta_fitted: dict | None = None

    # Provenance
    reconstruction_method: str = ""
    solver_iterations: int | None = None
    convergence_residual: float | None = None
```

### BootstrapProposal Schema

```python
class BootstrapProposal(BaseModel):
    proposal_id: str
    modality_key: str              # proposed modality key
    display_name: str
    submitted_by: int              # user_id
    submitted_at: str
    status: Literal["draft", "submitted", "under_review", "approved",
                     "revision_requested", "rejected"]
    version: int = 1

    # Modality basics
    physics_class: str             # "spectral", "tomographic", "coherent", ...
    forward_model_family: str      # "linear_projection", "fourier_sampling", ...
    sensor_type: str
    source_type: str
    geometry: str
    noise_model: str               # "gaussian", "poisson", "mixed"

    # Generated outputs
    operator_graph_template: dict = {}
    experiment_spec_template: dict = {}
    simulation_plan: dict = {}
    collection_checklist: list[str] = []
    calibration_modes: list[str] = []
    recommended_metrics: list[str] = []
    benchmark_tasks: list[str] = []
    uncertainty_notes: list[str] = []
    viability_checklist: dict[str, bool] = {}

    # Similarity
    similar_modalities: list[dict] = []   # [{key, score, explanation}]

    # Review
    reviewer_id: int | None = None
    review_notes: str = ""
    review_history: list[dict] = []
```

### ModalityBasics Schema

```python
class ModalityBasics(BaseModel):
    modality_key: str
    display_name: str
    category: str                  # "spectral", "medical", "coherent", "microscopy", ...

    # Physics
    physics_class: str
    forward_model_family: str
    primitive_gates: list[str]     # e.g., ["dispersion", "coded_aperture", "integration"]
    wave_model: str                # "ray", "scalar_wave", "full_em", "particle"

    # Sensor / source
    sensor_type: str               # "ccd", "cmos", "photon_counter", "interferometric"
    source_type: str               # "broadband", "monochromatic", "xray_tube", "laser"
    geometry: str                  # "planar", "rotational", "ptychographic_scan", "mri_kspace"

    # Data
    typical_x_dims: list[int]
    typical_y_dims: list[int]
    typical_snr_range: list[float]

    # Calibration
    calibration_params: list[str]  # ["mask", "dispersion_curve", "psf", ...]
    mismatch_modes: list[str]      # ["shift_error", "spectral_response_drift", ...]

    # Noise
    noise_model: str
    noise_params: dict = {}

    # Reconstruction
    reconstruction_task_types: list[str]
    default_solver: str
    evaluation_metrics: list[str]

    # Setup template
    default_experiment_spec: dict = {}
    default_operator_graph: dict = {}

    # References
    canonical_references: list[str] = []
    canonical_datasets: list[str] = []

    # Metadata for similarity
    feature_vector: list[float] = []   # for embedding-based similarity
    tags: list[str] = []
```

### SimilarityMatch Schema

```python
class SimilarityMatch(BaseModel):
    query_modality: str
    matched_modality: str
    overall_score: float           # 0.0 to 1.0

    # Component scores
    physics_score: float = 0.0
    sensor_score: float = 0.0
    geometry_score: float = 0.0
    operator_score: float = 0.0
    task_score: float = 0.0
    noise_score: float = 0.0

    # Explanation
    explanation: str = ""
    reusable_components: list[str] = []  # ["dispersion_gate", "coded_aperture_mask"]
    adaptation_notes: list[str] = []
    confidence: float = 0.0
```

### BootstrapReviewDecision Schema

```python
class BootstrapReviewDecision(BaseModel):
    proposal_id: str
    reviewer_id: int
    decision: Literal["approve", "revise", "reject"]
    notes: str
    suggested_changes: dict = {}
    reviewed_at: str
    checklist_status: dict[str, bool] = {}  # viability checks
```

### User / Auth Schemas

```python
# Matching CompareGPT-AIScientist style, extended for PWM

class UserCreate(BaseModel):
    email: str
    username: str
    password: str                  # plaintext, hashed server-side

class UserLogin(BaseModel):
    email: str
    password: str

class ValidateRequest(BaseModel):
    """CompareGPT-compatible SSO token exchange."""
    sso_token: str | None = None

class ValidateResponse(BaseModel):
    success: bool
    access_token: str | None = None
    valid: bool | None = None
    user: dict | None = None

class UserInfo(BaseModel):
    user_id: int
    username: str
    email: str | None = None
    role: str = "user"

class UserProfile(BaseModel):
    user_info: UserInfo
    runs_count: int = 0
    modalities_used: list[str] = []

class LogoutResponse(BaseModel):
    success: bool
    message: str
```

### Example Instances

**CASSI (Spectral)**
```json
{
  "dataset_id": "cassi_sim_10scene_v1",
  "modality": "cassi",
  "data_type": "simulation",
  "description": "10-scene CASSI simulation, 28-band spectral cube, coded aperture",
  "num_samples": 10,
  "x_shape": [256, 256, 28],
  "y_shape": [256, 310],
  "gcs_prefix": "gs://pwm-canonical/datasets/simulation/cassi/cassi_sim_10scene_v1/"
}
```

**CT (Medical)**
```json
{
  "dataset_id": "ct_fan_beam_sim_v1",
  "modality": "ct",
  "data_type": "simulation",
  "description": "Fan-beam CT simulation, 256x256 phantoms, 180 projections",
  "num_samples": 50,
  "x_shape": [256, 256],
  "y_shape": [180, 362],
  "gcs_prefix": "gs://pwm-canonical/datasets/simulation/ct/ct_fan_beam_sim_v1/"
}
```

**Ptychography (Coherent)**
```json
{
  "dataset_id": "ptycho_sim_v1",
  "modality": "ptychography",
  "data_type": "simulation",
  "description": "Ptychographic imaging simulation, overlapping scans, complex-valued probe",
  "num_samples": 20,
  "x_shape": [512, 512],
  "y_shape": [64, 128, 128],
  "gcs_prefix": "gs://pwm-canonical/datasets/simulation/ptychography/ptycho_sim_v1/"
}
```

---

## H) Registry Schema (PostgreSQL)

```sql
-- Users (auth — CompareGPT-compatible + extensions)
CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(255) UNIQUE,
    username        VARCHAR(100) NOT NULL,
    password_hash   VARCHAR(255),          -- bcrypt, NULL for SSO-only users
    sso_user_id     INTEGER UNIQUE,        -- from SSO provider
    sso_token       VARCHAR(512),
    role            VARCHAR(20) DEFAULT 'user',  -- user/admin/reviewer
    api_key         VARCHAR(255),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Datasets
CREATE TABLE datasets (
    id              SERIAL PRIMARY KEY,
    dataset_id      VARCHAR(255) UNIQUE NOT NULL,
    version         VARCHAR(20) DEFAULT '1.0.0',
    modality        VARCHAR(100) NOT NULL,
    data_type       VARCHAR(20) NOT NULL,  -- simulation/real/benchmark
    description     TEXT,
    source          VARCHAR(255),
    license         VARCHAR(100) DEFAULT 'internal',
    num_samples     INTEGER DEFAULT 0,
    x_shape         JSONB,
    y_shape         JSONB,
    gcs_prefix      VARCHAR(1024),
    experiment_spec JSONB,                 -- snapshot of linked spec
    calibration_ref VARCHAR(1024),
    manifest_ref    VARCHAR(1024),
    tags            JSONB DEFAULT '[]',
    is_public       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    created_by      INTEGER REFERENCES users(id)
);
CREATE INDEX idx_datasets_modality ON datasets(modality);

-- Runs
CREATE TABLE runs (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(100) UNIQUE NOT NULL,
    user_id         INTEGER REFERENCES users(id),
    modality        VARCHAR(100) NOT NULL,
    task_kind       VARCHAR(50) NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',
    compute_mode    VARCHAR(10) DEFAULT 'auto',
    input_mode      VARCHAR(20),
    experiment_spec JSONB,
    dataset_id      VARCHAR(255) REFERENCES datasets(dataset_id),
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    duration_seconds FLOAT,
    modal_job_id    VARCHAR(255),
    worker_type     VARCHAR(50),
    gcs_bundle_path VARCHAR(1024),
    error_message   TEXT,
    pwm_version     VARCHAR(50),
    git_hash        VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_runs_user_id ON runs(user_id);
CREATE INDEX idx_runs_status ON runs(status);

-- Triad Reports
CREATE TABLE triad_reports (
    id              SERIAL PRIMARY KEY,
    run_id          VARCHAR(100) REFERENCES runs(run_id) UNIQUE,
    modality        VARCHAR(100),
    psnr            FLOAT,
    ssim            FLOAT,
    lpips           FLOAT,
    sam             FLOAT,
    custom_metrics  JSONB DEFAULT '{}',
    diagnosis_severity VARCHAR(20),
    diagnosis_codes JSONB DEFAULT '[]',
    recommended_actions JSONB DEFAULT '[]',
    uncertainty_mean FLOAT,
    uncertainty_max FLOAT,
    confidence_interval JSONB,
    operator_mismatch_detected BOOLEAN DEFAULT FALSE,
    mismatch_type   VARCHAR(100),
    theta_fitted    JSONB,
    reconstruction_method VARCHAR(100),
    solver_iterations INTEGER,
    convergence_residual FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Bootstrap Proposals
CREATE TABLE bootstrap_proposals (
    id              SERIAL PRIMARY KEY,
    proposal_id     VARCHAR(100) UNIQUE NOT NULL,
    modality_key    VARCHAR(100) NOT NULL,
    display_name    VARCHAR(255),
    submitted_by    INTEGER REFERENCES users(id),
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    status          VARCHAR(30) DEFAULT 'draft',
    version         INTEGER DEFAULT 1,
    physics_class   VARCHAR(100),
    forward_model_family VARCHAR(100),
    sensor_type     VARCHAR(100),
    source_type     VARCHAR(100),
    geometry        VARCHAR(100),
    noise_model     VARCHAR(50),
    operator_graph_template JSONB,
    experiment_spec_template JSONB,
    simulation_plan JSONB,
    collection_checklist JSONB DEFAULT '[]',
    calibration_modes JSONB DEFAULT '[]',
    recommended_metrics JSONB DEFAULT '[]',
    benchmark_tasks JSONB DEFAULT '[]',
    uncertainty_notes JSONB DEFAULT '[]',
    viability_checklist JSONB DEFAULT '{}',
    similar_modalities JSONB DEFAULT '[]',
    reviewer_id     INTEGER REFERENCES users(id),
    review_notes    TEXT,
    review_history  JSONB DEFAULT '[]'
);
CREATE INDEX idx_bootstrap_status ON bootstrap_proposals(status);

-- Modality Basics (knowledge base)
CREATE TABLE modality_basics (
    id              SERIAL PRIMARY KEY,
    modality_key    VARCHAR(100) UNIQUE NOT NULL,
    display_name    VARCHAR(255),
    category        VARCHAR(100),
    physics_class   VARCHAR(100),
    forward_model_family VARCHAR(100),
    primitive_gates JSONB DEFAULT '[]',
    wave_model      VARCHAR(50),
    sensor_type     VARCHAR(100),
    source_type     VARCHAR(100),
    geometry        VARCHAR(100),
    typical_x_dims  JSONB,
    typical_y_dims  JSONB,
    typical_snr_range JSONB,
    calibration_params JSONB DEFAULT '[]',
    mismatch_modes  JSONB DEFAULT '[]',
    noise_model     VARCHAR(50),
    noise_params    JSONB DEFAULT '{}',
    reconstruction_task_types JSONB DEFAULT '[]',
    default_solver  VARCHAR(100),
    evaluation_metrics JSONB DEFAULT '[]',
    default_experiment_spec JSONB DEFAULT '{}',
    default_operator_graph JSONB DEFAULT '{}',
    canonical_references JSONB DEFAULT '[]',
    canonical_datasets JSONB DEFAULT '[]',
    feature_vector  JSONB DEFAULT '[]',
    tags            JSONB DEFAULT '[]',
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

## I) BigQuery Manifest Schema + Example SQL

### BigQuery Table: `pwm.manifests.datasets`

```sql
CREATE TABLE `pwm_analytics.manifests.datasets` (
  dataset_id      STRING NOT NULL,
  version         STRING,
  modality        STRING NOT NULL,
  data_type       STRING,          -- simulation/real/benchmark
  num_samples     INT64,
  x_shape         ARRAY<INT64>,
  y_shape         ARRAY<INT64>,
  gcs_prefix      STRING,
  is_public       BOOL,
  tags            ARRAY<STRING>,
  created_at      TIMESTAMP,
  created_by      STRING
);
```

### BigQuery Table: `pwm.manifests.runs`

```sql
CREATE TABLE `pwm_analytics.manifests.runs` (
  run_id          STRING NOT NULL,
  user_id         INT64,
  modality        STRING,
  task_kind       STRING,
  status          STRING,
  compute_mode    STRING,
  input_mode      STRING,
  submitted_at    TIMESTAMP,
  completed_at    TIMESTAMP,
  duration_seconds FLOAT64,
  worker_type     STRING,
  psnr            FLOAT64,
  ssim            FLOAT64,
  lpips           FLOAT64,
  reconstruction_method STRING,
  operator_mismatch_detected BOOL,
  pwm_version     STRING
);
```

### Example Queries

```sql
-- Average PSNR by modality
SELECT modality, AVG(psnr) as avg_psnr, COUNT(*) as n_runs
FROM `pwm_analytics.manifests.runs`
WHERE status = 'completed' AND psnr IS NOT NULL
GROUP BY modality
ORDER BY avg_psnr DESC;

-- Datasets per modality
SELECT modality, data_type, COUNT(*) as n_datasets, SUM(num_samples) as total_samples
FROM `pwm_analytics.manifests.datasets`
GROUP BY modality, data_type;

-- GPU vs CPU performance comparison
SELECT modality, worker_type,
       AVG(duration_seconds) as avg_duration,
       AVG(psnr) as avg_psnr,
       COUNT(*) as n_runs
FROM `pwm_analytics.manifests.runs`
WHERE status = 'completed'
GROUP BY modality, worker_type;

-- Modalities with operator mismatch issues
SELECT modality, COUNT(*) as mismatch_count,
       AVG(psnr) as avg_psnr_with_mismatch
FROM `pwm_analytics.manifests.runs`
WHERE operator_mismatch_detected = TRUE
GROUP BY modality;
```

---

## J) Lifecycle & Storage Policy

```yaml
# GCS lifecycle policy for pwm-canonical bucket
lifecycle:
  rules:
    # Move old run artifacts to Nearline after 90 days
    - action: { type: SetStorageClass, storageClass: NEARLINE }
      condition:
        age: 90
        matchesPrefix: ["runs/"]

    # Move old run artifacts to Coldline after 365 days
    - action: { type: SetStorageClass, storageClass: COLDLINE }
      condition:
        age: 365
        matchesPrefix: ["runs/"]

    # Keep canonical datasets in Standard always
    # (no lifecycle rule needed — default is Standard)

    # Delete scratch/temp after 30 days
    - action: { type: Delete }
      condition:
        age: 30
        matchesPrefix: ["scratch/"]

# Cost strategy:
# - Canonical datasets (Standard): always-hot, frequently accessed
# - Run artifacts (Standard → Nearline → Coldline): accessed less over time
# - Scratch (delete after 30d): temporary workspace
# - Bootstrap proposals (Standard): small, keep hot
```

---

## K) Ingestion Pipeline CLI + Module Structure

### Pipeline Steps

```
1. Validate schema (DatasetSpec YAML/JSON)
2. Compute checksums (SHA-256 per file)
3. Verify completeness (all files referenced in manifest exist)
4. Generate manifest.json + splits.json
5. Register metadata in Postgres
6. Upload to GCS (with checksum verification)
7. Publish BigQuery-compatible manifest (JSONL)
8. Generate ingestion report (audit log)
```

### CLI Commands

```bash
# Ingest a new dataset
pwm-platform dataset ingest \
  --spec dataset_spec.yaml \
  --data-dir ./data/ \
  --modality cassi \
  --type simulation \
  --upload-to-gcs

# Validate without uploading
pwm-platform dataset validate \
  --spec dataset_spec.yaml \
  --data-dir ./data/

# Regenerate manifest
pwm-platform dataset manifest \
  --data-dir ./data/ \
  --output manifest.json

# Register in metadata DB only (data already in GCS)
pwm-platform dataset register \
  --spec dataset_spec.yaml \
  --gcs-prefix gs://pwm-canonical/datasets/simulation/cassi/v1/

# Export BigQuery manifest
pwm-platform dataset export-bq \
  --dataset-id cassi_sim_10scene_v1
```

### Module Structure

```
platform/
├── ingestion/
│   ├── __init__.py
│   ├── cli.py              # Click/Typer CLI entrypoint
│   ├── validator.py         # Schema validation (Pydantic)
│   ├── checksums.py         # SHA-256 computation + verification
│   ├── manifest.py          # Manifest generation
│   ├── splits.py            # Train/val/test split generation
│   ├── uploader.py          # GCS upload with verification
│   ├── registry.py          # Postgres metadata registration
│   ├── bigquery.py          # BigQuery manifest export
│   └── report.py            # Ingestion audit report
```

### Example Ingestion YAML Config

```yaml
# dataset_spec.yaml
dataset_id: cassi_sim_10scene_v1
version: "1.0.0"
modality: cassi
data_type: simulation
description: "10-scene CASSI simulation dataset"
source: pwm_simulator_v0.2
license: internal

num_samples: 10
x_shape: [256, 256, 28]
y_shape: [256, 310]

splits:
  train: [0, 1, 2, 3, 4, 5, 6, 7]
  val: [8]
  test: [9]

tags:
  - spectral
  - coded_aperture
  - benchmark

experiment_spec_ref: setup/cassi/default_experiment_spec.yaml
calibration_ref: ""
```

---

## L) Modality Bootstrap Engine Design

### L.1) Knowledge Base

The modality basics knowledge base stores structured information about every known imaging modality. Stored in Postgres `modality_basics` table + a JSON/YAML seed file.

**What to store per modality:**

| Field | Example (CASSI) | Example (CT) |
|---|---|---|
| physics_class | spectral_coding | tomographic |
| forward_model_family | coded_aperture_dispersion | radon_transform |
| primitive_gates | [dispersion, coded_aperture, integration] | [xray_source, rotation, line_integral, detection] |
| wave_model | ray | ray |
| sensor_type | cmos | scintillator_detector |
| source_type | broadband | xray_tube |
| geometry | planar | rotational |
| noise_model | gaussian+poisson | poisson |
| calibration_params | [mask, dispersion_curve, dark_frame] | [flat_field, center_of_rotation, beam_hardening_correction] |
| mismatch_modes | [mask_shift, spectral_response_drift] | [center_offset, beam_hardening, scatter] |
| default_solver | gap_tv | fbp_then_tv |
| evaluation_metrics | [psnr, ssim, sam] | [psnr, ssim, hu_accuracy] |

### L.2) Similarity Retrieval

```python
class ModalitySimilarityEngine:
    """Find similar modalities for bootstrap guidance."""

    # Weight vector for similarity dimensions
    WEIGHTS = {
        "physics_class": 0.25,
        "forward_model_family": 0.20,
        "sensor_type": 0.10,
        "geometry": 0.15,
        "noise_model": 0.10,
        "wave_model": 0.10,
        "primitive_gates_overlap": 0.10,
    }

    def find_similar(self, query: ModalityBasics, top_k: int = 5) -> list[SimilarityMatch]:
        """
        Hybrid similarity:
        1. Rule-based: exact match scores on categorical fields
        2. Jaccard overlap: on primitive_gates, calibration_params, mismatch_modes
        3. Optional: embedding cosine similarity (if feature_vector populated)
        4. Weighted combination → ranked list
        """
        ...

    def extract_reusable_components(self, match: SimilarityMatch) -> list[str]:
        """Identify which components from matched modality can be reused."""
        ...
```

### L.3) Bootstrap Outputs

When a user requests bootstrap for a new modality, the engine generates:

1. **Candidate OperatorGraph template** — assembled from similar modalities' gate patterns
2. **Candidate ExperimentSpec template** — filled with typical dims/ranges from similar modalities
3. **Simulation dataset design plan** — number of samples, parameter ranges, noise levels
4. **Real-data collection checklist** — what physical setup needs, calibration measurements required
5. **Expected calibration/mismatch modes** — from similar modalities' known failure modes
6. **Recommended metrics + benchmark tasks** — from similar modalities' evaluation conventions
7. **Uncertainty/risk notes** — what's uncertain, what might fail, confidence level
8. **Minimal viability checklist** — gates before claiming modality support:
   - [ ] Simulator produces plausible measurements
   - [ ] At least one solver converges
   - [ ] PSNR > baseline threshold
   - [ ] Calibration pipeline tested
   - [ ] Real-data format parser implemented

### L.4) Human-in-the-Loop Workflow

```
User submits bootstrap request
    ↓
Engine generates draft proposal (status: "draft")
    ↓
User reviews, edits → submits (status: "submitted")
    ↓
Admin/reviewer assigned (status: "under_review")
    ↓
Reviewer approves / requests revision / rejects
    ↓
If approved → generate initial dataset pack
    ↓
Register new modality in catalog
    ↓
All decisions recorded in review_history with timestamps
```

### L.5) Learning Loop

After real runs use a bootstrapped modality:
- Track PSNR/SSIM outcomes vs predictions
- If mismatch modes predicted correctly → boost similarity weights
- If novel failure mode discovered → add to knowledge base
- Periodically retrain/update feature vectors for embedding similarity
- Store learning events in a `bootstrap_outcomes` table

### L.6) Integration

Bootstrap outputs integrate with:
- **Metadata registry**: new modality entry in `modality_basics` + `datasets` tables
- **GCS**: templates stored under `gs://pwm-canonical/bootstrap/proposals/{id}/`
- **Manifests**: initial dataset manifest generated at approval time
- **BigQuery**: proposal metadata exported for analytics
- **RunBundle/TriadReport**: runs against bootstrapped modality tagged with `bootstrap_proposal_id`

---

## M) Modal Integration Design

### Modal App Structure

```python
# platform/modal_app.py

import modal

app = modal.App("pwm-platform")

# Shared image with PWM dependencies
pwm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pwm-core", "numpy", "scipy", "torch", "pyyaml")
    .pip_install("google-cloud-storage")
)

# --- CPU Functions (lightweight, for orchestration) ---

@app.function(image=pwm_image, cpu=2, memory=4096, timeout=300)
def validate_spec(spec_dict: dict) -> dict:
    """Validate and resolve an ExperimentSpec."""
    from pwm_core.api.endpoints import resolve_validate
    result = resolve_validate(spec_dict)
    return {"ok": result.validation.ok, "messages": [...]}


# --- GPU Functions (heavy compute) ---

@app.function(image=pwm_image, gpu="A10G", timeout=1800)
def simulate_and_reconstruct(spec_dict: dict, dataset_config: dict) -> dict:
    """Run full simulate → reconstruct → analyze pipeline on GPU."""
    from pwm_core.api.endpoints import run
    result = run(spec_dict, dataset_config)
    return result.to_dict()

@app.function(image=pwm_image, gpu="A10G", timeout=1800)
def reconstruct_only(y_data: bytes, operator_config: dict, solver: str) -> dict:
    """Reconstruct from measured data."""
    ...

@app.function(image=pwm_image, gpu="A10G", timeout=3600)
def fit_operator(y_data: bytes, x_ref: bytes, operator_config: dict) -> dict:
    """Fit operator parameters theta."""
    ...

@app.function(image=pwm_image, gpu="A10G", timeout=1800)
def calibrate_and_reconstruct(spec_dict: dict) -> dict:
    """Calibration-aware reconstruction."""
    ...

@app.function(image=pwm_image, gpu="A10G", timeout=900)
def bootstrap_simulation(proposal_dict: dict) -> dict:
    """Generate initial simulation dataset for bootstrapped modality."""
    ...
```

### Job Submission Protocol

```python
# platform/services/job_service.py

class JobService:
    """Manages job dispatch: CPU server → Modal GPU workers."""

    async def submit_run(self, run: RunBundleRecord) -> str:
        """Submit a run to the appropriate compute backend."""

        if run.compute_mode == "cpu" or self._should_run_cpu(run):
            # Run locally in arq worker
            job = await arq_queue.enqueue("run_cpu", run_id=run.run_id)
            return f"arq:{job.job_id}"

        # Dispatch to Modal
        modal_fn = self._select_modal_function(run.task_kind)
        call = modal_fn.spawn(
            spec_dict=run.experiment_spec,
            dataset_config={"dataset_id": run.dataset_id}
        )

        # Store Modal call ID for polling
        run.modal_job_id = call.object_id
        await self.run_repo.update(run)

        return f"modal:{call.object_id}"

    def _should_run_cpu(self, run: RunBundleRecord) -> bool:
        """Auto-route: small jobs → CPU, large jobs → GPU."""
        spec = run.experiment_spec
        x_shape = spec.get("sim", {}).get("x_shape", [64, 64])
        total_pixels = 1
        for d in x_shape:
            total_pixels *= d
        # Threshold: ~1M pixels → GPU
        return total_pixels < 1_000_000

    async def poll_status(self, run: RunBundleRecord) -> str:
        """Check Modal job status."""
        if run.modal_job_id:
            fn_call = modal.functions.FunctionCall.from_id(run.modal_job_id)
            try:
                result = fn_call.get(timeout=0)  # non-blocking
                return "completed"
            except TimeoutError:
                return "running"
            except Exception as e:
                return "failed"
        return run.status
```

### Result Retrieval

- **Polling**: arq background task polls Modal job status every 30s
- **Result flow**: Modal function returns dict → CPU server stores in Postgres + uploads artifacts to GCS
- **Timeout**: 30 min default, configurable per task kind
- **Retry**: Up to 2 retries on transient failures (OOM, preemption)
- **Idempotency**: run_id used as idempotency key; re-submitting same run_id is a no-op

### Cost Control

```python
# Auto-routing policy
COMPUTE_POLICY = {
    "simulate_recon_analyze": {
        "cpu_threshold": 1_000_000,   # total x pixels
        "default_gpu": "A10G",
        "fallback_gpu": "T4",
        "max_gpu_hours": 1.0,
    },
    "fit_operator_only": {
        "cpu_threshold": 500_000,
        "default_gpu": "A10G",
        "max_gpu_hours": 2.0,
    },
    "qc_report": {
        "cpu_threshold": float("inf"),  # always CPU
    },
}
```

### Environment Variables (Placeholders)

```bash
# .env (NEVER commit this file)

# Modal credentials
MODAL_TOKEN_ID=<your-modal-token-id>
MODAL_TOKEN_SECRET=<your-modal-token-secret>

# GCP credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_BUCKET=pwm-canonical
BIGQUERY_PROJECT=<your-gcp-project>
BIGQUERY_DATASET=pwm_analytics

# Database
DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/pwm
REDIS_URL=redis://localhost:6379

# Auth
SECRET_KEY=<generate-with-openssl-rand-hex-32>
CSRF_SECRET=<generate-with-openssl-rand-hex-32>
SSO_VALIDATE_URL=https://auth.comparegpt.io/api/sso/validate
SSO_REDIRECT_URL=https://auth.comparegpt.io/sso-redirect?redirect=https://pwm.platformai.org/sso/callback
```

---

## N) Deployment Plan for 34.63.169.185 / pwm.platformai.org

### Repo Layout

```
platform/
├── docker-compose.yml
├── Caddyfile
├── .env.example              # NEVER .env in git
├── alembic/                   # DB migrations
│   ├── alembic.ini
│   └── versions/
├── platform/
│   ├── __init__.py
│   ├── main.py               # FastAPI app entry
│   ├── config.py              # Settings (env vars)
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── token_manager.py   # JWT (from CompareGPT)
│   │   ├── service.py         # Auth business logic
│   │   ├── dependencies.py    # get_current_user, require_role
│   │   └── models.py          # User SQLAlchemy model
│   ├── routers/
│   │   ├── auth.py            # /api/v1/auth/*
│   │   ├── runs.py            # /api/v1/runs/*
│   │   ├── datasets.py        # /api/v1/datasets/*
│   │   ├── modalities.py      # /api/v1/modalities/*
│   │   ├── bootstrap.py       # /api/v1/bootstrap/*
│   │   └── pages.py           # Server-rendered UI pages
│   ├── services/
│   │   ├── job_service.py     # Modal dispatch + arq queue
│   │   ├── dataset_service.py
│   │   ├── bootstrap_service.py
│   │   └── similarity_engine.py
│   ├── db/
│   │   ├── database.py        # async SQLAlchemy engine
│   │   ├── models.py          # all SQLAlchemy models
│   │   ├── repo.py            # repository pattern
│   │   └── migrations/
│   ├── templates/             # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── signup.html
│   │   ├── dashboard.html
│   │   ├── run_new.html
│   │   ├── run_status.html
│   │   ├── run_results.html
│   │   ├── datasets.html
│   │   ├── modalities.html
│   │   ├── bootstrap_new.html
│   │   ├── bootstrap_review.html
│   │   └── components/
│   │       ├── navbar.html
│   │       ├── run_row.html
│   │       └── metrics_card.html
│   ├── static/
│   │   ├── css/tailwind.min.css
│   │   ├── js/htmx.min.js
│   │   └── img/logo.svg
│   ├── ingestion/             # Dataset ingestion pipeline
│   │   └── ...
│   └── modal_app.py           # Modal function definitions
├── tests/
├── scripts/
│   ├── init_db.py
│   ├── seed_modalities.py
│   └── create_admin.py
├── pyproject.toml
└── Dockerfile
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: pwm
      POSTGRES_USER: pwm
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "127.0.0.1:5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pwm"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redisdata:/data
    ports:
      - "127.0.0.1:6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  app:
    build: .
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://pwm:${DB_PASSWORD}@postgres:5432/pwm
      REDIS_URL: redis://redis:6379
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./platform:/app/platform
      - ${GOOGLE_APPLICATION_CREDENTIALS:-/dev/null}:/app/gcp-key.json:ro
    command: >
      uvicorn platform.main:app
      --host 0.0.0.0 --port 8000
      --workers 2 --log-level info

  arq_worker:
    build: .
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://pwm:${DB_PASSWORD}@postgres:5432/pwm
      REDIS_URL: redis://redis:6379
    volumes:
      - ${GOOGLE_APPLICATION_CREDENTIALS:-/dev/null}:/app/gcp-key.json:ro
    command: arq platform.workers.WorkerSettings

  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config

volumes:
  pgdata:
  redisdata:
  caddy_data:
  caddy_config:
```

### Caddyfile

```
pwm.platformai.org {
    reverse_proxy app:8000

    encode gzip

    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    log {
        output file /var/log/caddy/access.log
        format json
    }
}
```

### HTTPS/TLS Setup Steps

```bash
# 1. DNS: Point pwm.platformai.org A record to 34.63.169.185
# 2. Caddy auto-obtains Let's Encrypt certificate on first request
# 3. Auto-renews certificates (no cron needed)
# 4. Forces HTTP → HTTPS redirect automatically

# Verify DNS:
dig pwm.platformai.org +short
# Should return: 34.63.169.185

# Verify TLS:
curl -I https://pwm.platformai.org
```

### Startup Checklist

```bash
# 1. Clone and configure
git clone <repo> && cd platform
cp .env.example .env
# Edit .env with real values (SECRET_KEY, DB_PASSWORD, etc.)

# 2. Generate secrets
openssl rand -hex 32  # → SECRET_KEY
openssl rand -hex 32  # → CSRF_SECRET
openssl rand -hex 16  # → DB_PASSWORD

# 3. Start infrastructure
docker compose up -d postgres redis

# 4. Initialize database
docker compose run --rm app python scripts/init_db.py
docker compose run --rm app alembic upgrade head

# 5. Create admin user
docker compose run --rm app python scripts/create_admin.py

# 6. Seed modality knowledge base
docker compose run --rm app python scripts/seed_modalities.py

# 7. Start application
docker compose up -d app arq_worker caddy

# 8. Verify
curl https://pwm.platformai.org/api/v1/health
# → {"status": "ok", "version": "0.1.0"}

# 9. Auth deployment checks
# - Verify cookies have Secure flag (requires HTTPS)
# - Verify SameSite=Lax on auth cookies
# - Test CSRF token validation on forms
# - Verify rate limiting on /api/v1/auth/login
# - Check session store (Redis) is connected
# - Confirm CORS_ORIGINS is restrictive
```

### Logging & Monitoring

```bash
# View logs
docker compose logs -f app
docker compose logs -f arq_worker

# Basic monitoring: healthcheck endpoint
GET /api/v1/health → {"status": "ok", "db": "ok", "redis": "ok", "modal": "ok"}

# Recommended additions (Phase 2):
# - Prometheus metrics endpoint (/metrics)
# - Grafana dashboard
# - Sentry for error tracking
# - Structured JSON logging
```

### Backup Plan

```bash
# Daily Postgres backup (add to crontab)
0 3 * * * docker compose exec -T postgres pg_dump -U pwm pwm | gzip > /backups/pwm-$(date +\%Y\%m\%d).sql.gz

# Keep 30 days of backups
find /backups -name "pwm-*.sql.gz" -mtime +30 -delete

# Critical metadata also in GCS (runs, manifests)
# Redis: ephemeral (queue + cache), no backup needed
```

---

## O) Phased Rollout Plan

### Phase 1: MVP (Weeks 1-4)

**Deliverables:**
- FastAPI backend with auth (SSO + local login, CompareGPT-compatible)
- Jinja2 + HTMX UI (login, dashboard, new run, run status, results)
- PostgreSQL schema (users, runs, triad_reports, datasets)
- Redis + arq job queue
- Modal GPU integration (simulate_and_reconstruct, reconstruct_only)
- Basic GCS upload/download for RunBundles
- Minimal modality catalog (from existing modalities.yaml)
- Caddy reverse proxy with auto-TLS

**Required Services:** CPU server, Modal account, GCS bucket, domain DNS

**Risks:**
- Modal cold start latency (mitigate: keep-warm for common modalities)
- SSO provider availability (mitigate: local login fallback)

**Cost:** ~$50-100/mo (CPU server + minimal Modal + GCS)

**Success Criteria:**
- User can login, submit a CASSI run, see results
- GPU jobs complete within 10 minutes
- HTTPS works on pwm.platformai.org

### Phase 2: Team Scale (Weeks 5-10)

**Deliverables:**
- Modality bootstrap engine (knowledge base, similarity, proposal workflow)
- Dataset ingestion pipeline + CLI
- BigQuery manifest analytics
- Bootstrap review workflow (admin queue)
- Improved UI: dataset catalog, modality detail pages, bootstrap wizard
- Role-based access (user/admin/reviewer)
- Reproducibility: RunBundle provenance, config snapshots
- More modality support (CT, MRI, ptychography solvers on Modal)

**Required Services:** BigQuery, additional Modal GPU types

**Risks:**
- Knowledge base quality (mitigate: seed with existing PWM modality data)
- BigQuery cost on large scans (mitigate: partitioning, query limits)

**Cost:** ~$150-300/mo

**Success Criteria:**
- 5+ modalities fully supported
- Bootstrap engine produces usable templates
- Team of 5+ users active
- Ingestion pipeline works end-to-end

### Phase 3: Public Ecosystem (Weeks 11-20)

**Deliverables:**
- Public benchmark releases (versioned, checksummed, documented)
- Improved similarity engine (embedding-based, learning from outcomes)
- Learning loop from benchmark performance
- Richer UI: comparison views, leaderboards, uncertainty visualization
- Shared auth module (optional: extract PWM + CompareGPT auth into library)
- Governance: audit logs, data retention policies, access controls

**Required Services:** Public GCS bucket, enhanced monitoring

**Risks:**
- Public data governance (mitigate: review process, licensing)
- Scaling to many concurrent users (mitigate: Modal auto-scaling)

**Cost:** ~$300-800/mo (depends on usage)

**Success Criteria:**
- Public benchmarks available for 3+ modalities
- External users submit runs
- Bootstrap predictions improve over time

---

## P) Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Modal cold start (30-60s) | Medium | Keep-warm for popular modalities; show estimated wait in UI |
| SSO provider downtime | Medium | Local login fallback always available |
| GCS egress costs | Medium | Regional bucket, local cache on CPU server |
| Secret leakage | High | All secrets via env vars, .env in .gitignore, no defaults |
| JWT stolen (XSS) | Medium | HttpOnly cookies (not localStorage), CSP headers |
| Database loss | High | Daily backups to GCS, WAL archiving |
| Modal costs spike | Medium | Per-user quotas, auto CPU routing for small jobs |
| Knowledge base gaps | Medium | Seed from PWM's 70+ existing modality scripts |
| Single CPU server SPOF | Medium | Phase 2: add health monitoring + auto-restart |

---

## Q) Recommended MVP Build Order (Weeks 1-4)

```
Week 1: Foundation
├── Day 1-2: Repo setup, Docker Compose (Postgres + Redis + Caddy)
├── Day 3-4: FastAPI skeleton, config, DB models, Alembic migrations
├── Day 5: Auth system (token_manager, auth_service, login/signup routes)
│           Port from CompareGPT, add local login + HttpOnly cookies
└── Day 6-7: Basic Jinja2 templates (login, signup, base layout)

Week 2: Core Pipeline
├── Day 1-2: Run submission API + arq background worker
├── Day 3-4: Modal integration (simulate_and_reconstruct for CASSI)
├── Day 5: Run status polling + results display
└── Day 6-7: TriadReport storage + results page

Week 3: Data + UI
├── Day 1-2: GCS upload/download for RunBundles
├── Day 3: Dataset catalog (read from modalities.yaml + Postgres)
├── Day 4-5: Dashboard page (my runs, stats)
├── Day 6: Modality detail page
└── Day 7: Error handling, loading states, edge cases

Week 4: Hardening + Deploy
├── Day 1: DNS setup, Caddy TLS, HTTPS verification
├── Day 2: Security audit (CSRF, rate limiting, cookie flags)
├── Day 3: Seed modality knowledge base (from existing PWM data)
├── Day 4: End-to-end testing (login → submit → GPU run → results)
├── Day 5: Logging, health checks, monitoring basics
├── Day 6: Backup setup, documentation
└── Day 7: Production deploy + smoke test
```

### Security Checklist (Mandatory Before Launch)

- [ ] SECRET_KEY is unique, random, 32+ bytes
- [ ] CSRF_SECRET is unique, random
- [ ] DB_PASSWORD is unique, random, 16+ chars
- [ ] .env is in .gitignore and not committed
- [ ] CORS_ORIGINS is restricted to `["https://pwm.platformai.org"]`
- [ ] Cookies: HttpOnly=True, Secure=True, SameSite=Lax
- [ ] Rate limiting active on /api/v1/auth/login (10/min)
- [ ] HTTPS enforced (Caddy auto-redirect)
- [ ] Security headers set (HSTS, X-Content-Type-Options, X-Frame-Options)
- [ ] Modal credentials via env vars only
- [ ] GCP credentials via mounted service account key, not in code
- [ ] No debug mode in production
- [ ] Password hashing: bcrypt, 12 rounds minimum
- [ ] SQL injection: all queries via SQLAlchemy ORM (parameterized)
- [ ] XSS: Jinja2 auto-escaping enabled
- [ ] Input validation: Pydantic models on all API inputs

---

## Filestore vs GCS Decision Framework

| Use Case | Recommended | Reason |
|---|---|---|
| Canonical datasets | GCS | Durable, versioned, shared, lifecycle policies |
| RunBundle artifacts | GCS | Permanent record, signed URLs for download |
| Active preprocessing workspace | Local SSD or tmpfs | Fast random I/O during job execution |
| Model checkpoints | GCS (+ local cache) | Persist across runs, cache locally for speed |
| Redis cache | Local Redis | Ephemeral, fast access |
| Postgres data | Local volume | Docker volume, backed up to GCS daily |
| Scratch / temp files | Local /tmp | Auto-cleaned, fast |

**Rule of thumb:** If it must survive a server restart → GCS. If it's ephemeral or needs fast random access → local.

---

*This document is the complete PWM Platform architecture and implementation plan. All secrets use placeholders. Auth reuses CompareGPT-AIScientist patterns with security hardening.*
