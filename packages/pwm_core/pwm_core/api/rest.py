"""pwm_core.api.rest

FastAPI REST API exposing core pwm CLI commands over HTTP.

Endpoints
---------
GET  /api/v1/health
GET  /api/v1/primitives?domain=general
GET  /api/v1/modalities
GET  /api/v1/certificate/{run_id}
POST /api/v1/evaluate
POST /api/v1/scaffold-claim
POST /api/v1/synthesize
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fastapi and pydantic are required for pwm_core.api.rest. "
        "Install with: pip install fastapi uvicorn pydantic"
    ) from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for pwm_core.api.rest. "
        "Install with: pip install pyyaml"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERSION = "1.0.0"

_KNOWN_MODALITIES = [
    "ct",
    "mri",
    "cassi",
    "cacti",
    "pet",
    "ultrasound",
    "oct",
    "hyperspectral",
    "acoustics",
    "combustion",
    "particle_physics",
    "remote_sensing",
    "generic",
]

# Repo root is four levels above this file:
# packages/pwm_core/pwm_core/api/rest.py → repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]

_PRIMITIVE_REGISTRY_DIR = _REPO_ROOT / "contrib" / "primitive_registry"
_RUNS_DIR = _REPO_ROOT / "runs"

# ---------------------------------------------------------------------------
# Request body schemas
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    modality: str
    solver: str
    track: str = "correct"
    n_scenes: int = 5


class ScaffoldClaimRequest(BaseModel):
    arxiv_id: str
    modality: str = ""
    out: str = "claims"


class SynthesizeRequest(BaseModel):
    modality: str
    n_samples: int = 10
    tier: str = "public"
    out_dir: str = "datasets"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application instance."""

    app = FastAPI(
        title="Physics World Model API",
        description="REST interface to the pwm CLI toolkit.",
        version=_VERSION,
    )

    # ------------------------------------------------------------------
    # GET /api/v1/health
    # ------------------------------------------------------------------

    @app.get("/api/v1/health")
    def health() -> dict:
        """Return service health status."""
        return {"status": "ok", "version": _VERSION}

    # ------------------------------------------------------------------
    # GET /api/v1/primitives
    # ------------------------------------------------------------------

    @app.get("/api/v1/primitives")
    def list_primitives(domain: str = "general") -> dict:
        """List primitives from the primitive registry for a given domain."""
        primitives_file = _PRIMITIVE_REGISTRY_DIR / domain / "v1" / "primitives.yaml"
        if not primitives_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Primitive registry not found for domain '{domain}'. "
                       f"Expected path: {primitives_file}",
            )
        with primitives_file.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return {"domain": domain, "primitives": data}

    # ------------------------------------------------------------------
    # GET /api/v1/modalities
    # ------------------------------------------------------------------

    @app.get("/api/v1/modalities")
    def list_modalities() -> dict:
        """Return the list of known modalities."""
        return {"modalities": _KNOWN_MODALITIES}

    # ------------------------------------------------------------------
    # GET /api/v1/certificate/{run_id}
    # ------------------------------------------------------------------

    @app.get("/api/v1/certificate/{run_id}")
    def get_certificate(run_id: str) -> dict:
        """Search runs/ directory for a matching bundle and return its certificate."""
        if not _RUNS_DIR.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Runs directory does not exist: {_RUNS_DIR}",
            )

        # Walk all subdirectories looking for a directory whose name contains run_id
        # and which holds a certificate.json.
        for candidate in sorted(_RUNS_DIR.rglob("certificate.json")):
            bundle_dir = candidate.parent
            if run_id in bundle_dir.name or run_id in str(bundle_dir.relative_to(_RUNS_DIR)):
                with candidate.open("r", encoding="utf-8") as fh:
                    cert = json.load(fh)
                return {"run_id": run_id, "bundle_dir": str(bundle_dir), "certificate": cert}

        raise HTTPException(
            status_code=404,
            detail=f"No certificate found for run_id '{run_id}' under {_RUNS_DIR}",
        )

    # ------------------------------------------------------------------
    # POST /api/v1/evaluate
    # ------------------------------------------------------------------

    @app.post("/api/v1/evaluate")
    def evaluate(req: EvaluateRequest) -> dict:
        """Submit an evaluation job by running `pwm evaluate` as a subprocess."""
        run_id = str(uuid.uuid4())[:8]
        cmd = [
            "pwm", "evaluate",
            "--modality", req.modality,
            "--solver", req.solver,
            "--track", req.track,
            "--n-scenes", str(req.n_scenes),
            "--run-id", run_id,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(_REPO_ROOT),
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="'pwm' CLI not found. Ensure the package is installed.",
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Evaluation timed out after 600 s.",
            )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"pwm evaluate failed (rc={result.returncode}): {result.stderr.strip()}",
            )

        return {
            "status": "submitted",
            "run_id": run_id,
            "message": result.stdout.strip() or "Evaluation completed.",
        }

    # ------------------------------------------------------------------
    # POST /api/v1/scaffold-claim
    # ------------------------------------------------------------------

    @app.post("/api/v1/scaffold-claim")
    def scaffold_claim(req: ScaffoldClaimRequest) -> dict:
        """Run `pwm scaffold-claim` as a subprocess."""
        cmd = ["pwm", "scaffold-claim", req.arxiv_id, "--out", req.out]
        if req.modality:
            cmd += ["--modality", req.modality]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(_REPO_ROOT),
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="'pwm' CLI not found. Ensure the package is installed.",
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="scaffold-claim timed out after 120 s.",
            )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"pwm scaffold-claim failed (rc={result.returncode}): {result.stderr.strip()}",
            )

        return {"status": "ok", "output": result.stdout.strip()}

    # ------------------------------------------------------------------
    # POST /api/v1/synthesize
    # ------------------------------------------------------------------

    @app.post("/api/v1/synthesize")
    def synthesize(req: SynthesizeRequest) -> dict:
        """Run `pwm synthesize` as a subprocess."""
        cmd = [
            "pwm", "synthesize",
            "--modality", req.modality,
            "--n-samples", str(req.n_samples),
            "--tier", req.tier,
            "--out-dir", req.out_dir,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(_REPO_ROOT),
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="'pwm' CLI not found. Ensure the package is installed.",
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="synthesize timed out after 300 s.",
            )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"pwm synthesize failed (rc={result.returncode}): {result.stderr.strip()}",
            )

        return {"status": "ok", "output": result.stdout.strip()}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the REST API server with uvicorn on port 8080."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run the pwm REST API. "
            "Install with: pip install uvicorn"
        ) from exc

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
