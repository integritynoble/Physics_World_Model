"""
Submissions Router — competition & contribution submission system.

Competition flow:
  1. Public tier  — contestant downloads public data, tests locally, reports score
  2. Dev tier     — contestant submits reconstruction (x_hat), PWM returns score
  3. Hidden tier  — contestant submits algorithm, PWM runs it on hidden data (costs credits)

Contribution flow:
  - Users submit datasets for any tier (public, dev, hidden)

Admin review:
  - platformaigpt@gmail.com can see all submissions and approve/reject with feedback

Endpoints:
  GET   /api/v1/submissions/form               — inline HTMX form partial
  POST  /api/v1/submissions/                    — create submission (multipart/form-data)
  POST  /api/v1/submissions/score-report        — create score report (application/json)
  POST  /api/v1/submissions/report-score        — report public tier score
  PATCH /api/v1/submissions/{id}/review         — admin approve/reject with feedback
  PATCH /api/v1/submissions/{id}/set-score      — admin sets score for dev/hidden result
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user, require_role
from pwm_platform.db.database import get_db
from pwm_platform.db.models import ChallengeSubmission, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/submissions", tags=["Submissions"])
templates = Jinja2Templates(directory="pwm_platform/templates")

# ── Constants ──────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".h5", ".hdf5", ".py", ".zip", ".tar.gz", ".npy", ".npz"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR = Path("pwm_platform/static/uploads/submissions")

HIDDEN_TIER_CREDIT_COST = 10.0  # credits charged for hidden tier evaluation

# Validation: submission_type → allowed tiers
_TYPE_TIER_RULES = {
    "score_report":    {"public"},
    "reconstruction":  {"dev"},
    "algorithm":       {"hidden"},
    "dataset":         {"public", "dev", "hidden"},
    "solver":          {"public", "dev", "hidden"},
}


def _allowed_extension(filename: str) -> bool:
    """Check if file extension is allowed (handles .tar.gz)."""
    if filename.endswith(".tar.gz"):
        return True
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


# ── GET /form — inline HTMX form partial ─────────────────────────────────


@router.get("/form", response_class=HTMLResponse)
async def submission_form(
    request: Request,
    variant_key: str = "",
    tier_name: str = "",
    submission_type: str = "",
    category: str = "competition",
    user: User | None = Depends(get_optional_user),
):
    """Return the inline submission form partial (or sign-in prompt)."""
    if user is None:
        return templates.TemplateResponse(request, "_submission_signin.html", {
            "tier_name": tier_name,
            "submission_type": submission_type,
        })
    return templates.TemplateResponse(request, "_submission_form.html", {
        "user": user,
        "variant_key": variant_key,
        "tier_name": tier_name,
        "submission_type": submission_type,
        "category": category,
        "credit_balance": user.credit_balance or 0,
        "hidden_credit_cost": HIDDEN_TIER_CREDIT_COST,
    })


# ── POST / — create submission (competition or contribution) ─────────────


@router.post("/", response_class=HTMLResponse)
async def create_submission(
    request: Request,
    variant_key: str = Form(...),
    tier_name: str = Form(...),
    submission_type: str = Form(...),
    category: str = Form("competition"),
    method_name: str = Form(""),
    method_description: str = Form(""),
    paper_url: str = Form(""),
    code_url: str = Form(""),
    reported_psnr: str = Form(""),
    reported_ssim: str = Form(""),
    file: UploadFile = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new submission (competition entry or dataset contribution)."""
    # Validate submission_type / tier combination
    allowed_tiers = _TYPE_TIER_RULES.get(submission_type)
    if allowed_tiers is None:
        raise HTTPException(400, f"Invalid submission type: {submission_type}")
    if tier_name not in allowed_tiers:
        raise HTTPException(400, f"{submission_type} submissions are not allowed on {tier_name} tier")

    # Auto-generate method_name for solver uploads if not provided
    if not method_name.strip() and submission_type == "solver" and file and file.filename:
        method_name = os.path.splitext(file.filename)[0]
    if not method_name.strip():
        raise HTTPException(400, "Method name is required")

    # Score report doesn't need a file
    is_score_report = submission_type == "score_report"

    if not is_score_report:
        # Validate file
        if file is None or file.filename == "":
            raise HTTPException(400, f"A file is required. Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        if not _allowed_extension(file.filename):
            raise HTTPException(400, f"File type not allowed. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # Credit check for hidden tier algorithm submission
    credit_cost = 0.0
    if submission_type == "algorithm" and tier_name == "hidden":
        balance = user.credit_balance or 0
        if balance < HIDDEN_TIER_CREDIT_COST:
            raise HTTPException(
                400,
                f"Insufficient credits. Hidden tier evaluation costs {HIDDEN_TIER_CREDIT_COST} credits. "
                f"Your balance: {balance:.1f} credits."
            )
        credit_cost = HIDDEN_TIER_CREDIT_COST

    # Generate submission ID
    submission_id = f"sub-{uuid.uuid4().hex[:12]}"

    # Handle file save (or placeholder for score reports)
    file_path_str = ""
    original_filename = ""
    file_size = 0

    if not is_score_report:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)} MB")

        save_dir = UPLOAD_DIR / submission_id
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file.filename
        save_path.write_bytes(contents)
        file_path_str = str(save_path)
        original_filename = file.filename
        file_size = len(contents)
    else:
        file_path_str = "score_report"
        original_filename = "score_report"

    # Build scores dict for score reports
    scores = None
    if is_score_report:
        scores = {}
        if reported_psnr:
            try:
                scores["psnr"] = float(reported_psnr)
            except ValueError:
                pass
        if reported_ssim:
            try:
                scores["ssim"] = float(reported_ssim)
            except ValueError:
                pass

    # Deduct credits for hidden tier
    if credit_cost > 0:
        user.credit_balance = (user.credit_balance or 0) - credit_cost

    # Create DB record
    submission = ChallengeSubmission(
        submission_id=submission_id,
        submission_type=submission_type,
        category=category,
        variant_key=variant_key,
        tier_name=tier_name,
        credit_cost=credit_cost,
        submitted_by=user.id,
        method_name=method_name.strip(),
        method_description=method_description.strip(),
        paper_url=paper_url.strip(),
        code_url=code_url.strip(),
        file_path=file_path_str,
        original_filename=original_filename,
        file_size_bytes=file_size,
        scores=scores,
        status="pending",
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    logger.info(
        "Submission %s created by user %s for %s/%s (%s, %s, cost=%.1f)",
        submission_id, user.username, variant_key, tier_name,
        submission_type, category, credit_cost,
    )

    return templates.TemplateResponse(request, "_submission_success.html", {
        "submission": submission,
    })


# ── POST /score-report — JSON-compatible score report submission ──────────


class ScoreReportRequest(BaseModel):
    variant_key: str
    tier_name: str = "public"
    method_name: str
    method_description: str = ""
    paper_url: str = ""
    code_url: str = ""
    reported_psnr: float | None = None
    reported_ssim: float | None = None
    category: str = "competition"


@router.post("/score-report")
async def create_score_report_json(
    body: ScoreReportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a score_report submission from a JSON body (programmatic API)."""
    allowed_tiers = _TYPE_TIER_RULES["score_report"]
    if body.tier_name not in allowed_tiers:
        raise HTTPException(400, f"score_report submissions are only allowed on {allowed_tiers} tiers")

    if not body.method_name.strip():
        raise HTTPException(400, "method_name is required")

    submission_id = f"sub-{uuid.uuid4().hex[:12]}"

    scores: dict = {}
    if body.reported_psnr is not None:
        scores["psnr"] = body.reported_psnr
    if body.reported_ssim is not None:
        scores["ssim"] = body.reported_ssim

    submission = ChallengeSubmission(
        submission_id=submission_id,
        submission_type="score_report",
        category=body.category,
        variant_key=body.variant_key,
        tier_name=body.tier_name,
        credit_cost=0.0,
        submitted_by=user.id,
        method_name=body.method_name.strip(),
        method_description=body.method_description.strip(),
        paper_url=body.paper_url.strip(),
        code_url=body.code_url.strip(),
        file_path="score_report",
        original_filename="score_report",
        file_size_bytes=0,
        scores=scores or None,
        status="pending",
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    logger.info(
        "Score report %s created by user %s for %s/%s",
        submission_id, user.username, body.variant_key, body.tier_name,
    )

    return JSONResponse({
        "submission_id": submission.submission_id,
        "status": submission.status,
        "variant_key": submission.variant_key,
        "tier_name": submission.tier_name,
        "method_name": submission.method_name,
        "scores": submission.scores,
        "submitted_at": submission.submitted_at.isoformat() if submission.submitted_at else None,
    })


# ── PATCH /{submission_id}/review — admin review ─────────────────────────


@router.patch("/{submission_id}/review", response_class=HTMLResponse)
async def review_submission(
    request: Request,
    submission_id: str,
    decision: str = Form(...),
    notes: str = Form(""),
    score_psnr: str = Form(""),
    score_ssim: str = Form(""),
    score_composite: str = Form(""),
    user: User = Depends(require_role("admin", "reviewer")),
    db: AsyncSession = Depends(get_db),
):
    """Admin/reviewer approves or rejects a submission, optionally setting scores."""
    if decision not in ("approved", "rejected"):
        raise HTTPException(400, "Decision must be 'approved' or 'rejected'")

    result = await db.execute(
        select(ChallengeSubmission).where(ChallengeSubmission.submission_id == submission_id)
    )
    submission = result.scalar_one_or_none()
    if submission is None:
        raise HTTPException(404, "Submission not found")

    submission.status = decision
    submission.reviewer_id = user.id
    submission.review_notes = notes.strip()
    submission.reviewed_at = datetime.now(timezone.utc)

    # Set scores if provided (admin evaluated the submission)
    scores = submission.scores or {}
    if score_psnr:
        try:
            scores["psnr"] = float(score_psnr)
        except ValueError:
            pass
    if score_ssim:
        try:
            scores["ssim"] = float(score_ssim)
        except ValueError:
            pass
    if score_composite:
        try:
            scores["composite"] = float(score_composite)
        except ValueError:
            pass
    if scores:
        submission.scores = scores

    # Refund credits if rejected on hidden tier
    if decision == "rejected" and submission.credit_cost > 0:
        submitter_result = await db.execute(
            select(User).where(User.id == submission.submitted_by)
        )
        submitter = submitter_result.scalar_one_or_none()
        if submitter:
            submitter.credit_balance = (submitter.credit_balance or 0) + submission.credit_cost
            logger.info("Refunded %.1f credits to user %s", submission.credit_cost, submitter.username)

    await db.commit()
    await db.refresh(submission)

    logger.info(
        "Submission %s %s by %s", submission_id, decision, user.username,
    )

    # Best-effort GCS upload on approval
    if decision == "approved" and submission.file_path != "score_report":
        try:
            _upload_to_gcs(submission)
        except Exception:
            logger.warning("GCS upload failed for %s (non-fatal)", submission_id, exc_info=True)

    return templates.TemplateResponse(request, "_submission_review_card.html", {
        "s": submission,
    })


def _upload_to_gcs(submission: ChallengeSubmission) -> None:
    """Best-effort upload of approved submission to GCS."""
    try:
        from google.cloud import storage
    except ImportError:
        logger.debug("google-cloud-storage not installed, skipping GCS upload")
        return

    bucket_name = "pwm-benchmark-datasets"
    gcs_object = (
        f"submissions/v1.0/{submission.variant_key}/{submission.tier_name}"
        f"/{submission.submission_id}/{submission.original_filename}"
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_object)
    blob.upload_from_filename(submission.file_path)
    logger.info("Uploaded %s to gs://%s/%s", submission.submission_id, bucket_name, gcs_object)
