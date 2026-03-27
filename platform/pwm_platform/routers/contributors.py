"""
Contributor economy -- role assignment, badge tracking, profiles.

Admin endpoints:
  GET  /contributors/api/all              -- list all contributors (JSON)
  GET  /contributors/api/{user_id}        -- view contributor profile (JSON)
  POST /contributors/api/{user_id}/roles  -- assign/remove roles (admin only)
  GET  /contributors/api/{user_id}/badges -- list earned badges (JSON)
  POST /contributors/api/check-badges     -- scan all users and award earned badges
  POST /contributors/api/{user_id}/record-action -- record a contribution action

Public endpoints:
  GET  /contributors/api/modality/{modality} -- list maintainers for a modality (JSON)

Page endpoints:
  GET  /contributors                      -- contributors leaderboard (HTML)
  GET  /contributors/{username}           -- contributor profile page (HTML)
  GET  /admin/roles                       -- admin role management page (HTML)
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from pwm_platform.auth.dependencies import get_current_user, get_optional_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import ContributorProfile, User

logger = logging.getLogger(__name__)

# ── File-based application store ──────────────────────────────────────
APPLICATIONS_DIR = Path("/tmp/pwm_role_applications")
APPLICATIONS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/contributors", tags=["Contributors"])

templates = Jinja2Templates(directory="pwm_platform/templates")

_BADGE_TIERS = {"none", "bronze", "silver", "gold", "platinum"}


# ── Badge definitions ────────────────────────────────────────────────────

BADGE_DEFINITIONS = {
    "first_certified": {
        "name": "First Certified",
        "description": "Got one result to Certified tier",
        "icon": "&#x1F3C6;",
        "threshold": 1,
        "field": "total_certifications",
    },
    "reproducer": {
        "name": "Reproducer",
        "description": "Reproduced 1 result independently",
        "icon": "&#x1F504;",
        "threshold": 1,
        "field": "total_reproductions",
    },
    "reproducer_10": {
        "name": "Reproducer x10",
        "description": "Reproduced 10 results",
        "icon": "&#x1F504;&#x1F504;",
        "threshold": 10,
        "field": "total_reproductions",
    },
    "claim_curator": {
        "name": "Claim Curator",
        "description": "Reviewed 5 claims",
        "icon": "&#x1F4CB;",
        "threshold": 5,
        "field": "total_claims_reviewed",
    },
    "claim_curator_25": {
        "name": "Senior Curator",
        "description": "Reviewed 25 claims",
        "icon": "&#x1F4CB;&#x2B50;",
        "threshold": 25,
        "field": "total_claims_reviewed",
    },
    "modality_maintainer": {
        "name": "Modality Maintainer",
        "description": "Maintains at least one modality",
        "icon": "&#x1F527;",
        "threshold": 1,
        "field": "maintained_modalities_count",
    },
}

VALID_ROLES = [
    "modality_maintainer",
    "dataset_steward",
    "method_integrator",
    "judge_rule_author",
    "red_team_contributor",
    "instrument_contributor",
    "claim_curator",
    "benchmark_reviewer",
]


# ── Schemas ──────────────────────────────────────────────────────────────


class RoleAssignment(BaseModel):
    roles: List[str]
    maintained_modalities: Optional[List[str]] = None


class RecordActionRequest(BaseModel):
    action: str
    details: str = ""


class ProfileUpsertRequest(BaseModel):
    bio: Optional[str] = None
    orcid: Optional[str] = None
    github_handle: Optional[str] = None
    website_url: Optional[str] = None
    modalities_contributed: Optional[list] = None
    solver_count: Optional[int] = None
    dataset_count: Optional[int] = None
    claim_count: Optional[int] = None
    verified_claim_count: Optional[int] = None
    badge_tier: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────


def _check_and_award_badges(profile: ContributorProfile) -> List[str]:
    """Check badge thresholds and award any newly earned badges."""
    existing_badges = profile.badges or []
    existing_ids = {b["badge"] for b in existing_badges}
    newly_awarded = []

    for badge_id, defn in BADGE_DEFINITIONS.items():
        if badge_id in existing_ids:
            continue
        field = defn["field"]
        if field == "maintained_modalities_count":
            value = len(profile.maintained_modalities or [])
        else:
            value = getattr(profile, field, 0) or 0

        if value >= defn["threshold"]:
            existing_badges.append({
                "badge": badge_id,
                "name": defn["name"],
                "icon": defn["icon"],
                "earned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            newly_awarded.append(badge_id)

    if newly_awarded:
        # Force SQLAlchemy to detect JSONB mutation
        profile.badges = list(existing_badges)

    return newly_awarded


async def _get_or_create_profile(
    user_id: int, db: AsyncSession
) -> ContributorProfile:
    """Fetch contributor profile by user_id, creating one if it does not exist."""
    result = await db.execute(
        select(ContributorProfile)
        .options(selectinload(ContributorProfile.user))
        .where(ContributorProfile.user_id == user_id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        profile = ContributorProfile(user_id=user_id)
        db.add(profile)
        await db.flush()
        # Re-query to eagerly load the user relationship
        result = await db.execute(
            select(ContributorProfile)
            .options(selectinload(ContributorProfile.user))
            .where(ContributorProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()
    return profile


def _profile_to_dict(profile: ContributorProfile, username: str = "", email: str = "") -> dict:
    """Serialize a ContributorProfile to a JSON-friendly dict."""
    return {
        "user_id": profile.user_id,
        "username": username,
        "email": email,
        "roles": profile.roles or [],
        "badges": profile.badges or [],
        "badge_tier": profile.badge_tier or "none",
        "maintained_modalities": profile.maintained_modalities or [],
        "modalities_contributed": profile.modalities_contributed or [],
        "contribution_history": profile.contribution_history or [],
        "total_reproductions": profile.total_reproductions or 0,
        "total_certifications": profile.total_certifications or 0,
        "total_claims_reviewed": profile.total_claims_reviewed or 0,
        "solver_count": profile.solver_count or 0,
        "dataset_count": profile.dataset_count or 0,
        "claim_count": profile.claim_count or 0,
        "verified_claim_count": profile.verified_claim_count or 0,
        "bio": profile.bio or "",
        "orcid": profile.orcid or "",
        "github_handle": profile.github_handle or "",
        "website_url": profile.website_url or "",
    }


# ═════════════════════════════════════════════════════════════════════════
#  JSON API endpoints (prefixed /contributors/api/...)
# ═════════════════════════════════════════════════════════════════════════


@router.get("/api/all")
async def api_list_contributors(db: AsyncSession = Depends(get_db)):
    """List all contributor profiles (JSON)."""
    result = await db.execute(
        select(ContributorProfile)
        .options(selectinload(ContributorProfile.user))
        .order_by(ContributorProfile.solver_count.desc())
    )
    profiles = result.scalars().all()
    contributors = []
    for p in profiles:
        uname = p.user.username if p.user else ""
        uemail = p.user.email if p.user else ""
        contributors.append(_profile_to_dict(p, uname, uemail))
    return {"contributors": contributors, "total": len(contributors)}


@router.get("/api/modality/{modality}")
async def api_get_modality_maintainers(
    modality: str,
    db: AsyncSession = Depends(get_db),
):
    """List maintainers for a specific modality (JSON)."""
    result = await db.execute(
        select(ContributorProfile).options(selectinload(ContributorProfile.user))
    )
    profiles = result.scalars().all()
    maintainers = []
    for p in profiles:
        if modality in (p.maintained_modalities or []):
            uname = p.user.username if p.user else ""
            uemail = p.user.email if p.user else ""
            maintainers.append(_profile_to_dict(p, uname, uemail))
    return {"modality": modality, "maintainers": maintainers}


# ═════════════════════════════════════════════════════════════════════════
#  Role Application endpoints (self-service)
#  NOTE: These must be defined BEFORE /api/{user_id} to avoid route conflicts.
# ═════════════════════════════════════════════════════════════════════════


@router.post("/api/apply")
async def apply_for_role(
    request: Request,
    current_user: User = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """User applies for a contributor role."""
    content_type = request.headers.get("content-type", "")
    if "form" in content_type:
        form = await request.form()
        user_id = int(form.get("user_id", 0))
        role = form.get("role", "")
        reason = form.get("reason", "")
        modalities = form.get("modalities", "")
    else:
        body = await request.json()
        user_id = body.get("user_id", 0)
        role = body.get("role", "")
        reason = body.get("reason", "")
        modalities = body.get("modalities", "")

    # Fall back to authenticated user's ID if not provided in body
    if not user_id and current_user:
        user_id = current_user.id

    if not role or not user_id:
        raise HTTPException(400, "role is required; either pass user_id or authenticate")

    if role not in VALID_ROLES:
        raise HTTPException(400, f"Invalid role: {role}. Valid: {VALID_ROLES}")

    # Look up user info
    username = ""
    email = ""
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user_obj = result.scalar_one_or_none()
        if user_obj:
            username = user_obj.username or ""
            email = user_obj.email or ""
    except Exception:
        pass

    app_id = f"app_{user_id}_{role}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    application = {
        "app_id": app_id,
        "user_id": user_id,
        "username": username,
        "email": email,
        "role": role,
        "reason": reason,
        "modalities": modalities,
        "status": "pending",
        "applied_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    (APPLICATIONS_DIR / f"{app_id}.json").write_text(json.dumps(application, indent=2))
    return application


@router.get("/api/applications")
async def list_applications(
    status: str = "all",
    db: AsyncSession = Depends(get_db),
):
    """List role applications (admin view). Enriches with user info."""
    apps = []
    # Cache user lookups
    user_cache: dict = {}
    for f in sorted(APPLICATIONS_DIR.glob("*.json"), reverse=True):
        try:
            app = json.loads(f.read_text())
            if status != "all" and app.get("status") != status:
                continue
            # Enrich with user info if missing
            uid = app.get("user_id")
            if uid and not app.get("username"):
                if uid not in user_cache:
                    try:
                        result = await db.execute(select(User).where(User.id == uid))
                        u = result.scalar_one_or_none()
                        user_cache[uid] = {"username": u.username or "", "email": u.email or ""} if u else {"username": "", "email": ""}
                    except Exception:
                        user_cache[uid] = {"username": "", "email": ""}
                app["username"] = user_cache[uid]["username"]
                app["email"] = user_cache[uid]["email"]
            apps.append(app)
        except Exception:
            continue
    return {"applications": apps, "total": len(apps)}


@router.post("/api/applications/{app_id}/approve")
async def approve_application(
    app_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Admin approves a role application -- assigns the role to the user."""
    app_path = APPLICATIONS_DIR / f"{app_id}.json"
    if not app_path.exists():
        raise HTTPException(404, "Application not found")

    app = json.loads(app_path.read_text())
    if app["status"] != "pending":
        raise HTTPException(409, f"Application already {app['status']}")

    app["status"] = "approved"
    app["approved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    app_path.write_text(json.dumps(app, indent=2))

    # Actually assign the role to the user
    user_id = app["user_id"]
    role = app["role"]
    profile = await _get_or_create_profile(user_id, db)

    # Add role if not already present
    current_roles = list(profile.roles or [])
    if role not in current_roles:
        current_roles.append(role)
        profile.roles = current_roles

    # Handle modalities for modality_maintainer
    if role == "modality_maintainer" and app.get("modalities"):
        mods = [m.strip() for m in app["modalities"].split(",") if m.strip()]
        current_mods = list(profile.maintained_modalities or [])
        for m in mods:
            if m not in current_mods:
                current_mods.append(m)
        profile.maintained_modalities = current_mods

    # Record in history
    history = list(profile.contribution_history or [])
    history.append({
        "action": "role_application_approved",
        "role": role,
        "app_id": app_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    profile.contribution_history = history

    # Check if any badges are now earned
    _check_and_award_badges(profile)

    # Auto-promote platform role if needed
    # claim_curator and benchmark_reviewer need "reviewer" platform access
    ROLES_NEEDING_REVIEWER = {"claim_curator", "benchmark_reviewer"}
    platform_promotion = None
    if role in ROLES_NEEDING_REVIEWER:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user and user.role == "user":
            user.role = "reviewer"
            platform_promotion = "reviewer"

    await db.commit()

    msg = f"Approved: {role} for user {user_id}"
    if platform_promotion:
        msg += f" (platform role auto-promoted to {platform_promotion})"

    return {"message": msg, "application": app, "platform_promotion": platform_promotion}


@router.post("/api/applications/{app_id}/reject")
async def reject_application(app_id: str):
    """Admin rejects a role application."""
    app_path = APPLICATIONS_DIR / f"{app_id}.json"
    if not app_path.exists():
        raise HTTPException(404, "Application not found")

    app = json.loads(app_path.read_text())
    if app["status"] != "pending":
        raise HTTPException(409, f"Application already {app['status']}")

    app["status"] = "rejected"
    app["rejected_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    app_path.write_text(json.dumps(app, indent=2))
    return {"message": "Application rejected", "application": app}


# ═════════════════════════════════════════════════════════════════════════
#  Contributor profile endpoints (parametric user_id routes)
# ═════════════════════════════════════════════════════════════════════════


@router.get("/api/{user_id}")
async def api_get_contributor(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """View a contributor profile by user_id (JSON)."""
    profile = await _get_or_create_profile(user_id, db)
    uname = profile.user.username if profile.user else ""
    uemail = profile.user.email if profile.user else ""
    await db.commit()
    return _profile_to_dict(profile, uname, uemail)


@router.post("/api/{user_id}/roles")
async def api_assign_roles(
    user_id: int,
    assignment: RoleAssignment,
    db: AsyncSession = Depends(get_db),
):
    """Assign roles to a user. Admin only."""
    invalid = set(assignment.roles) - set(VALID_ROLES)
    if invalid:
        raise HTTPException(400, f"Invalid roles: {invalid}. Valid: {VALID_ROLES}")

    profile = await _get_or_create_profile(user_id, db)
    profile.roles = assignment.roles

    if assignment.maintained_modalities is not None:
        profile.maintained_modalities = assignment.maintained_modalities

    # Record in history
    history = list(profile.contribution_history or [])
    history.append({
        "action": "roles_assigned",
        "roles": assignment.roles,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    profile.contribution_history = history

    # Check if any badges are now earned
    new_badges = _check_and_award_badges(profile)

    await db.commit()
    await db.refresh(profile)

    uname = profile.user.username if profile.user else ""
    uemail = profile.user.email if profile.user else ""
    return {
        "profile": _profile_to_dict(profile, uname, uemail),
        "newly_awarded_badges": new_badges,
    }


@router.get("/api/{user_id}/badges")
async def api_get_badges(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List earned badges for a user (JSON)."""
    profile = await _get_or_create_profile(user_id, db)
    await db.commit()
    return {"badges": profile.badges or [], "available": BADGE_DEFINITIONS}


@router.post("/api/{user_id}/record-action")
async def api_record_action(
    user_id: int,
    body: RecordActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Record a contribution action (used by other endpoints to track badges)."""
    profile = await _get_or_create_profile(user_id, db)

    # Increment counters based on action
    if body.action == "reproduced":
        profile.total_reproductions = (profile.total_reproductions or 0) + 1
    elif body.action == "certified":
        profile.total_certifications = (profile.total_certifications or 0) + 1
    elif body.action == "reviewed_claim":
        profile.total_claims_reviewed = (profile.total_claims_reviewed or 0) + 1

    history = list(profile.contribution_history or [])
    history.append({
        "action": body.action,
        "details": body.details,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    profile.contribution_history = history

    new_badges = _check_and_award_badges(profile)

    await db.commit()
    await db.refresh(profile)

    uname = profile.user.username if profile.user else ""
    uemail = profile.user.email if profile.user else ""
    return {
        "profile": _profile_to_dict(profile, uname, uemail),
        "newly_awarded_badges": new_badges,
    }


@router.post("/api/check-maturity")
async def api_check_maturity(db: AsyncSession = Depends(get_db)):
    """Auto-compute maturity level for all contributor profiles.

    Level 0: Account exists
    Level 1: At least one contributor role approved
    Level 2: 3+ contributions in history
    Level 3: claim_curator or benchmark_reviewer + 5+ reproductions OR 10+ claims reviewed
    Level 4: Modality Maintainer + 5+ certifications (sustained contribution)
    """
    import datetime as _dt
    result = await db.execute(
        select(ContributorProfile).options(selectinload(ContributorProfile.user))
    )
    profiles = result.scalars().all()
    updates = {}
    for p in profiles:
        roles = p.roles or []
        history_len = len(p.contribution_history or [])
        reproductions = p.total_reproductions or 0
        claims_reviewed = p.total_claims_reviewed or 0
        certifications = p.total_certifications or 0

        if "modality_maintainer" in roles and certifications >= 5:
            level = 4
        elif ("claim_curator" in roles or "benchmark_reviewer" in roles) and (reproductions >= 5 or claims_reviewed >= 10):
            level = 3
        elif history_len >= 3:
            level = 2
        elif len(roles) >= 1:
            level = 1
        else:
            level = 0

        old_level = p.badge_tier  # reuse badge_tier field as proxy (or add a new field)
        # Store maturity level in contribution_history metadata (lightweight approach)
        # Update bio_extras via card_data or just track in a comment
        # For now, append a maturity check event to history if level changed
        history = list(p.contribution_history or [])
        last_level_entry = next((h for h in reversed(history) if h.get("action") == "maturity_check"), None)
        last_level = last_level_entry.get("level", 0) if last_level_entry else 0
        if level != last_level:
            history.append({
                "action": "maturity_check",
                "level": level,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            })
            p.contribution_history = history
            updates[p.user_id] = {"old_level": last_level, "new_level": level}

    await db.commit()
    return {"updated": updates, "total_profiles": len(profiles)}


@router.post("/api/check-badges")
async def api_check_all_badges(db: AsyncSession = Depends(get_db)):
    """Scan all profiles and award any earned badges."""
    result = await db.execute(
        select(ContributorProfile).options(selectinload(ContributorProfile.user))
    )
    profiles = result.scalars().all()
    results = {}
    for profile in profiles:
        new = _check_and_award_badges(profile)
        if new:
            results[profile.user_id] = new

    await db.commit()
    return {"awarded": results}


# ── Profile upsert (legacy) ─────────────────────────────────────────────


@router.post("/profile", response_model=dict)
async def upsert_profile(
    body: ProfileUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create or update the authenticated user's ContributorProfile."""
    result = await db.execute(
        select(ContributorProfile).where(ContributorProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()

    if profile is None:
        profile = ContributorProfile(user_id=current_user.id)
        db.add(profile)

    if body.bio is not None:
        profile.bio = body.bio
    if body.orcid is not None:
        profile.orcid = body.orcid or None
    if body.github_handle is not None:
        profile.github_handle = body.github_handle or None
    if body.website_url is not None:
        profile.website_url = body.website_url or None
    if body.modalities_contributed is not None:
        profile.modalities_contributed = body.modalities_contributed
    if body.solver_count is not None:
        profile.solver_count = body.solver_count
    if body.dataset_count is not None:
        profile.dataset_count = body.dataset_count
    if body.claim_count is not None:
        profile.claim_count = body.claim_count
    if body.verified_claim_count is not None:
        profile.verified_claim_count = body.verified_claim_count
    if body.badge_tier is not None:
        if body.badge_tier not in _BADGE_TIERS:
            raise HTTPException(
                status_code=422,
                detail=f"badge_tier must be one of: {sorted(_BADGE_TIERS)}",
            )
        profile.badge_tier = body.badge_tier

    await db.commit()
    await db.refresh(profile)

    logger.info(
        "ContributorProfile upserted for user %s (id=%s)",
        current_user.username,
        current_user.id,
    )

    return {
        "success": True,
        "user_id": profile.user_id,
        "badge_tier": profile.badge_tier,
        "solver_count": profile.solver_count,
        "dataset_count": profile.dataset_count,
    }


# ═════════════════════════════════════════════════════════════════════════
#  HTML page endpoints
# ═════════════════════════════════════════════════════════════════════════


@router.get("/apply", response_class=HTMLResponse)
async def apply_role_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """Render the role application form page."""
    if not user:
        return HTMLResponse(
            status_code=302,
            headers={"Location": "/login"},
        )
    return templates.TemplateResponse(
        request,
        "apply_role.html",
        {"user": user, "valid_roles": VALID_ROLES},
    )


@router.get("", response_class=HTMLResponse)
async def list_contributors_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Render the contributors leaderboard page (top contributors by solver_count)."""
    stmt = (
        select(ContributorProfile)
        .options(selectinload(ContributorProfile.user))
        .order_by(ContributorProfile.solver_count.desc())
        .limit(50)
    )
    result = await db.execute(stmt)
    profiles = result.scalars().all()

    return templates.TemplateResponse(
        request,
        "contributors.html",
        {
            "user": user,
            "profiles": profiles,
            "badge_definitions": BADGE_DEFINITIONS,
            "valid_roles": VALID_ROLES,
        },
    )


@router.get("/{username}", response_class=HTMLResponse)
async def contributor_profile_page(
    username: str,
    request: Request,
    current_user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Render the public profile page for a contributor."""
    user_result = await db.execute(select(User).where(User.username == username))
    profile_user = user_result.scalar_one_or_none()
    if profile_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    profile_result = await db.execute(
        select(ContributorProfile).where(
            ContributorProfile.user_id == profile_user.id
        )
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        # Auto-create an empty profile for any user
        profile = ContributorProfile(user_id=profile_user.id)
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return templates.TemplateResponse(
        request,
        "contributor_profile.html",
        {
            "user": current_user,
            "profile_user": profile_user,
            "profile": profile,
            "badge_definitions": BADGE_DEFINITIONS,
            "valid_roles": VALID_ROLES,
        },
    )
