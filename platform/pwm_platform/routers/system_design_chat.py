"""System Design Chat Router — interactive system design via Plan/Judge/Performance agents.

Prefix: /api/v1/system-design

Endpoints:
  POST /chat                  — Send a message (generates or refines spec, auto-judges)
  POST /run                   — Run the Performance Agent on the current spec
  POST /example/{example_id}  — Load a pre-built multi-round example
  GET  /spec/{session_id}     — Fetch plan.md or spec.md for the sidebar
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_optional_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User
from pwm_platform.services.gemini_client import (
    append_to_conversation,
    create_conversation,
    get_conversation,
    get_session_dataset_meta,
    update_session_dataset_meta,
)
from pwm_platform.services.system_design_agent import (
    detect_modality,
    detect_period,
    generate_plan,
    judge_plan,
    refine_plan,
    run_performance,
)
from pwm_platform.services.system_design_examples import (
    get_example_by_id,
    get_examples,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/system-design", tags=["System Design"])
templates = Jinja2Templates(directory="pwm_platform/templates")


# ── Chat ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(""),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Process a chat message — generate or refine a system design spec."""

    # Create or load session
    if not session_id:
        session_id = await create_conversation(
            db, user_id=user.id if user else None, variant_key="system_design"
        )

    history = await get_conversation(db, session_id) or []

    # Load current state
    meta_raw, _, _ = await get_session_dataset_meta(db, session_id)
    state: dict = meta_raw or {}
    current_spec = state.get("spec")
    period = state.get("period")
    modality = state.get("modality")

    spec_judged = False
    judge_text = ""
    plan_md = ""
    spec_md = ""

    try:
        if current_spec:
            spec, description, plan_md, spec_md = await refine_plan(
                message, current_spec, period, history
            )
        else:
            if not period:
                period = detect_period(message)
            if not modality:
                modality = detect_modality(message)
            spec, description, plan_md, spec_md = await generate_plan(
                message, period, modality
            )

        # Auto-judge
        judgment, judge_text = await judge_plan(spec, period)
        spec_judged = judgment.get("feasible", False)

        # Persist
        state = {
            "spec": spec,
            "judgment": judgment,
            "period": period,
            "modality": modality,
            "spec_judged": spec_judged,
            "plan_md": plan_md,
            "spec_md": spec_md,
        }
        await update_session_dataset_meta(db, session_id, dataset_meta=state)
        await append_to_conversation(db, session_id, "user", message)
        await append_to_conversation(
            db, session_id, "model", f"{description}\n\n---\n\n{judge_text}"
        )

    except Exception as exc:
        logger.error("System design chat error: %s", exc, exc_info=True)
        description = f"Error generating plan: {type(exc).__name__}: {exc}"

    return templates.TemplateResponse("_sysdesign_message.html", {
        "request": request,
        "user_message": message,
        "assistant_text": description,
        "judge_text": judge_text,
        "session_id": session_id,
        "spec_judged": spec_judged,
        "period": state.get("period", ""),
        "modality": state.get("modality", ""),
    })


# ── Run Performance ────────────────────────────────────────────────────────

@router.post("/run", response_class=HTMLResponse)
async def run(
    request: Request,
    session_id: str = Form(...),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the Performance Agent on the current spec."""

    meta_raw, _, _ = await get_session_dataset_meta(db, session_id)
    state: dict = meta_raw or {}

    if not state.get("spec"):
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg '
            'bg-red-50 border border-red-200 text-sm text-red-700">'
            "No spec found. Please design a system first.</div></div>"
        )

    if not state.get("spec_judged"):
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg '
            'bg-amber-50 border border-amber-200 text-sm text-amber-700">'
            "Spec has not passed judge validation. Refine the design first.</div></div>"
        )

    try:
        analysis = await run_performance(state["spec"], state.get("period", "forward"))
    except Exception as exc:
        logger.error("Performance agent error: %s", exc, exc_info=True)
        analysis = f"Performance analysis failed: {type(exc).__name__}: {exc}"

    await append_to_conversation(db, session_id, "user", "[Run Performance Analysis]")
    await append_to_conversation(db, session_id, "model", analysis)

    return templates.TemplateResponse("_sysdesign_message.html", {
        "request": request,
        "user_message": "Run Performance Analysis",
        "assistant_text": analysis,
        "judge_text": "",
        "session_id": session_id,
        "spec_judged": state.get("spec_judged", False),
        "period": state.get("period", ""),
        "modality": state.get("modality", ""),
        "is_performance": True,
    })


# ── Load Example ────────────────────────────────────────────────────────────

@router.post("/example/{example_id}", response_class=HTMLResponse)
async def load_example(
    request: Request,
    example_id: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Load a pre-built multi-round example into the chat."""

    example = get_example_by_id(example_id)
    if not example:
        return HTMLResponse(
            '<div class="text-sm text-red-600 p-2">Example not found.</div>',
            status_code=404,
        )

    return templates.TemplateResponse("_sysdesign_example.html", {
        "request": request,
        "example": example,
    })


# ── Get Spec ────────────────────────────────────────────────────────────────

@router.get("/spec/{session_id}", response_class=HTMLResponse)
async def get_spec(
    request: Request,
    session_id: str,
    view: str = "plan",
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current plan.md or spec.md for the sidebar."""

    meta_raw, _, _ = await get_session_dataset_meta(db, session_id)
    state: dict = meta_raw or {}
    content = state.get("plan_md", "") if view == "plan" else state.get("spec_md", "")

    return templates.TemplateResponse("_sysdesign_spec.html", {
        "request": request,
        "content": content,
        "view": view,
        "session_id": session_id,
        "spec_judged": state.get("spec_judged", False),
        "period": state.get("period", ""),
        "modality": state.get("modality", ""),
    })
