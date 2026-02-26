"""Spec Chat Router — interactive spec builder powered by Gemini 2.5 Flash.

Prefix: /api/v1/spec-chat

Endpoints:
  POST /{variant_key}           — Process a chat message (create / continue conversation)
  POST /{variant_key}/example   — Load a pre-built example spec (cassi, spc, cacti)
  POST /{variant_key}/simulate  — Run simulation on a spec JSON
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_optional_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User
from pwm_platform.services.benchmark_database import get_variant, get_spec_primitives
from pwm_platform.services.gemini_client import (
    append_to_conversation,
    call_gemini,
    create_conversation,
    get_conversation,
)
from pwm_platform.services.spec_chat_prompts import (
    build_system_prompt,
    get_example_spec,
    parse_spec_from_response,
)
from pwm_platform.services.spec_simulator import run_spec_simulation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/spec-chat", tags=["Spec Chat"])
templates = Jinja2Templates(directory="pwm_platform/templates")


@router.post("/{variant_key}", response_class=HTMLResponse)
async def chat_message(
    request: Request,
    variant_key: str,
    message: str = Form(...),
    session_id: str = Form(""),
    input_mode: str = Form("describe"),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Process a user chat message and return an HTMX partial."""
    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    user_id = user.id if user else None

    # Create or retrieve session
    if not session_id:
        session_id = await create_conversation(db, user_id=user_id, variant_key=variant_key)
    else:
        history = await get_conversation(db, session_id)
        if history is None:
            # Session not found — start fresh
            session_id = await create_conversation(db, user_id=user_id, variant_key=variant_key)

    # Build user message based on input mode
    if input_mode == "upload_spec":
        user_text = (
            "Please validate and describe the following spec JSON, then suggest "
            "improvements if any:\n\n" + message
        )
    else:
        user_text = message

    # Append user turn
    await append_to_conversation(db, session_id, "user", user_text)

    # Get full history for Gemini call
    history = await get_conversation(db, session_id)
    system_prompt = build_system_prompt(variant)

    try:
        response_text = await call_gemini(system_prompt, history)
    except Exception as exc:
        logger.error("Gemini API error: %s", exc, exc_info=True)
        # Remove the failed user turn so they can retry
        # (the DB row already has it, but we can leave it — next attempt will
        # just add another user turn which is fine for context)
        return templates.TemplateResponse("_spec_chat_message.html", {
            "request": request,
            "session_id": session_id,
            "variant_key": variant_key,
            "user_message": message,
            "assistant_text": f"Sorry, I encountered an error calling the AI service. Please try again. ({type(exc).__name__})",
            "spec": None,
            "primitives": get_spec_primitives(),
        })

    # Append assistant turn
    await append_to_conversation(db, session_id, "model", response_text)

    # Parse spec from response
    explanation, spec = parse_spec_from_response(response_text)

    return templates.TemplateResponse("_spec_chat_message.html", {
        "request": request,
        "session_id": session_id,
        "variant_key": variant_key,
        "user_message": message,
        "assistant_text": explanation,
        "spec": spec,
        "primitives": get_spec_primitives(),
    })


@router.post("/{variant_key}/example", response_class=HTMLResponse)
async def load_example(
    request: Request,
    variant_key: str,
    example: str = Form(...),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Load an example spec and start a new chat session."""
    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    example_spec = get_example_spec(example)
    if example_spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown example: {example}")

    user_id = user.id if user else None

    # Create a new session
    session_id = await create_conversation(db, user_id=user_id, variant_key=variant_key)

    # Build a synthetic first exchange
    user_text = f"Show me the {example_spec['label']} spec."
    spec_json = {
        "spec_notation": example_spec["spec_notation"],
        "forward_model": example_spec["forward_model"],
        "mismatch_params": example_spec["mismatch_params"],
        "noise_model": example_spec["noise_model"],
        "measurement_matrix": example_spec["measurement_matrix"],
    }

    # Build a synthetic assistant response
    dag_str = " → ".join(
        f"{n['primitive']}({n['params']})" if n.get("params") else n["primitive"]
        for n in example_spec["forward_model"]
    )
    assistant_text = (
        f"Here is the **{example_spec['label']}** spec.\n\n"
        f"The forward model pipeline is: {dag_str}\n\n"
        f"**Noise model:** {example_spec['noise_model']}\n\n"
        f"**Measurement matrix:** {example_spec['measurement_matrix']}\n\n"
        f"You can ask me to modify any part of this spec — change the noise model, "
        f"add or remove primitives, adjust mismatch parameters, etc."
    )

    full_response = assistant_text + "\n\n```json\n" + json.dumps(spec_json, indent=2) + "\n```"

    # Store in conversation history
    await append_to_conversation(db, session_id, "user", user_text)
    await append_to_conversation(db, session_id, "model", full_response)

    return templates.TemplateResponse("_spec_chat_message.html", {
        "request": request,
        "session_id": session_id,
        "variant_key": variant_key,
        "user_message": user_text,
        "assistant_text": assistant_text,
        "spec": spec_json,
        "primitives": get_spec_primitives(),
        "is_example_load": True,
    })


@router.post("/{variant_key}/simulate", response_class=HTMLResponse)
async def simulate_spec(
    request: Request,
    variant_key: str,
    spec_json: str = Form(...),
    session_id: str = Form(""),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Run simulation on a spec JSON and return results as an HTMX partial.

    Simulation works for ALL modalities — the variant_key is used for
    classification but does not need to be in the platform registry.
    """
    # Variant lookup is optional for simulation (all 168 modalities supported)
    variant = get_variant(variant_key)  # may be None for non-registry modalities

    # Parse the spec JSON
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Invalid spec JSON: %s", exc)
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            "Invalid spec JSON. Please build a spec first via chat.</div></div>",
            status_code=200,
        )

    try:
        result = await run_spec_simulation(spec, variant_key)
    except Exception as exc:
        logger.error("Simulation error: %s", exc, exc_info=True)
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            f"Simulation failed: {type(exc).__name__}: {exc}</div></div>",
            status_code=200,
        )

    return templates.TemplateResponse("_spec_simulation_result.html", {
        "request": request,
        "result": result,
    })
