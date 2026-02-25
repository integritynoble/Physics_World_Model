"""Spec Chat Router — interactive spec builder powered by Gemini 2.5 Flash.

Prefix: /api/v1/spec-chat

Endpoints:
  POST /{variant_key}          — Process a chat message (create / continue conversation)
  POST /{variant_key}/example  — Load a pre-built example spec (cassi, spc, cacti)
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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
):
    """Process a user chat message and return an HTMX partial."""
    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    # Create or retrieve session
    if not session_id:
        session_id = create_conversation()
    else:
        conv = get_conversation(session_id)
        if conv is None:
            # Session expired — start fresh
            session_id = create_conversation()

    # Build user message based on input mode
    if input_mode == "upload_spec":
        user_text = (
            "Please validate and describe the following spec JSON, then suggest "
            "improvements if any:\n\n" + message
        )
    else:
        user_text = message

    # Append user turn
    append_to_conversation(session_id, "user", user_text)

    # Call Gemini
    conv = get_conversation(session_id)
    system_prompt = build_system_prompt(variant)

    try:
        response_text = await call_gemini(system_prompt, conv["history"])
    except Exception as exc:
        logger.error("Gemini API error: %s", exc, exc_info=True)
        # Remove the failed user turn so they can retry
        conv["history"].pop()
        return templates.TemplateResponse("_spec_chat_message.html", {
            "request": request,
            "session_id": session_id,
            "user_message": message,
            "assistant_text": f"Sorry, I encountered an error calling the AI service. Please try again. ({type(exc).__name__})",
            "spec": None,
            "primitives": get_spec_primitives(),
        })

    # Append assistant turn
    append_to_conversation(session_id, "model", response_text)

    # Parse spec from response
    explanation, spec = parse_spec_from_response(response_text)

    return templates.TemplateResponse("_spec_chat_message.html", {
        "request": request,
        "session_id": session_id,
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
):
    """Load an example spec and start a new chat session."""
    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    example_spec = get_example_spec(example)
    if example_spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown example: {example}")

    # Create a new session
    session_id = create_conversation()

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
    append_to_conversation(session_id, "user", user_text)
    append_to_conversation(session_id, "model", full_response)

    return templates.TemplateResponse("_spec_chat_message.html", {
        "request": request,
        "session_id": session_id,
        "user_message": user_text,
        "assistant_text": assistant_text,
        "spec": spec_json,
        "primitives": get_spec_primitives(),
        "is_example_load": True,
    })
