"""Spec Chat Router — interactive spec builder powered by Gemini 2.5 Flash.

Prefix: /api/v1/spec-chat

Endpoints:
  POST /{variant_key}            — Process a chat message (create / continue conversation)
  POST /{variant_key}/example    — Load a pre-built example spec (cassi, spc, cacti)
  POST /{variant_key}/simulate   — Run simulation on a spec JSON
  POST /{variant_key}/upload     — Upload measurement data (+ optional matrix/GT)
  POST /{variant_key}/reconstruct — Reconstruct from uploaded data using best method
  GET  /example-data/{key}/{role} — Download example .npy dataset for a modality
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
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
    get_session_dataset_meta,
    update_session_dataset_meta,
)
from pwm_platform.services.spec_chat_prompts import (
    build_system_prompt,
    get_example_spec,
    parse_spec_from_response,
)
from pwm_platform.services.spec_simulator import run_spec_simulation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/spec-chat", tags=["Spec Chat"])


def _detect_variant_from_spec(spec: dict) -> str | None:
    """Detect the imaging modality from spec content when variant_key is absent.

    Uses mismatch parameter names, measurement matrix text, and forward-model
    labels as progressively weaker signals.
    """
    # 1. Mismatch parameter names — most reliable signal
    param_names = {p.get("name", "").lower() for p in spec.get("mismatch_params", [])}
    if "gain_alpha" in param_names or param_names & {"sigma_y", "pattern_dx"}:
        return "spc_block"
    if "dispersion_step" in param_names:
        return "sd_cassi"
    if "timing_offset" in param_names:
        return "cacti"

    # 2. Measurement matrix description
    mm = (spec.get("measurement_matrix") or "").lower()
    if any(kw in mm for kw in ("single pixel", "single-pixel", "spc", "block-structured")):
        return "spc_block"
    if any(kw in mm for kw in ("coded aperture", "cassi", "dispersion")):
        return "sd_cassi"
    if any(kw in mm for kw in ("temporal mask", "cacti", "snapshot compressive", "time-varying")):
        return "cacti"

    # 3. Forward-model primitive labels
    labels = " ".join(n.get("label", "").lower() for n in spec.get("forward_model", []))
    params = " ".join(n.get("params", "").lower() for n in spec.get("forward_model", []))
    text = labels + " " + params
    if "single-pixel" in text or "single pixel" in text or "spatial summation" in text:
        return "spc_block"
    if "dispersion" in text or "spectral" in text:
        return "sd_cassi"
    if "temporal" in text:
        return "cacti"

    return None


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

    # Check for dataset mode
    ds_meta, mat_meta, _gt_meta = await get_session_dataset_meta(db, session_id)
    dataset_mode = ds_meta is not None

    # Get full history for Gemini call
    history = await get_conversation(db, session_id)
    system_prompt = build_system_prompt(
        variant,
        dataset_meta=ds_meta,
        matrix_meta=mat_meta,
    )

    try:
        response_text = await call_gemini(system_prompt, history)
    except Exception as exc:
        logger.error("Gemini API error: %s", exc, exc_info=True)
        return templates.TemplateResponse("_spec_chat_message.html", {
            "request": request,
            "session_id": session_id,
            "variant_key": variant_key,
            "user_message": message,
            "assistant_text": f"Sorry, I encountered an error calling the AI service. Please try again. ({type(exc).__name__})",
            "spec": None,
            "primitives": get_spec_primitives(),
            "dataset_mode": dataset_mode,
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
        "dataset_mode": dataset_mode,
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
        "variant_key": example_spec.get("variant_key", example),
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

    # Determine the correct modality:
    #   1. Explicit variant_key in spec JSON (Gemini output)
    #   2. Detected from spec content (mismatch params, measurement matrix, labels)
    #   3. URL variant_key (fallback — may be the dashboard default "sd_cassi")
    effective_vk = (
        spec.get("variant_key")
        or _detect_variant_from_spec(spec)
        or variant_key
    )
    logger.info("Simulation: URL vk=%s, spec vk=%s, detected=%s, effective=%s",
                variant_key, spec.get("variant_key"),
                _detect_variant_from_spec(spec), effective_vk)

    try:
        result = await run_spec_simulation(spec, effective_vk)
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


# ── Dataset upload endpoint ──────────────────────────────────────────────


@router.post("/{variant_key}/upload", response_class=HTMLResponse)
async def upload_dataset(
    request: Request,
    variant_key: str,
    measurement_file: UploadFile = File(...),
    matrix_file: UploadFile = File(None),
    ground_truth_file: UploadFile = File(None),
    session_id: str = Form(""),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload measurement data (+ optional sensing matrix / ground truth).

    Saves files, extracts metadata, stores on session, returns HTMX partial.
    """
    from pwm_platform.services.dataset_upload import (
        extract_dataset_metadata,
        save_uploaded_file,
    )

    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    user_id = user.id if user else None

    # Create session if needed
    if not session_id:
        session_id = await create_conversation(db, user_id=user_id, variant_key=variant_key)
    else:
        history = await get_conversation(db, session_id)
        if history is None:
            session_id = await create_conversation(db, user_id=user_id, variant_key=variant_key)

    try:
        # Save and extract measurement
        meas_content = await measurement_file.read()
        meas_path = await save_uploaded_file(
            meas_content, measurement_file.filename or "measurement.npy",
            session_id, "measurement",
        )
        meas_meta = extract_dataset_metadata(meas_path, "measurement")
        ds_meta_dict = meas_meta.to_dict()

        # Optional: sensing matrix
        mat_meta_dict = None
        if matrix_file and matrix_file.filename:
            mat_content = await matrix_file.read()
            mat_path = await save_uploaded_file(
                mat_content, matrix_file.filename,
                session_id, "sensing_matrix",
            )
            mat_meta = extract_dataset_metadata(mat_path, "sensing_matrix")
            mat_meta_dict = mat_meta.to_dict()

        # Optional: ground truth
        gt_meta_dict = None
        if ground_truth_file and ground_truth_file.filename:
            gt_content = await ground_truth_file.read()
            gt_path = await save_uploaded_file(
                gt_content, ground_truth_file.filename,
                session_id, "ground_truth",
            )
            gt_meta = extract_dataset_metadata(gt_path, "ground_truth")
            gt_meta_dict = gt_meta.to_dict()

        # Store metadata on session
        await update_session_dataset_meta(
            db, session_id, ds_meta_dict, mat_meta_dict, gt_meta_dict
        )

    except ValueError as exc:
        return HTMLResponse(
            f'<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            f'Upload error: {exc}</div></div>',
            status_code=200,
        )
    except Exception as exc:
        logger.error("Upload error: %s", exc, exc_info=True)
        return HTMLResponse(
            f'<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            f'Upload failed: {type(exc).__name__}</div></div>',
            status_code=200,
        )

    # Build upload summary as assistant message
    shape_str = "x".join(str(s) for s in ds_meta_dict["shape"])
    summary_parts = [
        f"**Measurement uploaded:** `{ds_meta_dict['original_filename']}` "
        f"({ds_meta_dict['file_format'].upper()}, {shape_str}, {ds_meta_dict['dtype']})",
        f"Value range: [{ds_meta_dict['stats']['min']:.4g}, {ds_meta_dict['stats']['max']:.4g}], "
        f"mean={ds_meta_dict['stats']['mean']:.4g}, std={ds_meta_dict['stats']['std']:.4g}",
    ]
    if mat_meta_dict:
        m_shape = "x".join(str(s) for s in mat_meta_dict["shape"])
        summary_parts.append(
            f"**Sensing matrix:** `{mat_meta_dict['original_filename']}` ({m_shape})"
        )
    if gt_meta_dict:
        g_shape = "x".join(str(s) for s in gt_meta_dict["shape"])
        summary_parts.append(
            f"**Ground truth:** `{gt_meta_dict['original_filename']}` ({g_shape})"
        )
    summary_parts.append(
        "\nDataset Mode is now active. Describe your imaging system and I'll "
        "design a spec matched to your data. When ready, click **Reconstruct from Dataset**."
    )
    assistant_text = "\n".join(summary_parts)

    # Store in conversation
    await append_to_conversation(
        db, session_id, "user",
        f"[Uploaded measurement: {ds_meta_dict['original_filename']}]"
    )
    await append_to_conversation(db, session_id, "model", assistant_text)

    return templates.TemplateResponse("_spec_chat_message.html", {
        "request": request,
        "session_id": session_id,
        "variant_key": variant_key,
        "user_message": f"Uploaded: {ds_meta_dict['original_filename']}",
        "assistant_text": assistant_text,
        "spec": None,
        "primitives": get_spec_primitives(),
        "dataset_mode": True,
    })


# ── Dataset reconstruction endpoint ──────────────────────────────────────


@router.post("/{variant_key}/reconstruct", response_class=HTMLResponse)
async def reconstruct_dataset(
    request: Request,
    variant_key: str,
    spec_json: str = Form(...),
    session_id: str = Form(""),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Reconstruct from uploaded data using the best benchmark method."""
    from pwm_platform.services.dataset_reconstructor import run_reconstruction

    # Parse spec
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Invalid spec JSON for reconstruction: %s", exc)
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            "Invalid spec JSON. Please build a spec first via chat.</div></div>",
            status_code=200,
        )

    # Get dataset metadata from session
    ds_meta, mat_meta, gt_meta = await get_session_dataset_meta(db, session_id)
    if ds_meta is None:
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            "No dataset uploaded. Please upload measurement data first.</div></div>",
            status_code=200,
        )

    # Determine effective variant key
    effective_vk = (
        spec.get("variant_key")
        or _detect_variant_from_spec(spec)
        or variant_key
    )

    try:
        result = await run_reconstruction(
            spec, effective_vk, ds_meta, mat_meta, gt_meta
        )
    except Exception as exc:
        logger.error("Reconstruction error: %s", exc, exc_info=True)
        return HTMLResponse(
            f'<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            f'Reconstruction failed: {type(exc).__name__}: {exc}</div></div>',
            status_code=200,
        )

    return templates.TemplateResponse("_spec_reconstruction_result.html", {
        "request": request,
        "result": result,
    })


# ── Example dataset download ─────────────────────────────────────────────


@router.get("/example-data/{key}/{role}")
async def download_example_dataset(key: str, role: str):
    """Download an example .npy dataset for trying Dataset Mode.

    Uses pre-cached .npy bytes so the response is instant (no CPU work
    at request time).  The cache is warmed at app startup via warmup_all().
    """
    from fastapi.responses import Response

    from pwm_platform.services.example_datasets import (
        EXAMPLE_DATASETS,
        get_npy_bytes,
    )

    if key not in EXAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown example: {key}")
    if role not in ("measurement", "matrix", "ground_truth"):
        raise HTTPException(status_code=400, detail="role must be measurement, matrix, or ground_truth")

    info = EXAMPLE_DATASETS[key]
    if role == "matrix" and not info.get("has_matrix"):
        raise HTTPException(status_code=404, detail=f"No matrix for {key}")

    try:
        npy_bytes = get_npy_bytes(key, role)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filename = f"{key}_{role}.npy"

    return Response(
        content=npy_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
