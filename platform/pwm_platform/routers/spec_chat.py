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
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User
from pwm_platform.services.benchmark_database import get_variant, get_spec_primitives
from pwm_platform.services.gemini_client import (
    append_to_conversation,
    call_gemini,
    create_conversation,
    delete_session,
    get_conversation,
    get_session_dataset_meta,
    get_session_row,
    list_user_sessions,
    update_session_dataset_meta,
)
from pwm_platform.services.spec_chat_prompts import (
    build_system_prompt,
    get_example_spec,
    parse_spec_from_response,
)
from pwm_platform.services.spec_simulator import run_spec_simulation
from pwm_platform.services.system_recommender import (
    parse_system_query,
    recommend,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/spec-chat", tags=["Spec Chat"])

# Limit concurrent heavy reconstructions to prevent thread pool exhaustion.
# CASSI DL models can take 60–150 s on CPU; without a cap, multiple simultaneous
# requests fill all executor threads and make the entire server unresponsive.
import asyncio as _asyncio
_recon_semaphore = _asyncio.Semaphore(3)


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
        return "cassi"
    if "timing_offset" in param_names:
        return "cacti"

    # 2. Measurement matrix description
    mm = (spec.get("measurement_matrix") or "").lower()
    if any(kw in mm for kw in ("single pixel", "single-pixel", "spc", "block-structured")):
        return "spc_block"
    if any(kw in mm for kw in ("coded aperture", "cassi", "dispersion")):
        return "cassi"
    if any(kw in mm for kw in ("temporal mask", "cacti", "snapshot compressive", "time-varying")):
        return "cacti"

    # 3. Forward-model primitive labels
    labels = " ".join(n.get("label", "").lower() for n in spec.get("forward_model", []))
    params = " ".join(n.get("params", "").lower() for n in spec.get("forward_model", []))
    text = labels + " " + params
    if "single-pixel" in text or "single pixel" in text or "spatial summation" in text:
        return "spc_block"
    if "dispersion" in text or "spectral" in text:
        return "cassi"
    if "temporal" in text:
        return "cacti"

    return None


templates = Jinja2Templates(directory="pwm_platform/templates")


@router.post("/system-recommend", response_class=HTMLResponse)
async def system_recommend(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(""),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run system recommendation from a natural language query (requires login)."""
    from pwm_platform.services.system_recommender import TaskQuery

    query = parse_system_query(message)
    if query is None:
        query = TaskQuery(purpose=message)

    try:
        result = recommend(query)
    except Exception as exc:
        logger.error("System recommendation error: %s", exc, exc_info=True)
        return HTMLResponse(
            '<div class="flex justify-start"><div class="px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">'
            f"Recommendation failed: {type(exc).__name__}: {exc}</div></div>",
            status_code=200,
        )

    return templates.TemplateResponse("_spec_system_result.html", {
        "request": request,
        "result": result,
        "session_id": session_id,
    })


# ── Common-mode chat endpoint (LLM-assisted modality/algorithm selection) ──


@router.post("/common-chat", response_class=HTMLResponse)
async def common_chat(
    request: Request,
    message: str = Form(...),
):
    """Interpret a natural-language prompt to select modality + algorithm.

    No login required.  Uses Gemini 2.5 Flash to map free-text descriptions
    to a ``variant_key`` and ``algorithm_name`` from the existing catalogs.
    Returns a JSON payload that the frontend JS uses to set the dropdowns.
    """
    from pwm_platform.services.benchmark_database import VARIANT_DATABASE, get_algorithms, resolve_algorithm

    # Build a compact catalog string for the system prompt
    variant_entries = []
    for vk, v in sorted(VARIANT_DATABASE.items()):
        cat = v.get("category", "compressive")
        algos = get_algorithms(vk, cat)
        algo_names = [a["name"] for a in algos[:8]]  # top 8 algorithms
        algo_str = ", ".join(algo_names) if algo_names else "N/A"
        variant_entries.append(
            f"- {vk}: {v.get('display_name', vk)} | algorithms: [{algo_str}]"
        )
    variants_str = "\n".join(variant_entries[:120])  # cap for context

    system_prompt = (
        "You are a reconstruction assistant for the Physics World Model (PWM) platform.\n"
        "The user will describe an imaging modality and/or reconstruction algorithm.\n"
        "Your job is to match their description to the best variant_key and algorithm_name "
        "from the catalogs below.\n\n"
        "## Available Modalities (variant_key: display_name | algorithms)\n"
        f"{variants_str}\n\n"
        "## Instructions\n"
        "1. Match the user's modality description to the closest variant_key.\n"
        "2. If the user mentions an algorithm name, match it to one from that modality's "
        "algorithm list above. Use the EXACT algorithm name from the catalog.\n"
        "3. If the user does not specify an algorithm, pick the best classical algorithm "
        "for that modality (e.g. FBP for CT, GAP-TV for CASSI, Tikhonov for SPC).\n"
        "4. Respond with ONLY a JSON object, no markdown fences, no explanation:\n"
        '   {"variant_key": "...", "algorithm_name": "...", "explanation": "brief 1-line reason"}\n'
        "5. If you cannot determine the modality, set variant_key to \"\" and explain in explanation.\n"
        "6. The algorithm_name MUST be one that exists in the modality's algorithm list."
    )

    history = [{"role": "user", "content": message}]

    try:
        response_text = await call_gemini(system_prompt, history)
    except Exception as exc:
        logger.error("Common-chat Gemini error: %s", exc, exc_info=True)
        return HTMLResponse(json.dumps({
            "variant_key": "",
            "algorithm_name": "",
            "explanation": f"AI service error: {type(exc).__name__}",
        }), status_code=200)

    # Parse JSON from response (strip markdown fences if present)
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text).strip().rstrip("`")
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse common-chat response: %s", response_text[:200])
        parsed = {
            "variant_key": "",
            "algorithm_name": "",
            "explanation": response_text[:200],
        }

    # Validate variant_key exists
    vk = parsed.get("variant_key", "")
    if vk and vk not in VARIANT_DATABASE:
        # Fuzzy match
        vk_lower = vk.lower()
        for k in VARIANT_DATABASE:
            if k.lower() == vk_lower or vk_lower in k.lower():
                parsed["variant_key"] = k
                break

    # If we have a valid variant_key, get the algorithm list from the central catalog
    algo_options = ""
    vk = parsed.get("variant_key", "")
    if vk and vk in VARIANT_DATABASE:
        v = VARIANT_DATABASE[vk]
        category = v.get("category", "compressive")
        algos = get_algorithms(vk, category)
        algo_options = json.dumps([a["name"] for a in algos])
        # Validate algorithm_name via central resolve_algorithm()
        algo_name = parsed.get("algorithm_name", "")
        if algo_name:
            resolved = resolve_algorithm(vk, category, algo_name)
            if resolved:
                parsed["algorithm_name"] = resolved["name"]
            else:
                # No match found — use first classical algorithm
                parsed["algorithm_name"] = algos[0]["name"] if algos else ""

    parsed["algo_options"] = algo_options
    return HTMLResponse(json.dumps(parsed), status_code=200)


# ── Common-mode reconstruction endpoint ──────────────────────────────────
# MUST be registered before /{variant_key} to avoid route shadowing.


@router.post("/reconstruct-common", response_class=HTMLResponse)
async def reconstruct_common(
    request: Request,
    modality: str = Form(...),
    algorithm: str = Form(...),
    sample_index: int = Form(0),
    measurement_file: Optional[UploadFile] = File(None),
    matrix_file: Optional[UploadFile] = File(None),
):
    """Run common-mode reconstruction (no login required for standard data)."""
    import numpy as np

    from pwm_platform.services.common_reconstructor import run_common_reconstruction

    # Parse user uploads if provided
    user_measurement = None
    user_matrix = None

    if measurement_file and measurement_file.filename:
        try:
            content = await measurement_file.read()
            import io as _io
            user_measurement = np.load(_io.BytesIO(content))
        except Exception as exc:
            logger.warning("Failed to load measurement file: %s", exc)
            return HTMLResponse(
                '<div class="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">'
                f'Failed to load measurement file: {exc}</div>',
                status_code=200,
            )

    if matrix_file and matrix_file.filename:
        try:
            content = await matrix_file.read()
            import io as _io
            user_matrix = np.load(_io.BytesIO(content))
        except Exception as exc:
            logger.warning("Failed to load matrix file: %s", exc)

    if not _recon_semaphore._value:  # all slots occupied
        return HTMLResponse(
            '<div class="p-4 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">'
            'Server busy — another reconstruction is running. Please wait a moment and try again.</div>',
            status_code=200,
        )

    try:
        async with _recon_semaphore:
            result = await run_common_reconstruction(
                variant_key=modality,
                algorithm_name=algorithm,
                user_measurement=user_measurement,
                user_matrix=user_matrix,
                sample_index=sample_index,
            )
    except Exception as exc:
        logger.error("Common reconstruction error: %s", exc, exc_info=True)
        return HTMLResponse(
            '<div class="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">'
            f'Reconstruction failed: {type(exc).__name__}: {exc}</div>',
            status_code=200,
        )

    return templates.TemplateResponse("_spec_common_result.html", {
        "request": request,
        "result": result,
    })


@router.post("/{variant_key}", response_class=HTMLResponse)
async def chat_message(
    request: Request,
    variant_key: str,
    message: str = Form(...),
    session_id: str = Form(""),
    input_mode: str = Form("describe"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Process a user chat message and return an HTMX partial (requires login)."""
    # Check if this is a system-level query → route to recommender
    sys_query = parse_system_query(message)
    if sys_query is not None and input_mode != "upload_spec":
        try:
            result = recommend(sys_query)
            # Also show user message bubble + system result
            return templates.TemplateResponse("_spec_system_result.html", {
                "request": request,
                "result": result,
                "session_id": session_id,
                "user_message": message,
            })
        except Exception as exc:
            logger.warning("System recommendation fallback to spec chat: %s", exc)
            # Fall through to regular spec chat

    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    user_id = user.id

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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Load an example spec and start a new chat session (requires login)."""
    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    example_spec = get_example_spec(example)
    if example_spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown example: {example}")

    user_id = user.id

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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run simulation on a spec JSON (requires login).

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
    #   3. URL variant_key (fallback — may be the SpecLab default "cassi")
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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload measurement data (requires login).

    Saves files, extracts metadata, stores on session, returns HTMX partial.
    """
    from pwm_platform.services.dataset_upload import (
        extract_dataset_metadata,
        save_uploaded_file,
    )

    variant = get_variant(variant_key)
    if variant is None:
        raise HTTPException(status_code=404, detail="Variant not found")

    user_id = user.id

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
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reconstruct from uploaded data (requires login)."""
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


# ── Chat history endpoints ────────────────────────────────────────────────


@router.get("/sessions", response_class=HTMLResponse)
async def list_sessions(
    request: Request,
    current_session: str = "",
    variant_key: str = "",
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """List chat sessions for the current user (returns HTML partial).

    When ``variant_key`` is provided, only sessions for that variant are
    returned (per-chatbox history).
    """
    sessions: list[dict] = []
    if user:
        sessions = await list_user_sessions(
            db, user.id, variant_key=variant_key or None,
        )

    return templates.TemplateResponse("_spec_chat_history.html", {
        "request": request,
        "user": user,
        "sessions": sessions,
        "current_session_id": current_session,
    })


@router.get("/sessions/{session_id}/load", response_class=HTMLResponse)
async def load_session(
    request: Request,
    session_id: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Load the full conversation for a session (returns HTML messages)."""
    row = await get_session_row(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Security: only the owning user can load their sessions
    if user is None or row.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your session")

    history = row.history or []
    variant_key = row.variant_key or "cassi"

    # Render each user+assistant pair as a _spec_chat_message.html partial
    html_parts: list[str] = []
    i = 0
    while i < len(history):
        turn = history[i]
        if turn.get("role") == "user":
            user_msg = turn.get("content", "")
            assistant_text = ""
            spec = None
            # Look for the following assistant turn
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                assistant_text = history[i + 1].get("content", "")
                _, spec = parse_spec_from_response(assistant_text)
                # Use just the explanation portion
                explanation, _ = parse_spec_from_response(assistant_text)
                assistant_text = explanation
                i += 2
            else:
                i += 1

            rendered = templates.TemplateResponse("_spec_chat_message.html", {
                "request": request,
                "session_id": session_id,
                "variant_key": variant_key,
                "user_message": user_msg,
                "assistant_text": assistant_text,
                "spec": spec,
                "primitives": get_spec_primitives(),
                "dataset_mode": row.dataset_meta is not None,
            })
            html_parts.append(rendered.body.decode())
        else:
            i += 1

    # Return concatenated HTML + a script to set the session ID
    session_script = (
        f'<script>(function(){{'
        f'var el=document.getElementById("chat-session-id");'
        f'if(el)el.value="{session_id}";'
        f'var log=document.getElementById("chat-log");'
        f'if(log)log.scrollTop=log.scrollHeight;'
        f'}})();</script>'
    )
    return HTMLResponse("".join(html_parts) + session_script)


@router.delete("/sessions/{session_id}", response_class=HTMLResponse)
async def delete_session_endpoint(
    request: Request,
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session (ownership check enforced)."""
    row = await get_session_row(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if row.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your session")

    await delete_session(db, session_id)
    return HTMLResponse("")  # empty response for HTMX swap


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


# ── Algorithm list endpoint (HTMX) ──────────────────────────────────────


@router.get("/algorithms/{variant_key}", response_class=HTMLResponse)
async def get_variant_algorithms(variant_key: str):
    """Return algorithm <option> tags for a variant (HTMX population).

    Uses the central ``get_algorithms()`` from the benchmark database
    package — same source as common-chat, dataset reconstruction, and
    leaderboard generation.
    """
    from pwm_platform.services.benchmark_database import get_algorithms, get_variant

    variant = get_variant(variant_key)
    if variant is None:
        return HTMLResponse('<option value="">Unknown modality</option>')

    category = variant.get("category", "compressive")
    algos = get_algorithms(variant_key, category)
    options = "".join(
        f'<option value="{a["name"]}">{a["name"]} ({a["type"]})</option>'
        for a in algos
    )
    return HTMLResponse(options or '<option value="">No algorithms available</option>')


@router.get("/samples/{variant_key}")
async def get_variant_samples(variant_key: str):
    """Return available sample indices for a variant's benchmark data.

    Checks the GCS challenge HDF5 file for available sample keys.
    Returns JSON: {"samples": [{"index": 0, "label": "Sample 1"}, ...]}
    """
    from fastapi.responses import JSONResponse
    from pwm_platform.services.common_reconstructor import _ensure_challenge_h5

    try:
        h5_path = _ensure_challenge_h5(variant_key, "public")
        import h5py
        with h5py.File(h5_path, "r") as f:
            sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
        samples = []
        for i, sk in enumerate(sample_keys[:3]):  # Cap at 3
            samples.append({"index": i, "label": f"Sample {i + 1}", "key": sk})
        if not samples:
            samples = [{"index": 0, "label": "Sample 1", "key": "sample_00"}]
        return JSONResponse({"samples": samples})
    except Exception:
        return JSONResponse({"samples": [
            {"index": 0, "label": "Sample 1", "key": "sample_00"},
            {"index": 1, "label": "Sample 2", "key": "sample_01"},
            {"index": 2, "label": "Sample 3", "key": "sample_02"},
        ]})


