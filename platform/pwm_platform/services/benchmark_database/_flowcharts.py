"""Flowcharts — visual pipeline descriptions for the benchmark flow."""

from __future__ import annotations

FLOWCHARTS = {
    "flow_b1_b2": {
        "title": "Flow 1: Spec Selection + Reconstruction (Benchmark 1 \u2192 Benchmark 2)",
        "description": "User describes imaging system in natural language; LLM routes to spec; algorithm reconstructs.",
        "steps": [
            {"label": "User Prompt", "detail": "Natural-language description of imaging modality", "color": "blue"},
            {"label": "Benchmark 1: LLM Spec Router", "detail": "Selects spec from 11 primitives", "color": "indigo"},
            {"label": "Spec Correct?", "detail": "Validate against known specs", "color": "amber"},
            {"label": "Refine Prompt", "detail": "User clarifies if spec mismatch", "color": "amber"},
            {"label": "Benchmark 2: Algorithm Recon", "detail": "Corrects forward model + reconstructs x\u0302", "color": "indigo"},
            {"label": "Output x\u0302", "detail": "Reconstructed signal with PSNR/SSIM", "color": "green"},
        ],
    },
    "flow_b3_b4": {
        "title": "Flow 2: Ground-Truth Validation + Drift Robustness (Benchmark 3 \u2192 Benchmark 4)",
        "description": "User uploads measurements and true forward model; LLM validates spec; algorithm handles drift.",
        "steps": [
            {"label": "Upload y, H", "detail": "User provides measurements + true forward model", "color": "blue"},
            {"label": "Benchmark 3: LLM Spec Validator", "detail": "Compares candidate spec vs true spec", "color": "indigo"},
            {"label": "Spec Match?", "detail": "Score spec against ground-truth H", "color": "amber"},
            {"label": "Flag Drift", "detail": "Low-scoring specs indicate system drift", "color": "amber"},
            {"label": "Benchmark 4: Algorithm Recon", "detail": "Reconstructs from drifted forward model", "color": "indigo"},
            {"label": "Output x\u0302", "detail": "Drift-robust reconstruction", "color": "green"},
        ],
    },
}
