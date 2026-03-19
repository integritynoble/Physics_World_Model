"""Flowcharts — visual pipeline descriptions for the benchmark flow."""

from __future__ import annotations

FLOWCHARTS = {
    "flow_challenge": {
        "title": "Blind Reconstruction Challenge",
        "description": "Given measurements with unknown mismatch, reconstruct the original signal across three difficulty tiers.",
        "steps": [
            {"label": "Download Dataset", "detail": "Measurements y, ideal forward model H, spec ranges", "color": "blue"},
            {"label": "Estimate Mismatch", "detail": "Use spec ranges to estimate forward model parameters", "color": "indigo"},
            {"label": "Reconstruct Signal", "detail": "Apply algorithm to recover x̂ from y", "color": "indigo"},
            {"label": "Submit Reconstruction", "detail": "Upload x̂ for scoring on PSNR/SSIM/consistency", "color": "green"},
        ],
    },
}
