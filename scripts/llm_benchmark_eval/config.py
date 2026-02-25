"""Configuration: model registry, API settings, file paths."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DATA_DIR = (
    PROJECT_ROOT / "platform" / "pwm_platform" / "static" / "benchmark-data" / "v1.0"
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.json"
SUMMARY_FILE = RESULTS_DIR / "summary.json"

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
COMPAREGPT_BASE_URL = "https://comparegpt.io/api"
COMPAREGPT_API_KEY_ENV = "COMPAREGPT_API_KEY"

MAX_CONCURRENCY = 10          # asyncio.Semaphore limit
REQUEST_TIMEOUT = 120         # seconds per request
MAX_RETRIES = 5               # retry count on 429 / 5xx
RETRY_BASE_DELAY = 2.0        # exponential backoff base (seconds)


def get_api_key() -> str:
    """Return the comparegpt.io API key from the environment."""
    key = os.environ.get(COMPAREGPT_API_KEY_ENV, "")
    if not key:
        raise RuntimeError(
            f"Environment variable {COMPAREGPT_API_KEY_ENV} is not set. "
            "Set it to your comparegpt.io API key."
        )
    return key


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelEntry:
    """A model available on comparegpt.io."""

    model_id: str   # exact string sent to the API
    short_key: str   # local shorthand for CLI / filenames


MODEL_REGISTRY: dict[str, ModelEntry] = {
    e.short_key: e
    for e in [
        ModelEntry("gemini-3-pro-preview",         "gemini_3_pro"),
        ModelEntry("gemini-2.5-pro",               "gemini_2_5_pro"),
        ModelEntry("gemini-2.5-flash",             "gemini_2_5_flash"),
        ModelEntry("gemini-2.5-flash-lite",        "gemini_2_5_flash_lite"),
        ModelEntry("gemini-2.5-flash-image-preview", "gemini_2_5_flash_image"),
        ModelEntry("claude-haiku-4-5",             "claude_haiku_4_5"),
        ModelEntry("claude-sonnet-4-5",            "claude_sonnet_4_5"),
        ModelEntry("deepseek-v3.2",                "deepseek_v3_2"),
        ModelEntry("deepseek-v3.1",                "deepseek_v3_1"),
        ModelEntry("deepseek-r1",                  "deepseek_r1"),
        ModelEntry("qwen-3",                       "qwen_3"),
        ModelEntry("qwen-3-next",                  "qwen_3_next"),
        ModelEntry("qwen-3-next-thinking",         "qwen_3_next_thinking"),
    ]
}

# ---------------------------------------------------------------------------
# Variant list (all 65 modalities)
# ---------------------------------------------------------------------------
ALL_VARIANTS: list[str] = [
    "cacti", "sd_cassi", "spc_block", "spc_kronecker", "matrix",
    "ct", "cbct", "pet", "spect",
    "xray_radiography", "mammography", "fluoroscopy", "angiography", "dexa",
    "dot", "photoacoustic",
    "mri", "fmri", "diffusion_mri", "mrs",
    "ultrasound", "doppler_ultrasound", "elastography",
    "holography", "ptychography", "phase_retrieval",
    "widefield", "widefield_lowdose", "confocal_livecell", "confocal_3d",
    "lightsheet", "two_photon", "sted", "tirf", "fundus",
    "sim", "fpm", "flim", "palm_storm", "polarization",
    "sem", "tem", "stem", "electron_tomography",
    "electron_diffraction", "electron_holography", "ebsd", "eels",
    "oct", "octa", "endoscopy",
    "light_field", "integral", "lensless",
    "panorama", "nerf", "gaussian_splatting",
    "tof_camera", "structured_light", "lidar",
    "sar", "sonar",
    "neutron_tomo", "proton_radiography", "muon_tomo",
]

BENCHMARKS = ("b1", "b3")
