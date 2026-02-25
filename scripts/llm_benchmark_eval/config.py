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
COMPAREGPT_BASE_URL = "https://api.comparegpt.io/v1"
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
        ModelEntry("Gemini 3 Pro Preview",           "gemini_3_pro"),
        ModelEntry("Gemini 2.5 Pro",                 "gemini_2_5_pro"),
        ModelEntry("Gemini 2.5 Flash",               "gemini_2_5_flash"),
        ModelEntry("Gemini 2.5 Flash Lite",          "gemini_2_5_flash_lite"),
        ModelEntry("Gemini 2.5 Flash Image Preview", "gemini_2_5_flash_image"),
        ModelEntry("Claude Opus 4.6",                "claude_opus_4_6"),
        ModelEntry("Claude Opus 4.5",                "claude_opus_4_5"),
        ModelEntry("Claude Haiku 4.5",               "claude_haiku_4_5"),
        ModelEntry("Claude Sonnet 4.5",              "claude_sonnet_4_5"),
        ModelEntry("DeepSeek V3.2",                  "deepseek_v3_2"),
        ModelEntry("DeepSeek V3.1",                  "deepseek_v3_1"),
        ModelEntry("DeepSeek R1",                    "deepseek_r1"),
        ModelEntry("Qwen 3",                         "qwen_3"),
        ModelEntry("Qwen 3 Next",                    "qwen_3_next"),
        ModelEntry("Qwen 3 Next Thinking",           "qwen_3_next_thinking"),
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
