"""Configuration for the multi-agent system design pipeline."""
from __future__ import annotations
import os

# ── Models ────────────────────────────────────────────────────────────────────
PLAN_MODEL    = "claude-opus-4-6"
JUDGE_MODEL   = "claude-opus-4-6"
PERF_MODEL    = "claude-opus-4-6"

# ── Orchestrator settings ─────────────────────────────────────────────────────
MAX_REDESIGN_ITERATIONS = 3   # plan → judge → redesign loop limit

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR  = os.path.join(BASE_DIR, "database")
MODALITY_DIR  = os.path.join(DATABASE_DIR, "modalities")
ALGORITHM_DIR = os.path.join(DATABASE_DIR, "algorithms")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")

# Path to the existing platform modality database (for fallback lookup)
PLATFORM_DB_PATH = os.path.join(
    BASE_DIR, "..", "..", "platform",
    "pwm_platform", "services", "modality_database.py"
)

# ── Quality defaults ──────────────────────────────────────────────────────────
PSNR_TARGET = 40.0   # dB
SSIM_TARGET = 0.90
