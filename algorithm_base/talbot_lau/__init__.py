"""Talbot-Lau X-ray Grating Interferometry (talbot_lau) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME
from .solvers import run_traditional_cpu
from .solvers import run_best_quality
from .solvers import run_talbot_dl

__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]
