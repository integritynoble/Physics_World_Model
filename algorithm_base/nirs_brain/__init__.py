"""Functional Near-Infrared Spectroscopy (fNIRS) (nirs_brain) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME
from .solvers import run_traditional_cpu
from .solvers import run_best_quality
from .solvers import run_nirs_dl

__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]
