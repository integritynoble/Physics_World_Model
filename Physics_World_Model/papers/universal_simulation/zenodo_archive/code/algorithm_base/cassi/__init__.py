"""Coded Aperture Snapshot Spectral Imaging (CASSI) (cassi) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME
from .solvers import run_traditional_cpu
from .solvers import run_best_quality
from .solvers import run_famous_dl
from .solvers import run_small_gpu
from .solvers import run_mst_l
from .solvers import run_hdnet
from .solvers import run_hsi_sdecnn

__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]
