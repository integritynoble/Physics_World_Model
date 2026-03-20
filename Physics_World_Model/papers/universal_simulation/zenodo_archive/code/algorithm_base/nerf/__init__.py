"""Neural Radiance Fields (NeRF) (nerf) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME
from .solvers import run_traditional_cpu
from .solvers import run_best_quality
from .solvers import run_famous_dl
from .solvers import run_small_gpu
from .solvers import run_rl_proxy
from .solvers import run_fista_proxy

__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME"]
