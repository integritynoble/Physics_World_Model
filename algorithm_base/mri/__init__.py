"""Magnetic Resonance Imaging (MRI) (mri) — algorithm solvers."""
from .solvers import run_solver, list_solvers, SOLVERS, MODALITY_ID, DISPLAY_NAME


def run_traditional_cpu(y, op, cfg=None):
    return run_solver("traditional_cpu", y, op, cfg or {})


def run_best_quality(y, op, cfg=None):
    return run_solver("best_quality", y, op, cfg or {})


def run_famous_dl(y, op, cfg=None):
    return run_solver("famous_dl", y, op, cfg or {})


def run_small_gpu(y, op, cfg=None):
    return run_solver("small_gpu", y, op, cfg or {})


def run_sense(y, op, cfg=None):
    return run_solver("sense", y, op, cfg or {})


__all__ = ["run_solver", "list_solvers", "SOLVERS", "MODALITY_ID", "DISPLAY_NAME",
           "run_traditional_cpu", "run_best_quality", "run_famous_dl",
           "run_small_gpu", "run_sense"]
