"""Performance Agent — executes the approved plan.

Forward period:
  - Simulates each flowchart element in order
  - Applies noise models at each stage
  - Returns measurements y (e.g. sinogram, k-space, image with blur)

Reconstruction period:
  - Runs the specified algorithm on measurements y
  - Applies mismatch corrections as specified
  - Returns reconstructed image x_hat + PSNR/SSIM metrics
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schemas import PlanDocument, ForwardAction, ReconAction
from ..pipeline.forward_simulator import ForwardSimulator
from ..pipeline.reconstructor import Reconstructor
from ..utils.metrics import psnr, ssim


@dataclass
class ForwardResult:
    """Output of the Performance Agent for the forward period."""
    measurements: np.ndarray      # y
    ground_truth: np.ndarray      # x (phantom)
    metadata: dict[str, Any] = field(default_factory=dict)
    element_outputs: dict[str, np.ndarray] = field(default_factory=dict)
    plan_path: str = ""


@dataclass
class ReconResult:
    """Output of the Performance Agent for the reconstruction period."""
    x_hat: np.ndarray             # reconstructed image
    x_true: np.ndarray | None     # ground truth (if available)
    psnr_db: float | None = None
    ssim_val: float | None = None
    convergence_history: list[float] = field(default_factory=list)
    runtime_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    plan_path: str = ""

    def __str__(self) -> str:
        lines = [f"ReconResult: PSNR={self.psnr_db:.2f} dB  SSIM={self.ssim_val:.4f}" ]
        lines.append(f"  Runtime: {self.runtime_s:.1f}s  Iterations: {len(self.convergence_history)}")
        return "\n".join(lines)


class PerformanceAgent:
    """Executes an approved PlanDocument end-to-end.

    All action steps defined in the plan must be completed in one call — the
    agent does not stop mid-plan.
    """

    def __init__(self):
        self._forward_sim  = ForwardSimulator()
        self._reconstructor = Reconstructor()

    def run(
        self,
        plan: PlanDocument,
        phantom: np.ndarray | None = None,
        measurements: np.ndarray | None = None,
        x_true: np.ndarray | None = None,
    ) -> ForwardResult | ReconResult:
        """Execute the plan.

        Args:
            plan:          Approved PlanDocument.
            phantom:       (forward) Ground truth image. Auto-generated if None.
            measurements:  (reconstruction) Pre-computed measurements. Required for recon.
            x_true:        (reconstruction) Ground truth for metric evaluation.

        Returns:
            ForwardResult for forward period, ReconResult for reconstruction period.
        """
        if plan.period == "forward":
            return self._run_forward(plan, phantom)
        else:
            if measurements is None:
                raise ValueError("measurements must be provided for the reconstruction period")
            return self._run_reconstruction(plan, measurements, x_true)

    # ── Forward ────────────────────────────────────────────────────────────────

    def _run_forward(
        self,
        plan: PlanDocument,
        phantom: np.ndarray | None,
    ) -> ForwardResult:
        assert isinstance(plan.action, ForwardAction)

        print(f"[PerformanceAgent] Running FORWARD simulation for '{plan.modality}'")
        print(f"  Elements: {[el.id for el in plan.action.elements]}")

        if phantom is None:
            phantom = self._default_phantom(plan.modality)
            print(f"  Using default phantom shape={phantom.shape}")

        measurements, element_outputs = self._forward_sim.simulate(
            action=plan.action,
            phantom=phantom,
            modality=plan.modality,
        )

        print(f"  Measurements shape: {measurements.shape}  "
              f"range=[{measurements.min():.3g}, {measurements.max():.3g}]")

        return ForwardResult(
            measurements=measurements,
            ground_truth=phantom,
            metadata={"modality": plan.modality, "plan_iter": plan.iteration},
            element_outputs=element_outputs,
            plan_path=plan.markdown_path,
        )

    # ── Reconstruction ─────────────────────────────────────────────────────────

    def _run_reconstruction(
        self,
        plan: PlanDocument,
        measurements: np.ndarray,
        x_true: np.ndarray | None,
    ) -> ReconResult:
        import time
        assert isinstance(plan.action, ReconAction)

        print(f"[PerformanceAgent] Running RECONSTRUCTION '{plan.action.algorithm_name}' "
              f"for '{plan.modality}'")

        t0 = time.time()
        x_hat, history = self._reconstructor.reconstruct(
            action=plan.action,
            measurements=measurements,
            modality=plan.modality,
        )
        runtime = time.time() - t0

        psnr_val = ssim_val = None
        if x_true is not None:
            psnr_val = psnr(x_hat, x_true)
            ssim_val = ssim(x_hat, x_true)
            print(f"  PSNR={psnr_val:.2f} dB  SSIM={ssim_val:.4f}  Runtime={runtime:.1f}s")
        else:
            print(f"  Runtime={runtime:.1f}s  (no ground truth for metrics)")

        return ReconResult(
            x_hat=x_hat,
            x_true=x_true,
            psnr_db=psnr_val,
            ssim_val=ssim_val,
            convergence_history=history,
            runtime_s=runtime,
            metadata={"modality": plan.modality, "algorithm": plan.action.algorithm_name},
            plan_path=plan.markdown_path,
        )

    # ── Phantom factory ────────────────────────────────────────────────────────

    def _default_phantom(self, modality: str) -> np.ndarray:
        """Generate a standard test phantom appropriate for the modality."""
        from skimage.data import shepp_logan_phantom
        phantom = shepp_logan_phantom()                   # 400×400 float64
        from skimage.transform import resize
        if modality in ("ct", "cbct"):
            phantom = resize(phantom, (256, 256), anti_aliasing=True)
        elif modality == "mri":
            phantom = resize(phantom, (256, 256), anti_aliasing=True)
        elif modality in ("widefield", "confocal"):
            # Synthetic cell pattern
            rng = np.random.default_rng(42)
            phantom = rng.random((256, 256)).astype(np.float32)
        else:
            phantom = resize(phantom, (128, 128), anti_aliasing=True)
        return phantom.astype(np.float32) / phantom.max()
