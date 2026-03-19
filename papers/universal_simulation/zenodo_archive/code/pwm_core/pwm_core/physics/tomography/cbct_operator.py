"""CBCT (Cone-Beam Computed Tomography) operator.

Fan-beam / cone-beam projection with divergent geometry.
Extends parallel-beam CTOperator with distance-weighted line integrals
and FDK-style weighted backprojection as adjoint.

Forward: y[angle, det] = sum_along_ray x[z,x] * (D_so/dist)^2
Adjoint: x_hat[z,x] = sum_angles y[angle, s(z,x)] * (D_so/dist)^2

Mismatch ThetaSpace:
    detector_offset: [-5, 5] pixels
    projection_angle_offset: [-15, 15] degrees
    scatter_fraction: [0, 0.4]

References:
- Feldkamp, L.A., Davis, L.C. & Kress, J.W. (1984). "Practical cone-beam
  algorithm", JOSA A.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class CBCTOperator(BaseOperator):
    """Cone-beam CT operator (fan-beam geometry).

    Forward: x (ny, nx) -> y (n_angles, n_det)
        Fan-beam projection with distance weighting.

    Adjoint: y (n_angles, n_det) -> x (ny, nx)
        FDK-style weighted backprojection.

    Primitives: Pi (cone-beam projection), M (Beer-Lambert), D (photon sensor)
    """

    def __init__(
        self,
        operator_id: str = "cbct",
        theta: Optional[Dict[str, Any]] = None,
        ny: int = 64,
        nx: int = 64,
        n_angles: int = 180,
        n_det: int = 92,
        D_so: float = 100.0,
        D_sd: float = 150.0,
        detector_offset: float = 0.0,
        angle_offset_deg: float = 0.0,
        scatter_fraction: float = 0.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.ny = ny
        self.nx = nx
        self.n_angles = n_angles
        self.n_det = n_det
        self.D_so = D_so
        self.D_sd = D_sd
        self.detector_offset = detector_offset
        self.angle_offset_deg = angle_offset_deg
        self.scatter_fraction = scatter_fraction

        self._x_shape = (ny, nx)
        self._y_shape = (n_angles, n_det)
        self._is_linear = True
        self._supports_autodiff = False

        self._precompute()

    def _precompute(self) -> None:
        """Precompute projection angles and detector positions."""
        self._angles_rad = (
            np.linspace(0, np.pi, self.n_angles, endpoint=False)
            + np.deg2rad(self.angle_offset_deg)
        )
        # Detector pixel positions (centered, with offset)
        det_half = self.n_det / 2.0
        self._det_pos = np.arange(self.n_det) - det_half + self.detector_offset
        # Magnification factor
        self._magnification = self.D_sd / self.D_so

    def set_theta(self, **kwargs: Any) -> None:
        """Update mismatch parameters and recompute."""
        changed = False
        for key in ("detector_offset", "angle_offset_deg", "scatter_fraction"):
            if key in kwargs:
                setattr(self, key, float(kwargs[key]))
                changed = True
        if changed:
            self._precompute()

    def get_theta(self) -> Dict[str, float]:
        """Return current mismatch parameters."""
        return {
            "detector_offset": self.detector_offset,
            "angle_offset_deg": self.angle_offset_deg,
            "scatter_fraction": self.scatter_fraction,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Fan-beam forward projection.

        Args:
            x: Object image (ny, nx).

        Returns:
            Sinogram (n_angles, n_det).
        """
        x64 = np.asarray(x, dtype=np.float64)
        ny, nx = self.ny, self.nx
        sinogram = np.zeros(self._y_shape, dtype=np.float64)

        center_y, center_x = ny / 2.0, nx / 2.0

        for i, angle in enumerate(self._angles_rad):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Source position
            src_y = -self.D_so * sin_a
            src_x = self.D_so * cos_a

            for d_idx, d_pos in enumerate(self._det_pos):
                # Detector pixel position (on detector plane)
                det_y = self.D_sd * sin_a + d_pos * cos_a
                det_x = -self.D_sd * cos_a + d_pos * sin_a

                # Ray direction
                ray_y = det_y - src_y
                ray_x = det_x - src_x
                ray_len = np.sqrt(ray_y ** 2 + ray_x ** 2)
                ray_y /= ray_len
                ray_x /= ray_len

                # ---- Ray-box intersection: only sample within the object ----
                # Object bounding box in world coords: y in [-center_y, ny-1-center_y],
                # x in [-center_x, nx-1-center_x].  Compute t_entry, t_exit.
                t_min, t_max = 0.0, ray_len
                for dim_src, dim_ray, dim_lo, dim_hi in [
                    (src_y, ray_y, -center_y, ny - 1 - center_y),
                    (src_x, ray_x, -center_x, nx - 1 - center_x),
                ]:
                    if abs(dim_ray) > 1e-12:
                        t1 = (dim_lo - dim_src) / dim_ray
                        t2 = (dim_hi - dim_src) / dim_ray
                        if t1 > t2:
                            t1, t2 = t2, t1
                        t_min = max(t_min, t1)
                        t_max = min(t_max, t2)
                    else:
                        # Ray parallel to slab — check if inside
                        if dim_src < dim_lo or dim_src > dim_hi:
                            t_min = t_max + 1  # no intersection
                if t_min >= t_max:
                    continue

                # Dense sampling within the object region (~2 samples per pixel)
                diag = np.sqrt(ny ** 2 + nx ** 2)
                n_steps = max(int(diag * 2), 4)
                step_size = (t_max - t_min) / n_steps
                t_vals = np.linspace(t_min, t_max, n_steps + 1)
                sample_y = src_y + ray_y * t_vals + center_y
                sample_x = src_x + ray_x * t_vals + center_x

                # Bilinear interpolation (clamp to valid range)
                sy = np.clip(sample_y, 0, ny - 1 - 1e-6)
                sx = np.clip(sample_x, 0, nx - 1 - 1e-6)
                iy = np.floor(sy).astype(int)
                ix = np.floor(sx).astype(int)
                iy = np.clip(iy, 0, ny - 2)
                ix = np.clip(ix, 0, nx - 2)
                fy = sy - iy
                fx = sx - ix

                vals = (
                    x64[iy, ix] * (1 - fy) * (1 - fx)
                    + x64[iy + 1, ix] * fy * (1 - fx)
                    + x64[iy, ix + 1] * (1 - fy) * fx
                    + x64[iy + 1, ix + 1] * fy * fx
                )

                # 2D fan-beam: simple line integral (no distance weighting).
                sinogram[i, d_idx] = np.sum(vals) * step_size

        # Add scatter if specified
        if self.scatter_fraction > 0:
            scatter = np.mean(sinogram) * self.scatter_fraction
            sinogram += scatter

        return sinogram.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """FDK-style weighted backprojection.

        Args:
            y: Sinogram (n_angles, n_det).

        Returns:
            Backprojected image (ny, nx).
        """
        y64 = np.asarray(y, dtype=np.float64)
        ny, nx = self.ny, self.nx
        recon = np.zeros(self._x_shape, dtype=np.float64)

        center_y, center_x = ny / 2.0, nx / 2.0
        pixel_y = np.arange(ny, dtype=np.float64) - center_y
        pixel_x = np.arange(nx, dtype=np.float64) - center_x
        PY, PX = np.meshgrid(pixel_y, pixel_x, indexing="ij")

        det_half = self.n_det / 2.0

        for i, angle in enumerate(self._angles_rad):
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Project each pixel onto the detector.
            # Source at (-D_so sinθ, D_so cosθ); rays travel to -x side.
            # t = pixel's perpendicular coordinate (detector direction)
            #   = PY cosθ + PX sinθ
            # U = distance from source along projection direction to pixel
            #   = D_so - PX cosθ + PY sinθ  (source is at +cosθ in x)
            # SID = D_so + D_sd = total source-to-detector distance
            t = PY * cos_a + PX * sin_a
            U = self.D_so - PX * cos_a + PY * sin_a
            SID = self.D_so + self.D_sd
            fan_t = SID * t / np.maximum(U, 1e-6)

            # Convert to detector index
            det_idx = fan_t + det_half - self.detector_offset

            # Bilinear interpolation on detector
            det_floor = np.floor(det_idx).astype(int)
            frac = det_idx - det_floor
            valid = (det_floor >= 0) & (det_floor < self.n_det - 1)

            contrib = np.zeros((ny, nx), dtype=np.float64)
            contrib[valid] = (
                y64[i, det_floor[valid]] * (1 - frac[valid])
                + y64[i, det_floor[valid] + 1] * frac[valid]
            )

            # FDK weight: (D_so / U)^2
            weight = (self.D_so / np.maximum(U, 1e-6)) ** 2
            recon += contrib * weight

        recon *= np.pi / self.n_angles
        return recon.astype(np.float32)

    @property
    def x_shape(self) -> Tuple[int, ...]:
        return self._x_shape

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return self._y_shape

    @property
    def is_linear(self) -> bool:
        return True

    @property
    def supports_autodiff(self) -> bool:
        return False

    def info(self) -> Dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "ny": self.ny,
            "nx": self.nx,
            "n_angles": self.n_angles,
            "n_det": self.n_det,
            "D_so": self.D_so,
            "D_sd": self.D_sd,
            "detector_offset": self.detector_offset,
            "angle_offset_deg": self.angle_offset_deg,
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="cbct",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "y_spatial",
                "x_dim1": "x_spatial",
                "y_dim0": "angle",
                "y_dim1": "detector",
            },
            units={
                "D_so": "pixels",
                "D_sd": "pixels",
                "detector_offset": "pixels",
                "angle_offset": "degrees",
            },
        )
