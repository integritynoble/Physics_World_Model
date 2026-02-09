"""DOT (Diffuse Optical Tomography) operator.

Born-approximation forward model using the diffusion Green's function.
Forward: absorption perturbation volume -> boundary measurements via Jacobian
Adjoint: transpose of Jacobian matrix

The Jacobian is built from the diffusion equation Green's function:
    G(r1, r2) = exp(-k_d * |r1-r2|) / (4 * pi * D * |r1-r2|)
where
    D = 1 / (3 * (mu_a + mu_s'))
    k_d = sqrt(mu_a / D)

References:
- Arridge, S.R. (1999). "Optical tomography in medical imaging",
  Inverse Problems.
- Boas, D.A. et al. (2001). "Imaging the body with diffuse optical
  tomography", IEEE Signal Processing Magazine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.physics.base import BaseOperator, OperatorMetadata


class DOTOperator(BaseOperator):
    """Diffuse Optical Tomography operator (Born approximation).

    Forward: x (grid_size, grid_size, grid_size) -> y (n_sources * n_detectors,)
        y = J @ x.ravel()
        where J is the Born-approximation Jacobian built from diffusion
        Green's functions.

    Adjoint: y (n_sources * n_detectors,) -> x (grid_size, grid_size, grid_size)
        x = (J^T @ y).reshape(volume_shape)
    """

    def __init__(
        self,
        operator_id: str = "dot",
        theta: Optional[Dict[str, Any]] = None,
        n_sources: int = 8,
        n_detectors: int = 8,
        grid_size: int = 16,
        mu_a_bg: float = 0.01,
        mu_s_prime: float = 1.0,
    ):
        self.operator_id = operator_id
        self.theta = theta or {}
        self.n_sources = n_sources
        self.n_detectors = n_detectors
        self.grid_size = grid_size
        self.mu_a_bg = mu_a_bg
        self.mu_s_prime = mu_s_prime
        self.volume_shape = (grid_size, grid_size, grid_size)
        self._x_shape = (grid_size, grid_size, grid_size)
        self._y_shape = (n_sources * n_detectors,)
        self._is_linear = True
        self._supports_autodiff = False

        # Diffusion parameters
        D = 1.0 / (3.0 * (mu_a_bg + mu_s_prime))
        k_d = np.sqrt(mu_a_bg / D)

        # Build source positions on one face (x=0) of the unit cube
        s_yz = np.linspace(0.1, 0.9, n_sources)
        source_pos = np.zeros((n_sources, 3), dtype=np.float64)
        source_pos[:, 0] = 0.0  # x = 0 face
        source_pos[:, 1] = s_yz
        source_pos[:, 2] = 0.5  # centered in z

        # Build detector positions on opposite face (x=1) of the unit cube
        d_yz = np.linspace(0.1, 0.9, n_detectors)
        detector_pos = np.zeros((n_detectors, 3), dtype=np.float64)
        detector_pos[:, 0] = 1.0  # x = 1 face
        detector_pos[:, 1] = d_yz
        detector_pos[:, 2] = 0.5  # centered in z

        # Build voxel grid centers within unit cube
        voxel_edges = np.linspace(0.0, 1.0, grid_size + 1)
        voxel_centers_1d = 0.5 * (voxel_edges[:-1] + voxel_edges[1:])
        vx, vy, vz = np.meshgrid(
            voxel_centers_1d, voxel_centers_1d, voxel_centers_1d, indexing="ij"
        )
        # (n_voxels, 3)
        voxel_pos = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=-1)
        n_voxels = voxel_pos.shape[0]

        voxel_volume = (1.0 / grid_size) ** 3

        # Green's function helper (vectorized)
        def greens(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
            """Compute G(r1, r2) for all pairs.

            Args:
                r1: (N, 3) positions.
                r2: (M, 3) positions.

            Returns:
                G: (N, M) Green's function values.
            """
            # Pairwise distances: (N, M)
            diff = r1[:, np.newaxis, :] - r2[np.newaxis, :, :]  # (N, M, 3)
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (N, M)
            dist = np.maximum(dist, 1e-12)  # avoid division by zero
            G = np.exp(-k_d * dist) / (4.0 * np.pi * D * dist)
            return G

        # G_sv: (n_sources, n_voxels) -- source to voxel
        G_sv = greens(source_pos, voxel_pos)

        # G_vd: (n_voxels, n_detectors) -- voxel to detector
        G_vd = greens(voxel_pos, detector_pos)

        # G_sd: (n_sources, n_detectors) -- source to detector (direct)
        G_sd = greens(source_pos, detector_pos)

        # Build Jacobian: J[m, v] where m indexes (source, detector) pair
        # m = s * n_detectors + d
        # J[m, v] = -voxel_volume * G(r_s, r_v) * G(r_v, r_d) / G(r_s, r_d)
        n_measurements = n_sources * n_detectors
        jacobian = np.zeros((n_measurements, n_voxels), dtype=np.float64)

        for s in range(n_sources):
            for d in range(n_detectors):
                m = s * n_detectors + d
                # G_sv[s, :] -> (n_voxels,), G_vd[:, d] -> (n_voxels,)
                jacobian[m, :] = (
                    -voxel_volume * G_sv[s, :] * G_vd[:, d] / G_sd[s, d]
                )

        self.jacobian = jacobian
        self.source_pos = source_pos
        self.detector_pos = detector_pos

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute boundary measurements from absorption perturbation.

        Args:
            x: Absorption perturbation volume (grid_size, grid_size, grid_size).

        Returns:
            Boundary measurements (n_sources * n_detectors,).
        """
        x64 = x.astype(np.float64).ravel()
        y = self.jacobian @ x64
        return y.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Back-project measurements to absorption perturbation volume.

        Args:
            y: Boundary measurements (n_sources * n_detectors,).

        Returns:
            Absorption perturbation volume (grid_size, grid_size, grid_size).
        """
        y64 = y.astype(np.float64).ravel()
        x = self.jacobian.T @ y64
        return x.reshape(self.volume_shape).astype(np.float32)

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
            "n_sources": self.n_sources,
            "n_detectors": self.n_detectors,
            "grid_size": self.grid_size,
            "mu_a_bg": self.mu_a_bg,
            "mu_s_prime": self.mu_s_prime,
            "jacobian_shape": list(self.jacobian.shape),
        }

    def metadata(self) -> OperatorMetadata:
        return OperatorMetadata(
            modality="diffuse_optical_tomography",
            operator_id=self.operator_id,
            x_shape=list(self.x_shape),
            y_shape=list(self.y_shape),
            is_linear=True,
            supports_autodiff=False,
            axes={
                "x_dim0": "voxel_x",
                "x_dim1": "voxel_y",
                "x_dim2": "voxel_z",
                "y_dim0": "source_detector_pair",
            },
            wavelength_range_nm=[650.0, 900.0],
            units={
                "mu_a": "mm^-1",
                "mu_s_prime": "mm^-1",
                "measurement": "relative_intensity",
            },
        )
