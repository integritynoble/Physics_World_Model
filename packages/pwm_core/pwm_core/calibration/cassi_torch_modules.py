"""CASSI PyTorch Differentiable Modules

This module provides differentiable PyTorch implementations of core CASSI operators
for gradient-based mismatch calibration. All modules are designed to be fully
differentiable through automatic differentiation.

Key classes:
- RoundSTE: Straight-Through Estimator for integer rounding
- DifferentiableMaskWarpFixed: Differentiable 2D affine mask warping
- DifferentiableCassiForwardSTE: Differentiable CASSI forward operator with STE integer offsets
- DifferentiableCassiAdjointSTE: Differentiable CASSI adjoint (back-projection) operator
- DifferentiableGAPTV: Unrolled differentiable GAP-TV solver with Gaussian denoising
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PI = 3.141592653589793


class RoundSTE(torch.autograd.Function):
    """Straight-Through Estimator for integer rounding.

    Rounds values to nearest integers in forward pass while passing gradients
    through in backward pass (identity gradient). Enables gradient flow through
    discrete dispersion offsets.

    Usage:
        x_rounded = RoundSTE.apply(x)  # x can be float, output is rounded integers
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer.

        Args:
            x: Input tensor (float)

        Returns:
            Rounded tensor (float, with integer values)
        """
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass gradients through unchanged (identity).

        Args:
            grad_output: Upstream gradients

        Returns:
            Same gradients (identity mapping)
        """
        return grad_output


class DifferentiableMaskWarpFixed(nn.Module):
    """Differentiable 2D affine warping for coded aperture masks.

    Matches scipy.ndimage.affine_transform convention exactly:
    - output[o] = input[M @ o + offset]
    - where M = R^T (transpose rotation), offset = (center - shift) - M @ center
    - Positive dx = shift right, positive dy = shift down

    Uses PyTorch F.affine_grid and F.grid_sample for differentiable warping.

    Attributes:
        dx: Translation in x-direction (columns), in pixels. Learnable parameter.
        dy: Translation in y-direction (rows), in pixels. Learnable parameter.
        theta_deg: Rotation angle in degrees. Learnable parameter.
        mask_nom: Buffer storing nominal (unwarped) mask [1, 1, H, W]

    Critical sign convention:
        F.affine_grid maps output coords to input coords in [-1,1] normalized space.
        Translation: tx = -2*dx/W, ty = -2*dy/H (matches scipy exactly)
    """

    def __init__(
        self,
        mask2d_nominal: np.ndarray,
        dx_init: float = 0.0,
        dy_init: float = 0.0,
        theta_init: float = 0.0,
    ):
        """Initialize differentiable mask warping module.

        Args:
            mask2d_nominal: Nominal (unwarped) mask [H, W] as numpy array
            dx_init: Initial x-translation in pixels (default 0.0)
            dy_init: Initial y-translation in pixels (default 0.0)
            theta_init: Initial rotation angle in degrees (default 0.0)
        """
        super().__init__()
        # Learnable parameters
        self.dx = nn.Parameter(torch.tensor(dx_init, dtype=torch.float32))
        self.dy = nn.Parameter(torch.tensor(dy_init, dtype=torch.float32))
        self.theta_deg = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        # Buffer for nominal mask (no gradients)
        mask_t = torch.from_numpy(mask2d_nominal.astype(np.float32))
        self.register_buffer("mask_nom", mask_t.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]

    def forward(self) -> torch.Tensor:
        """Warp nominal mask by current affine parameters.

        Returns:
            Warped mask [H, W] with gradients w.r.t. dx, dy, theta_deg
        """
        theta_rad = self.theta_deg * (PI / 180.0)
        cos_t = torch.cos(theta_rad)
        sin_t = torch.sin(theta_rad)
        H, W = self.mask_nom.shape[2], self.mask_nom.shape[3]

        # Build affine transformation matrix for F.affine_grid
        # Matches scipy affine_transform convention:
        # scipy: output[o] = input[R^T @ (o - center) + center - shift]
        # F.affine_grid: maps output normalized coords to input normalized coords
        # For rotation about center (norm=0) + translation:
        #   col_in = cos*col_out - sin*row_out + tx
        #   row_in = sin*col_out + cos*row_out + ty
        # Sign convention: positive dx = shift right => tx = -2*dx/W
        #                  positive dy = shift down  => ty = -2*dy/H
        tx = -2.0 * self.dx / float(W)
        ty = -2.0 * self.dy / float(H)

        row0 = torch.stack([cos_t, -sin_t, tx])
        row1 = torch.stack([sin_t, cos_t, ty])
        affine = torch.stack([row0, row1]).unsqueeze(0)  # [1, 2, 3]

        grid = F.affine_grid(affine, self.mask_nom.shape, align_corners=False)
        warped = F.grid_sample(
            self.mask_nom,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.squeeze(0).squeeze(0).clamp(0.0, 1.0)  # [H, W]


class DifferentiableCassiForwardSTE(nn.Module):
    """Differentiable CASSI forward operator with STE integer-offset dispersion.

    Computes integer spectral dispersion offsets using RoundSTE to enable
    gradients to flow through phi_d (dispersion axis angle) while maintaining
    discrete integer shifts for measurement simulation.

    Attributes:
        s_nom: Buffer storing nominal dispersion curve [L] as pixels/band

    Key design:
        - Integer offsets via RoundSTE: dx_i = RoundSTE(dx_f), dy_i = RoundSTE(dy_f)
        - Enables gradients to flow through phi_d despite integer rounding
        - Matches measurement simulation exactly (integer offsets via np.rint)
    """

    def __init__(self, s_nom: np.ndarray):
        """Initialize CASSI forward operator.

        Args:
            s_nom: Nominal dispersion curve [L] as pixels/band (numpy array)
        """
        super().__init__()
        self.register_buffer("s_nom", torch.from_numpy(s_nom.astype(np.float32)))

    def forward(
        self,
        x_1lhw: torch.Tensor,
        mask2d: torch.Tensor,
        phi_d_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CASSI measurement from spectral cube.

        Applies spectral dispersion (phi_d_deg) and coded aperture (mask2d) to
        produce measurement. Uses STE for integer offset rounding.

        Args:
            x_1lhw: Spectral cube [1, L, H, W] with gradients
            mask2d: Warped coded aperture mask [H, W] with gradients
            phi_d_deg: Dispersion axis angle (scalar) with gradients

        Returns:
            Simulated measurement [1, Hp, Wp] where Hp, Wp depend on max dispersion
        """
        L, H, W = x_1lhw.shape[1], x_1lhw.shape[2], x_1lhw.shape[3]

        # Convert dispersion angle to radians
        phi_rad = phi_d_deg * (PI / 180.0)
        dx_f = self.s_nom * torch.cos(phi_rad)
        dy_f = self.s_nom * torch.sin(phi_rad)

        # Shift to non-negative offsets
        dx_f = dx_f - dx_f.min()
        dy_f = dy_f - dy_f.min()

        # STE rounding: round in forward, identity gradient in backward
        dx_i = RoundSTE.apply(dx_f)
        dy_i = RoundSTE.apply(dy_f)

        # Compute canvas size from detached max (discrete integer)
        max_dx = int(dx_i.detach().max().item())
        max_dy = int(dy_i.detach().max().item())
        Wp = W + max_dx
        Hp = H + max_dy

        # Accumulate shifted masked frames
        y = x_1lhw.new_zeros(1, Hp, Wp)
        for l in range(L):
            ox = int(dx_i[l].detach().item())
            oy = int(dy_i[l].detach().item())
            band = mask2d * x_1lhw[0, l, :, :]  # [H, W]
            y[0, oy : oy + H, ox : ox + W] = y[0, oy : oy + H, ox : ox + W] + band

        return y


class DifferentiableCassiAdjointSTE(nn.Module):
    """Differentiable CASSI adjoint (back-projection) operator with STE integer offsets.

    Computes transpose of forward operator: back-projects measurement to spectral
    cube domain. Uses RoundSTE for differentiable integer dispersion offsets.

    Attributes:
        s_nom: Buffer storing nominal dispersion curve [L] as pixels/band
    """

    def __init__(self, s_nom: np.ndarray):
        """Initialize CASSI adjoint operator.

        Args:
            s_nom: Nominal dispersion curve [L] as pixels/band (numpy array)
        """
        super().__init__()
        self.register_buffer("s_nom", torch.from_numpy(s_nom.astype(np.float32)))

    def forward(
        self,
        y_1hw: torch.Tensor,
        mask2d: torch.Tensor,
        phi_d_deg: torch.Tensor,
        H: int,
        W: int,
        L: int,
    ) -> torch.Tensor:
        """Back-project measurement to spectral cube.

        Extracts regions from measurement at dispersion offsets, applies
        coded aperture mask to produce spectral bands.

        Args:
            y_1hw: Measurement [1, Hp, Wp] with gradients
            mask2d: Warped coded aperture mask [H, W] with gradients
            phi_d_deg: Dispersion axis angle (scalar) with gradients
            H, W, L: Target cube dimensions

        Returns:
            Back-projected spectral cube [1, L, H, W]
        """
        # Compute integer dispersion offsets (same as forward)
        phi_rad = phi_d_deg * (PI / 180.0)
        dx_f = self.s_nom * torch.cos(phi_rad)
        dy_f = self.s_nom * torch.sin(phi_rad)
        dx_f = dx_f - dx_f.min()
        dy_f = dy_f - dy_f.min()
        dx_i = RoundSTE.apply(dx_f)
        dy_i = RoundSTE.apply(dy_f)

        # Extract regions at each band's dispersion offset
        x = y_1hw.new_zeros(1, L, H, W)
        for l in range(L):
            ox = int(dx_i[l].detach().item())
            oy = int(dy_i[l].detach().item())
            x[0, l, :, :] = y_1hw[0, oy : oy + H, ox : ox + W] * mask2d

        return x


class DifferentiableGAPTV(nn.Module):
    """Unrolled differentiable GAP-TV solver for iterative reconstruction.

    Replaces Total Variation minimization (TV-Chambolle) with Gaussian denoising
    for full differentiability through automatic differentiation. Supports gradient
    checkpointing for memory-efficient training.

    The algorithm alternates between:
    1. Measurement residual update (data fitting)
    2. Gaussian denoising (regularization)

    Attributes:
        H, W, L: Spectral cube dimensions
        n_iter: Number of GAP iterations
        gauss_sigma: Gaussian filter standard deviation (pixels)
        use_checkpointing: Enable gradient checkpointing for memory efficiency
        fwd_op: DifferentiableCassiForwardSTE instance
        adj_op: DifferentiableCassiAdjointSTE instance
        gauss_kernel: Depthwise Gaussian convolution kernel [L, 1, k, k]
    """

    def __init__(
        self,
        s_nom: np.ndarray,
        H: int,
        W: int,
        L: int,
        n_iter: int = 12,
        gauss_sigma: float = 0.5,
        use_checkpointing: bool = True,
    ):
        """Initialize differentiable GAP-TV solver.

        Args:
            s_nom: Nominal dispersion curve [L] as pixels/band (numpy array)
            H: Spatial height of spectral cube
            W: Spatial width of spectral cube
            L: Number of spectral bands
            n_iter: Number of iterations (default 12)
            gauss_sigma: Gaussian filter sigma in pixels (default 0.5)
            use_checkpointing: Enable gradient checkpointing (default True)
        """
        super().__init__()
        self.H, self.W, self.L = H, W, L
        self.n_iter = n_iter
        self.gauss_sigma = gauss_sigma
        self.use_checkpointing = use_checkpointing

        # Create operators
        self.fwd_op = DifferentiableCassiForwardSTE(s_nom)
        self.adj_op = DifferentiableCassiAdjointSTE(s_nom)

        # Build Gaussian kernel (fixed, not learned)
        self._build_gauss_kernel(gauss_sigma, L)

    def _build_gauss_kernel(self, sigma: float, L: int) -> None:
        """Create depthwise Gaussian convolution kernel.

        Args:
            sigma: Gaussian standard deviation in pixels
            L: Number of spectral bands
        """
        if sigma <= 0:
            self.register_buffer("gauss_kernel", None)
            return

        # Compute kernel size (ensure odd)
        ksize = max(3, int(6 * sigma + 1) | 1)
        ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        g1d = g1d / g1d.sum()

        # Create 2D Gaussian kernel
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # [k, k]

        # Expand to depthwise format [L, 1, k, k]
        kernel = (
            g2d.unsqueeze(0)
            .unsqueeze(0)
            .expand(L, 1, ksize, ksize)
            .contiguous()
        )
        self.register_buffer("gauss_kernel", kernel)
        self.gauss_pad = ksize // 2

    def _gauss_denoise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing per band using depthwise convolution.

        Args:
            x: Spectral cube [1, L, H, W]

        Returns:
            Denoised cube [1, L, H, W]
        """
        if self.gauss_kernel is None:
            return x
        return F.conv2d(
            x, self.gauss_kernel, padding=self.gauss_pad, groups=self.L
        )

    def _compute_phi_sum(
        self, mask2d: torch.Tensor, phi_d_deg: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalization factor Phi_sum = sum_l mask shifted by band offset.

        Used to normalize residual in GAP-TV update. Accounts for varying
        coverage at different spatial locations due to spectral dispersion.

        Args:
            mask2d: Warped coded aperture mask [H, W]
            phi_d_deg: Dispersion axis angle (scalar)

        Returns:
            Normalization map [Hp, Wp] (clamped to >=1 to avoid division by zero)
        """
        phi_rad = phi_d_deg * (PI / 180.0)
        dx_f = self.fwd_op.s_nom * torch.cos(phi_rad)
        dy_f = self.fwd_op.s_nom * torch.sin(phi_rad)
        dx_f = dx_f - dx_f.min()
        dy_f = dy_f - dy_f.min()
        dx_i = RoundSTE.apply(dx_f)
        dy_i = RoundSTE.apply(dy_f)

        max_dx = int(dx_i.detach().max().item())
        max_dy = int(dy_i.detach().max().item())
        Wp = self.W + max_dx
        Hp = self.H + max_dy

        Phi_sum = mask2d.new_zeros(Hp, Wp)
        for l in range(self.L):
            ox = int(dx_i[l].detach().item())
            oy = int(dy_i[l].detach().item())
            Phi_sum[oy : oy + self.H, ox : ox + self.W] = (
                Phi_sum[oy : oy + self.H, ox : ox + self.W] + mask2d
            )
        return Phi_sum.clamp(min=1.0)

    def _gap_tv_iteration(
        self,
        x: torch.Tensor,
        y1: torch.Tensor,
        y: torch.Tensor,
        mask2d: torch.Tensor,
        phi_d_deg: torch.Tensor,
        Phi_sum: torch.Tensor,
        lam: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single GAP-TV iteration (compatible with gradient checkpointing).

        Performs one step of alternating projections between measurement
        consistency and regularization (Gaussian denoising).

        Args:
            x: Current estimate [1, L, H, W]
            y1: Measurement residual accumulator [1, Hp, Wp]
            y: True measurement [1, Hp, Wp]
            mask2d: Warped coded aperture mask [H, W]
            phi_d_deg: Dispersion axis angle (scalar)
            Phi_sum: Normalization map [Hp, Wp]
            lam: Step size for residual update (default 1.0)

        Returns:
            (updated_x, updated_y1) both with gradients
        """
        # Simulate measurement from current estimate
        yb = self.fwd_op(x, mask2d, phi_d_deg)  # [1, Hp, Wp]

        # Handle size mismatches (measurement may be smaller)
        hh = min(y.shape[1], yb.shape[1])
        ww = min(y.shape[2], yb.shape[2])

        # Update measurement residual accumulator
        y1_new = y1.clone()
        y1_new[:, :hh, :ww] = y1[:, :hh, :ww] + (y[:, :hh, :ww] - yb[:, :hh, :ww])

        # Normalize residual and back-project
        residual = y1_new.clone()
        residual[:, :hh, :ww] = (
            y1_new[:, :hh, :ww] - yb[:, :hh, :ww]
        ) / Phi_sum[:hh, :ww].unsqueeze(0)

        # Back-projection with step size lam
        x = x + lam * self.adj_op(residual, mask2d, phi_d_deg, self.H, self.W, self.L)

        # Gaussian denoising (regularization)
        x = self._gauss_denoise(x)
        x = x.clamp(0.0, 1.0)

        return x, y1_new

    def forward(
        self,
        y: torch.Tensor,
        mask2d: torch.Tensor,
        phi_d_deg: torch.Tensor,
    ) -> torch.Tensor:
        """Solve inverse problem: reconstruct spectral cube from measurement.

        Iteratively applies GAP-TV algorithm (data fidelity + Gaussian denoising)
        starting from adjoint-based initialization.

        Args:
            y: Measurement [1, Hy, Wy] with gradients
            mask2d: Warped coded aperture mask [H, W] with gradients
            phi_d_deg: Dispersion axis angle (scalar) with gradients

        Returns:
            Reconstructed spectral cube [1, L, H, W]
        """
        # Compute normalization factor Phi_sum
        Phi_sum = self._compute_phi_sum(mask2d, phi_d_deg)

        # Pad measurement if needed
        Hp, Wp = Phi_sum.shape
        y_pad = y.new_zeros(1, Hp, Wp)
        hh = min(y.shape[1], Hp)
        ww = min(y.shape[2], Wp)
        y_pad[:, :hh, :ww] = y[:, :hh, :ww]

        # Initial estimate via adjoint
        x = self.adj_op(
            y_pad / Phi_sum.unsqueeze(0),
            mask2d,
            phi_d_deg,
            self.H,
            self.W,
            self.L,
        )  # [1, L, H, W]
        y1 = y_pad.clone()

        # GAP-TV iterations
        for _ in range(self.n_iter):
            if self.use_checkpointing and self.training:
                x, y1 = torch.utils.checkpoint.checkpoint(
                    self._gap_tv_iteration,
                    x,
                    y1,
                    y_pad,
                    mask2d,
                    phi_d_deg,
                    Phi_sum,
                    1.0,
                    use_reentrant=False,
                )
            else:
                x, y1 = self._gap_tv_iteration(
                    x, y1, y_pad, mask2d, phi_d_deg, Phi_sum, 1.0
                )

        return x


__all__ = [
    "RoundSTE",
    "DifferentiableMaskWarpFixed",
    "DifferentiableCassiForwardSTE",
    "DifferentiableCassiAdjointSTE",
    "DifferentiableGAPTV",
]
