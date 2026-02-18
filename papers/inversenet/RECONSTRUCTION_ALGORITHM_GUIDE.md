# SPC & CACTI Reconstruction Algorithm Development Guide

This guide explains how to develop reconstruction algorithms for Single-Pixel Camera (SPC) and Coded Aperture Compressive Temporal Imaging (CACTI) following the PWM framework patterns.

## Architecture Overview

### 1. Classical Methods (Compressive Sensing)
Location: `/packages/pwm_core/pwm_core/recon/`

**For SPC:** Use iterative optimization solvers
- ISTA/FISTA with L1 regularization (basis pursuit)
- TVAL3 with TV regularization
- ADMM with sparse priors

**For CACTI:** Use TV regularization (temporal + spatial)
- GAP-TV: Gradient Ascent Proximal TV
- SART-TV: Simultaneous Algebraic Reconstruction with TV

### 2. Deep Learning Methods

**Unrolled Networks:** Unroll classical iterations with learnable parameters
- ISTA-Net: Unroll ISTA iterations (20-30 layers)
- ELP-Unfolding: Unroll ADMM with Vision Transformer blocks

**End-to-End Networks:** Direct mapping from measurement to reconstruction
- HATNet: Hybrid attention Transformer
- EfficientSCI: Densely connected with space-time factorization
- MST: Multi-stage Transformer

## Pattern 1: Classical Iterative Solvers

### Template: ADMM for SPC

```python
"""ADMM Solver for Single-Pixel Camera.

Solves: min_x ||x||_1 + (1/2) ||Ax - y||_2^2 (Basis Pursuit Denoising)
Using: Alternating Direction Method of Multipliers
"""

import numpy as np
from typing import Callable, Tuple

def admm_spc(
    y: np.ndarray,
    A: np.ndarray,  # or callable forward operator
    rho: float = 1.0,
    iterations: int = 100,
    tol: float = 1e-4,
    verbose: bool = False
) -> np.ndarray:
    """ADMM for SPC L1 minimization.

    Args:
        y: Measurement vector (M,)
        A: Forward operator matrix (M, N) or callable
        rho: ADMM penalty parameter
        iterations: Max iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        x: Reconstructed signal (N,)
    """
    M = y.shape[0]
    if callable(A):
        N = A.shape[1] if hasattr(A, 'shape') else estimate_N(A, M)
        A_fn = A
        def A_op(x): return A_fn(x)
        def AtA_op(x): return A_fn.T(A_fn(x))
        def Aty_op(y): return A_fn.T(y)
    else:
        N = A.shape[1]
        A_op = lambda x: A @ x
        AtA_op = lambda x: A.T @ (A @ x)
        Aty_op = lambda y: A.T @ y

    # Initialize
    x = np.zeros(N, dtype=np.float32)
    z = np.zeros(N, dtype=np.float32)
    u = np.zeros(N, dtype=np.float32)

    Aty = Aty_op(y)

    for k in range(iterations):
        x_old = x.copy()

        # Step 1: Update x via soft-thresholding of z
        # Minimize: (1/2) ||Ax - y||_2^2 + (rho/2) ||x - z + u||_2^2
        # Solution: x = (A^T A + rho*I)^{-1} (A^T y + rho(z - u))
        try:
            # Use LU decomposition for stability
            from scipy.linalg import solve
            ATA = AtA_op(np.eye(N))  # Compute A^T A
            rhs = Aty + rho * (z - u)
            x = solve(ATA + rho * np.eye(N), rhs)
        except:
            # Fallback: simple gradient step
            grad = AtA_op(x - np.zeros(N)) - Aty + rho * (x - z + u)
            x = x - 0.01 * grad

        # Step 2: Soft-thresholding (proximal operator of L1)
        # Minimize: ||z||_1 + (rho/2) ||z - x - u||_2^2
        threshold = 1.0 / rho  # Tunable parameter
        z = soft_threshold(x + u, threshold)

        # Step 3: Update dual variable
        u = u + x - z

        # Convergence check
        residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-8)
        if verbose and k % 10 == 0:
            print(f"Iter {k}: residual={residual:.2e}")
        if residual < tol:
            if verbose:
                print(f"Converged at iteration {k}")
            break

    return np.clip(x.reshape(-1), 0, 1).astype(np.float32)


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator for L1 norm: prox_tau ||·||_1."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
```

### Template: GAP-TV for CACTI

```python
"""GAP-TV Solver for CACTI.

Solves: min_x (1/2) ||sum_t(M_t * x_t) - y||_2^2 + lambda * TV(x)
Using: Gradient Ascent Proximal - Total Variation
"""

def gap_tv_cacti(
    y: np.ndarray,  # (H, W) measurement
    mask: np.ndarray,  # (H, W, T) temporal masks
    lambda_tv: float = 0.05,
    iterations: int = 50,
    step_size: float = 0.01,
    verbose: bool = False
) -> np.ndarray:
    """GAP-TV for CACTI video reconstruction.

    Args:
        y: Measurement (H, W)
        mask: Temporal coded masks (H, W, T)
        lambda_tv: TV regularization weight
        iterations: Number of iterations
        step_size: Gradient step size
        verbose: Print progress

    Returns:
        x: Reconstructed video (H, W, T)
    """
    H, W, T = mask.shape

    # Initialize with simple inverse: x_0 = M^T y / T
    x = np.zeros((H, W, T), dtype=np.float32)
    for t in range(T):
        x[:, :, t] = y * mask[:, :, t] / T

    for k in range(iterations):
        # Compute forward model
        y_recon = np.zeros((H, W), dtype=np.float32)
        for t in range(T):
            y_recon += mask[:, :, t] * x[:, :, t]

        # Residual
        residual = y_recon - y  # (H, W)

        # Backproject: gradient of ||sum(M_t x_t) - y||_2^2
        grad_fid = np.zeros_like(x)
        for t in range(T):
            grad_fid[:, :, t] = 2 * mask[:, :, t] * residual

        # TV gradient (backward differences)
        grad_tv = compute_tv_gradient(x)

        # Update: x = x - step_size * (grad_fid + lambda * grad_tv)
        x = x - step_size * (grad_fid + lambda_tv * grad_tv)

        # Clip to valid range
        x = np.clip(x, 0, 1)

        if verbose and k % 10 == 0:
            residual_norm = np.linalg.norm(residual)
            tv_norm = compute_tv_norm(x)
            print(f"Iter {k}: ||residual||={residual_norm:.2e}, TV={tv_norm:.2e}")

    return x


def compute_tv_gradient(x: np.ndarray) -> np.ndarray:
    """Compute gradient of TV norm (isotropic)."""
    H, W, T = x.shape
    grad = np.zeros_like(x)

    # Spatial differences
    for t in range(T):
        dx = np.diff(x[:, :, t], axis=1, prepend=0)  # x-direction
        dy = np.diff(x[:, :, t], axis=0, prepend=0)  # y-direction

        # Finite differences for divergence
        grad[:-1, :, t] += dy[:-1, :]
        grad[-1, :, t] += dy[-1, :]
        grad[:, :-1, t] += dx[:, :-1]
        grad[:, -1, t] += dx[:, -1]

    # Temporal differences (optional, for 3D TV)
    for h in range(H):
        for w in range(W):
            dt = np.diff(x[h, w, :], prepend=0)
            grad[h, w, :-1] += dt[:-1]
            grad[h, w, -1] += dt[-1]

    return grad


def compute_tv_norm(x: np.ndarray) -> float:
    """Compute isotropic TV norm."""
    tv = 0.0
    for t in range(x.shape[2]):
        dx = np.diff(x[:, :, t], axis=1)
        dy = np.diff(x[:, :, t], axis=0)
        tv += np.sum(np.sqrt(dx**2 + dy**2 + 1e-10))
    return float(tv)
```

## Pattern 2: Deep Unrolled Networks

### Template: ISTA-Net+ for SPC

```python
"""ISTA-Net+ for SPC: Unrolled Iterative Thresholding with Learnable Parameters.

Unrolls ISTA iterations: x_{k+1} = soft_threshold(x_k - step * A^T(Ax_k - y), tau_k)
Each layer learns: step_size, threshold_tau, and optionally A.
"""

import torch
import torch.nn as nn

class ISTANetPlus(nn.Module):
    """ISTA-Net+ with learnable soft-thresholding."""

    def __init__(
        self,
        num_layers: int = 30,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 32,
        learn_A: bool = False,
    ):
        """
        Args:
            num_layers: Number of unrolled ISTA iterations
            input_channels: Input measurement dimension (reshaped as image)
            output_channels: Output image channels
            hidden_channels: Hidden channels in each layer's refinement network
            learn_A: Whether to learn the forward operator A
        """
        super().__init__()

        self.num_layers = num_layers
        self.learn_A = learn_A

        # Learnable parameters per layer
        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
            for _ in range(num_layers)
        ])

        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            for _ in range(num_layers)
        ])

        # Optional: Learn A matrix (or keep fixed)
        if learn_A:
            self.A = nn.Parameter(torch.randn(512, 256) / 32)  # Example: 512→256
        else:
            self.register_buffer('A', torch.randn(512, 256) / 32)

        # Refinement CNN per layer (optional, for learned denoising)
        self.refine_nets = nn.ModuleList([
            self._build_refine_block(output_channels, hidden_channels)
            for _ in range(num_layers)
        ])

    def _build_refine_block(self, in_ch: int, hidden_ch: int) -> nn.Sequential:
        """Build a simple refinement network."""
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, 3, 1, 1),
        )

    def forward(
        self,
        y: torch.Tensor,  # Measurement (B, M)
        A: torch.Tensor = None,  # Optional custom A
    ) -> torch.Tensor:
        """
        Args:
            y: Measurements (B, M)
            A: Forward operator (M, N), defaults to self.A

        Returns:
            x: Reconstructed signals (B, H, W, C)
        """
        if A is None:
            A = self.A

        B, M = y.shape
        N = A.shape[1]

        # Initialize x via A^T y
        AtA = A.T @ A + 1e-3 * torch.eye(N, device=A.device)
        Aty = A.T @ y  # (B, N)
        x = torch.linalg.solve(AtA.unsqueeze(0), Aty.unsqueeze(1)).squeeze(1)  # (B, N)

        # Reshape to image
        H = W = int(np.sqrt(N))
        x = x.reshape(B, 1, H, W)  # (B, C, H, W)

        # Unroll ISTA iterations
        for k in range(self.num_layers):
            # Forward pass: y_pred = A @ x
            x_flat = x.reshape(B, -1)
            y_pred = x_flat @ A.T  # (B, M)

            # Gradient: A^T (y_pred - y)
            grad = (y_pred - y) @ A  # (B, N)
            grad = grad.reshape(B, 1, H, W)

            # Gradient step
            x = x - self.step_sizes[k] * grad

            # Soft thresholding
            x = self._soft_threshold(x, self.thresholds[k])

            # Refinement network (optional)
            x_refined = self.refine_nets[k](x)
            x = x + 0.1 * x_refined

            # Clip to [0, 1]
            x = torch.clamp(x, 0, 1)

        return x

    @staticmethod
    def _soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Soft thresholding: sign(x) * max(|x| - tau, 0)."""
        return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)


# Usage example:
# model = ISTANetPlus(num_layers=30, learn_A=False)
# y_meas = torch.randn(batch_size, 512)
# x_recon = model(y_meas)  # Output: (batch_size, 1, 256, 256)
```

### Template: ELP-Unfolding for CACTI

```python
"""ELP-Unfolding for CACTI: Unrolled ADMM with Vision Transformer.

Unfolds ADMM iterations with learnable parameters and Transformer blocks.
"""

class ELPUnfoldingCACTI(nn.Module):
    """ELP-Unfolding with Vision Transformer for CACTI."""

    def __init__(
        self,
        num_primal_layers: int = 8,
        num_dual_layers: int = 5,
        hidden_dim: int = 64,
        num_heads: int = 8,
    ):
        """
        Args:
            num_primal_layers: ADMM primal steps
            num_dual_layers: ADMM dual steps
            hidden_dim: Transformer embedding dimension
            num_heads: Multi-head attention heads
        """
        super().__init__()

        self.num_primal = num_primal_layers
        self.num_dual = num_dual_layers

        # Learnable rho (ADMM penalty)
        self.rho = nn.Parameter(torch.tensor(1.0))

        # Transformer blocks for refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(
        self,
        y: torch.Tensor,  # (B, H, W)
        mask: torch.Tensor,  # (B, H, W, T)
    ) -> torch.Tensor:
        """
        Args:
            y: Measurement (B, H, W)
            mask: Temporal masks (B, H, W, T)

        Returns:
            x: Reconstructed video (B, H, W, T)
        """
        B, H, W, T = mask.shape

        # Initialize
        x = torch.zeros(B, H, W, T, device=y.device)
        z = torch.zeros(B, H, W, T, device=y.device)
        u = torch.zeros(B, H, W, T, device=y.device)

        # ADMM iteration (primal-dual)
        for k in range(self.num_primal):
            # Primal step: update x
            # Minimize: (1/2) ||sum_t(M_t * x_t) - y||^2 + rho/2 ||x - z + u||^2
            y_recon = (mask * x).sum(dim=3)  # (B, H, W)
            residual = y_recon - y  # (B, H, W)

            # Gradient descent
            for t in range(T):
                grad_t = 2 * mask[:, :, :, t] * residual + self.rho * (x[:, :, :, t] - z[:, :, :, t] + u[:, :, :, t])
                x[:, :, :, t] = x[:, :, :, t] - 0.01 * grad_t

            # Dual step: update z (with Transformer refinement)
            z_candidate = x + u

            # Reshape for Transformer
            z_flat = z_candidate.reshape(B * H * W, T)
            z_refined = self.transformer(z_flat.unsqueeze(1)).squeeze(1)  # Apply attention
            z = z_refined.reshape(B, H, W, T)

            # Apply soft thresholding (optional, for sparsity)
            z = torch.sign(z) * torch.clamp(torch.abs(z) - 0.01, min=0)

            # Update dual variable
            u = u + x - z

            # Clip to valid range
            x = torch.clamp(x, 0, 1)

        return x
```

## Pattern 3: End-to-End Networks

### Template: HATNet for SPC

```python
"""HATNet: Hybrid Attention Transformer for SPC.

Two-stage:
1. Initial reconstruction via pseudo-inverse
2. Learned refinement with attention mechanism
"""

class HATNetSPC(nn.Module):
    """Hybrid Attention Transformer for SPC reconstruction."""

    def __init__(
        self,
        image_size: int = 256,
        num_heads: int = 8,
        num_blocks: int = 4,
        hidden_dim: int = 256,
    ):
        """
        Args:
            image_size: Spatial dimension (64 or 256)
            num_heads: Attention heads
            num_blocks: Number of transformer blocks
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.image_size = image_size

        # Stage 1: Initial estimation via learned linear layer
        self.linear = nn.Linear(614, image_size * image_size)  # SPC: 614 measurements

        # Stage 2: Spatial-spectral attention blocks
        self.blocks = nn.ModuleList([
            self._build_attention_block(hidden_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # Stage 3: Refinement and output
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def _build_attention_block(self, dim: int, num_heads: int) -> nn.Module:
        """Build spatial attention block."""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(dim, num_heads, batch_first=True),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Measurement (B, 614)

        Returns:
            x: Reconstructed image (B, 1, 256, 256)
        """
        B = y.shape[0]

        # Stage 1: Linear initialization
        x = self.linear(y)  # (B, 256*256)
        x = x.reshape(B, 1, self.image_size, self.image_size)  # (B, 1, H, W)

        # Stage 2: Attention refinement
        for block in self.blocks:
            # Reshape for attention: (B, H*W, C)
            x_flat = x.reshape(B, 1, -1).transpose(1, 2)  # (B, H*W, 1)

            # Apply self-attention
            x_attn, _ = block[1](x_flat, x_flat, x_flat)
            x_flat = x_flat + 0.1 * x_attn

            # Back to spatial
            x = x_flat.transpose(1, 2).reshape(B, 1, self.image_size, self.image_size)

        # Stage 3: Refinement
        x = self.refine(x)

        # Output
        return torch.clamp(x, 0, 1)
```

## Integration Steps

### Step 1: Register in `/packages/pwm_core/pwm_core/recon/__init__.py`

```python
# Add to __init__.py
from .spc_solvers import admm_spc, ista_net_plus_spc
from .cacti_solvers import gap_tv_cacti, elp_unfolding_cacti, efficient_sci_cacti

__all__ = [
    # SPC
    'admm_spc',
    'ista_net_plus_spc',

    # CACTI
    'gap_tv_cacti',
    'elp_unfolding_cacti',
    'efficient_sci_cacti',

    # ... existing methods
]
```

### Step 2: Create Wrapper Module Files

**File:** `/packages/pwm_core/pwm_core/recon/spc_solvers.py`
**File:** `/packages/pwm_core/pwm_core/recon/cacti_solvers.py`

Each file should follow the pattern:
1. Docstring with references and expected benchmarks
2. Classical solver(s)
3. Utility functions (soft_threshold, TV_gradient, etc.)
4. Error handling and logging

### Step 3: Add to Benchmark Runner

```python
# In benchmarks/run_all.py

def run_spc_benchmark(self) -> Dict:
    """Benchmark SPC with 3 methods."""
    from pwm_core.recon import admm_spc, ista_net_plus_spc

    # Load Set11, create measurement, run all 3 methods
    # Compute PSNR/SSIM vs reference
    # Return results dict

def run_cacti_benchmark(self) -> Dict:
    """Benchmark CACTI with 4 methods."""
    from pwm_core.recon import gap_tv_cacti, elp_unfolding_cacti

    # Load SCI Benchmark, create measurement, run all 4 methods
    # Compute PSNR/SSIM vs reference
    # Return results dict
```

### Step 4: Testing

Create unit tests in `/packages/pwm_core/tests/test_spc_recon.py`:

```python
def test_admm_spc():
    """Test ADMM on synthetic SPC problem."""
    np.random.seed(42)
    x_true = np.random.rand(256).astype(np.float32)
    A = np.random.randn(100, 256) / 10
    y = A @ x_true + 0.01 * np.random.randn(100)

    x_recon = admm_spc(y, A, iterations=50)

    psnr = compute_psnr(x_recon, x_true)
    assert psnr > 15.0, f"PSNR too low: {psnr}"

def test_gap_tv_cacti():
    """Test GAP-TV on synthetic CACTI problem."""
    np.random.seed(42)
    x_true = np.random.rand(256, 256, 8).astype(np.float32)
    mask = np.random.choice([0, 1], (256, 256, 8), p=[0.5, 0.5]).astype(np.float32)

    y = (mask * x_true).sum(axis=2)  # Compress temporally
    y += 0.01 * np.random.randn(256, 256)

    x_recon = gap_tv_cacti(y, mask, iterations=30)

    psnr = compute_psnr(x_recon, x_true)
    assert psnr > 15.0, f"PSNR too low: {psnr}"
```

## Expected Benchmarks

### SPC (Set11, 64×64, 15% sampling = 614 measurements)

| Method | Sampling | PSNR | SSIM | Params | Speed |
|--------|----------|------|------|--------|-------|
| ADMM | 15% | 28.5 | 0.85 | 0 | 2s |
| ISTA-Net+ | 15% | 32.0 | 0.88 | 0.5M | 0.3s |
| HATNet | 15% | 33.0 | 0.90 | 1.2M | 0.2s |

### CACTI (SCI Benchmark, 256×256×8, 8:1 compression)

| Method | PSNR | SSIM | Params | Speed |
|--------|------|------|--------|-------|
| GAP-TV | 26.6 | 0.84 | 0 | 50s |
| PnP-FFDNet | 29.4 | 0.89 | 0.6M | 15s |
| ELP-Unfolding | 33.9 | 0.96 | 1.5M | 20s |
| EfficientSCI | 36.3 | 0.98 | 3.8M | 5s |

## Key Design Principles

1. **Modular Architecture**: Each method is standalone and testable
2. **Flexible Operators**: Support both matrix and callable forward models
3. **Error Handling**: Graceful fallbacks for missing dependencies
4. **Benchmarkability**: Every method has expected PSNR/SSIM targets
5. **Documentation**: Clear docstrings with algorithm references
6. **Logging**: Progress tracking and diagnostics

## References

- ISTA-Net: Zhang & Ghanem (2018) "ISTA-Net: Interpretable Optimization-Inspired Deep Network"
- ELP-Unfolding: Cheng et al. (ECCV 2022) "Deep Unfolding Network for Image Super-Resolution"
- EfficientSCI: Wang et al. (CVPR 2023) "Densely Connected Network with Space-time Factorization"
- GAP-TV: Derived from Nesterov's gradient ascent + TV proximal operators
- ADMM: Boyd et al. (2010) "Distributed Optimization and Statistical Learning via ADMM"

## Next Steps

1. Implement one classical method (e.g., ADMM for SPC)
2. Test on synthetic data with ground truth
3. Integrate into benchmark runner
4. Add one deep learning method (e.g., ISTA-Net+)
5. Compare against published baselines
6. Optimize hyperparameters for Set11 / SCI Benchmark
7. Create comprehensive test suite
