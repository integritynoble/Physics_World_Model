"""Diffusion Posterior Sampling for Inverse Problems.

Score-based diffusion prior for generic linear inverse problems.

References:
- Chung, H. et al. (2023). "Diffusion Posterior Sampling for General Noisy
  Inverse Problems", ICLR 2023.
- Song, Y. et al. (2021). "Score-Based Generative Modeling through SDEs", ICLR.

Benchmark:
- SOTA quality for generic inverse problems (inpainting, SR, CS)
- Slow inference (~1000 diffusion steps)
- Params: ~60M (U-Net diffusion model), VRAM: 8GB+
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "Diffusion posterior sampling requires PyTorch. "
            "Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class SinusoidalPosEmb(nn.Module):
        """Sinusoidal positional embedding for diffusion timesteps.

        Maps scalar timestep t -> R^dim via sin/cos frequencies,
        following the Transformer positional-encoding scheme.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, t: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                t: [B] integer or float timesteps.

            Returns:
                [B, dim] embedding vectors.
            """
            device = t.device
            half = self.dim // 2
            emb = torch.log(torch.tensor(10000.0, device=device)) / (half - 1)
            emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
            emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    # -------------------------------------------------------------------
    # ResBlock with time conditioning
    # -------------------------------------------------------------------

    class ResBlock(nn.Module):
        """Residual block with GroupNorm, SiLU, and time-conditioning.

        Architecture:
            GroupNorm -> SiLU -> Conv -> (+ time_proj) -> GroupNorm -> SiLU -> Conv
            with a skip connection (optionally projected if channels change).
        """

        def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
            super().__init__()
            self.norm1 = nn.GroupNorm(groups, in_ch)
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_ch),
            )
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: "torch.Tensor", t_emb: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: [B, C, H, W]
                t_emb: [B, time_dim] time embedding

            Returns:
                [B, out_ch, H, W]
            """
            h = F.silu(self.norm1(x))
            h = self.conv1(h)
            # Add time conditioning: project t_emb to out_ch and broadcast
            h = h + self.time_mlp(t_emb)[:, :, None, None]
            h = F.silu(self.norm2(h))
            h = self.conv2(h)
            return h + self.skip(x)

    # -------------------------------------------------------------------
    # Self-attention block
    # -------------------------------------------------------------------

    class AttentionBlock(nn.Module):
        """Channel-wise self-attention at low spatial resolution.

        Uses GroupNorm + QKV projection + scaled dot-product attention.
        Applied only at bottleneck resolutions to keep cost manageable.
        """

        def __init__(self, channels: int, num_heads: int = 4, groups: int = 8):
            super().__init__()
            self.norm = nn.GroupNorm(groups, channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.proj = nn.Conv2d(channels, channels, 1)
            self.num_heads = num_heads

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: [B, C, H, W]

            Returns:
                [B, C, H, W]
            """
            B, C, H, W = x.shape
            h = self.norm(x)
            qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each [B, heads, d, HW]
            scale = (C // self.num_heads) ** -0.5
            attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
            attn = attn.softmax(dim=-1)
            out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
            out = out.reshape(B, C, H, W)
            return x + self.proj(out)

    # -------------------------------------------------------------------
    # Diffusion U-Net
    # -------------------------------------------------------------------

    class DiffusionUNet(nn.Module):
        """U-Net noise prediction network for diffusion models.

        Encoder/decoder with skip connections, time conditioning throughout,
        and self-attention at the bottleneck resolution.

        Channel schedule: [64, 128, 256, 512]  (~60M params for 3-channel input).

        Args:
            in_channels: image channels (e.g. 1 for grayscale, 3 for RGB).
            base_ch: base channel count (default 64).
            ch_mult: channel multipliers per encoder stage.
            time_dim: dimension of time embedding.
            attn_resolutions: indices of encoder stages that use attention
                (default: last two stages).
            num_heads: attention heads.
        """

        def __init__(
            self,
            in_channels: int = 1,
            base_ch: int = 64,
            ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
            time_dim: int = 256,
            attn_resolutions: Tuple[int, ...] = (2, 3),
            num_heads: int = 4,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.time_dim = time_dim

            # Time embedding MLP
            self.time_embed = nn.Sequential(
                SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim),
            )

            # Initial convolution
            self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

            channels = [base_ch * m for m in ch_mult]

            # ---- Encoder ----
            self.enc_blocks = nn.ModuleList()
            self.enc_downs = nn.ModuleList()
            ch_in = base_ch
            for idx, ch_out in enumerate(channels):
                self.enc_blocks.append(
                    nn.ModuleList([
                        ResBlock(ch_in, ch_out, time_dim),
                        ResBlock(ch_out, ch_out, time_dim),
                        AttentionBlock(ch_out, num_heads) if idx in attn_resolutions else nn.Identity(),
                    ])
                )
                self.enc_downs.append(
                    nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1)
                    if idx < len(channels) - 1
                    else nn.Identity()
                )
                ch_in = ch_out

            # ---- Bottleneck ----
            self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim)
            self.mid_attn = AttentionBlock(channels[-1], num_heads)
            self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim)

            # ---- Decoder ----
            self.dec_blocks = nn.ModuleList()
            self.dec_ups = nn.ModuleList()
            reversed_channels = list(reversed(channels))
            for idx in range(len(reversed_channels)):
                ch_out = reversed_channels[idx]
                ch_next = reversed_channels[idx + 1] if idx + 1 < len(reversed_channels) else base_ch
                enc_idx = len(channels) - 1 - idx  # corresponding encoder index
                # Input channels: upsampled + skip from encoder
                self.dec_blocks.append(
                    nn.ModuleList([
                        ResBlock(ch_out * 2, ch_out, time_dim),
                        ResBlock(ch_out, ch_out, time_dim),
                        AttentionBlock(ch_out, num_heads) if enc_idx in attn_resolutions else nn.Identity(),
                    ])
                )
                self.dec_ups.append(
                    nn.ConvTranspose2d(ch_out, ch_next, 4, stride=2, padding=1)
                    if idx < len(reversed_channels) - 1
                    else nn.Conv2d(ch_out, base_ch, 1)
                )

            # Final output
            self.final = nn.Sequential(
                nn.GroupNorm(8, base_ch),
                nn.SiLU(),
                nn.Conv2d(base_ch, in_channels, 1),
            )

        def forward(
            self, x: "torch.Tensor", t: "torch.Tensor"
        ) -> "torch.Tensor":
            """Predict noise from noisy image and timestep.

            Args:
                x: [B, C, H, W] noisy image.
                t: [B] integer timesteps.

            Returns:
                [B, C, H, W] predicted noise.
            """
            t_emb = self.time_embed(t)
            h = self.init_conv(x)

            # Encoder
            skips = []
            for (res1, res2, attn), down in zip(self.enc_blocks, self.enc_downs):
                h = res1(h, t_emb)
                h = res2(h, t_emb)
                h = attn(h) if not isinstance(attn, nn.Identity) else h
                skips.append(h)
                h = down(h) if not isinstance(down, nn.Identity) else h

            # Bottleneck
            h = self.mid_block1(h, t_emb)
            h = self.mid_attn(h)
            h = self.mid_block2(h, t_emb)

            # Decoder
            for (res1, res2, attn), up in zip(self.dec_blocks, self.dec_ups):
                skip = skips.pop()
                # Match spatial dimensions if needed (from rounding during downsample)
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = res1(h, t_emb)
                h = res2(h, t_emb)
                h = attn(h) if not isinstance(attn, nn.Identity) else h
                h = up(h)

            return self.final(h)


# ---------------------------------------------------------------------------
# DDPM noise scheduler
# ---------------------------------------------------------------------------

class DDPMScheduler:
    """Linear-beta DDPM noise schedule.

    Precomputes alpha, alpha_bar, and related quantities for T timesteps
    using a linear schedule from beta_start to beta_end.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        betas = np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = np.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar)
        self.device = device

    # -- forward process -------------------------------------------------

    def add_noise(
        self, x0: "torch.Tensor", noise: "torch.Tensor", t: int
    ) -> "torch.Tensor":
        """Forward diffusion: q(x_t | x_0) = N(sqrt_alpha_bar * x0, (1-alpha_bar) I).

        Args:
            x0: clean image [B, C, H, W].
            noise: standard Gaussian noise, same shape as x0.
            t: integer timestep.

        Returns:
            Noisy x_t.
        """
        s_ab = float(self.sqrt_alpha_bar[t])
        s_omab = float(self.sqrt_one_minus_alpha_bar[t])
        return s_ab * x0 + s_omab * noise

    # -- reverse process --------------------------------------------------

    def denoise_step(
        self,
        x_t: "torch.Tensor",
        eps_pred: "torch.Tensor",
        t: int,
    ) -> "torch.Tensor":
        """Single DDPM reverse step: p(x_{t-1} | x_t).

        Args:
            x_t: current noisy sample [B, C, H, W].
            eps_pred: predicted noise [B, C, H, W].
            t: current timestep.

        Returns:
            Denoised x_{t-1}.
        """
        alpha = float(self.alphas[t])
        alpha_bar = float(self.alpha_bar[t])
        beta = float(self.betas[t])

        # Predicted mean
        coeff1 = 1.0 / (alpha ** 0.5)
        coeff2 = beta / ((1.0 - alpha_bar) ** 0.5)
        mean = coeff1 * (x_t - coeff2 * eps_pred)

        if t > 0:
            # Posterior variance (simplified)
            alpha_bar_prev = float(self.alpha_bar[t - 1])
            sigma = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta) ** 0.5
            noise = torch.randn_like(x_t)
            return mean + sigma * noise
        return mean

    def predict_x0(
        self, x_t: "torch.Tensor", eps_pred: "torch.Tensor", t: int
    ) -> "torch.Tensor":
        """Estimate x_0 from x_t and predicted noise.

        x_0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
        """
        s_ab = float(self.sqrt_alpha_bar[t])
        s_omab = float(self.sqrt_one_minus_alpha_bar[t])
        return (x_t - s_omab * eps_pred) / max(s_ab, 1e-8)


# ---------------------------------------------------------------------------
# Diffusion Posterior Sampling (DPS)
# ---------------------------------------------------------------------------

def diffusion_posterior_sample(
    y: np.ndarray,
    forward_fn: Callable[[np.ndarray], np.ndarray],
    adjoint_fn: Callable[[np.ndarray], np.ndarray],
    weights_path: Optional[str] = None,
    n_steps: int = 1000,
    guidance_scale: float = 1.0,
    in_channels: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """Diffusion Posterior Sampling for linear inverse problems.

    Runs the full DDPM reverse process with DPS guidance:
        x_{t-1} = denoise_step(x_t, eps_theta)
                  - guidance_scale * grad_xt ||y - A(x0_hat)||^2

    where x0_hat is the Tweedie estimate of x_0 from x_t and eps_theta.

    Args:
        y: measurements, arbitrary shape (flattened internally if needed).
        forward_fn: forward operator A (numpy -> numpy).
        adjoint_fn: adjoint operator A^T (numpy -> numpy).
        weights_path: path to pretrained DiffusionUNet checkpoint.
            If None, searches ``{pkg_root}/weights/diffusion/diffusion_unet.pth``.
        n_steps: number of diffusion timesteps (default 1000).
        guidance_scale: strength of the likelihood gradient (default 1.0).
        in_channels: image channels expected by the U-Net.
        device: torch device string (auto-detected if None).

    Returns:
        Reconstructed image as numpy array (same spatial shape as adjoint(y)).
    """
    _require_torch()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # -- determine spatial shape from adjoint --
    x_init = adjoint_fn(y).astype(np.float32)
    if x_init.ndim == 1:
        side = int(np.sqrt(x_init.size / in_channels))
        spatial_shape = (side, side)
    elif x_init.ndim == 2:
        spatial_shape = x_init.shape
    else:
        spatial_shape = x_init.shape[:2]

    H, W = spatial_shape

    # -- build model and scheduler --
    model = DiffusionUNet(in_channels=in_channels, base_ch=64, ch_mult=(1, 2, 4, 8)).to(device)

    # Load weights
    if weights_path is None:
        weights_path = str(
            Path(__file__).resolve().parent / "weights" / "diffusion" / "diffusion_unet.pth"
        )
    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)

    model.eval()

    scheduler = DDPMScheduler(n_steps=n_steps, device=str(device))

    # -- prepare measurement tensor --
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    # -- start from pure noise --
    x_t = torch.randn(1, in_channels, H, W, device=device)

    # -- reverse diffusion loop --
    for t in reversed(range(n_steps)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)

        # Enable gradient for DPS guidance
        x_t = x_t.detach().requires_grad_(True)

        # Predict noise
        with torch.no_grad():
            eps_pred = model(x_t, t_batch)

        # Tweedie estimate of x_0
        x0_hat = scheduler.predict_x0(x_t, eps_pred.detach(), t)

        # -- DPS guidance: grad_xt ||y - A(x0_hat)||^2 --
        x0_np = x0_hat.squeeze(0).detach().cpu().numpy()
        if x0_np.shape[0] == in_channels:
            # [C, H, W] -> [H, W] or [H, W, C]
            x0_np = x0_np.transpose(1, 2, 0).squeeze() if in_channels == 1 else x0_np.transpose(1, 2, 0)

        residual = forward_fn(x0_np) - y
        grad_data_raw = adjoint_fn(residual)
        # Clip extreme values before float32 cast to avoid overflow
        grad_data_raw = np.clip(grad_data_raw, -1e6, 1e6)
        grad_data = np.nan_to_num(grad_data_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Reshape gradient to match x_t
        grad_tensor = torch.from_numpy(grad_data).to(device).float()
        if grad_tensor.ndim == 2:
            grad_tensor = grad_tensor.unsqueeze(0).unsqueeze(0)
        elif grad_tensor.ndim == 3:
            grad_tensor = grad_tensor.permute(2, 0, 1).unsqueeze(0)
        if grad_tensor.shape != x_t.shape:
            grad_tensor = grad_tensor.reshape(x_t.shape)

        # Guidance step size adapts with noise schedule
        alpha_bar_t = float(scheduler.alpha_bar[t])
        step_scale = guidance_scale / max(1.0 - alpha_bar_t, 1e-6)
        step_scale = min(step_scale, 1.0)  # Clip to prevent explosion

        # Sanitize gradient to prevent NaN propagation
        grad_tensor = torch.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip gradient norm to prevent explosion but preserve direction
        grad_norm = torch.norm(grad_tensor)
        max_grad_norm = 100.0
        if grad_norm > max_grad_norm:
            grad_tensor = grad_tensor * (max_grad_norm / grad_norm)

        # Standard reverse step (no grad needed)
        x_t = x_t.detach()
        x_t = scheduler.denoise_step(x_t, eps_pred.detach(), t)

        # Apply DPS posterior guidance
        x_t = x_t - step_scale * grad_tensor
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
        x_t = torch.clamp(x_t, -3.0, 3.0)

    # -- extract result --
    result = x_t.squeeze(0).detach().cpu().numpy()
    if result.shape[0] == in_channels and in_channels == 1:
        result = result.squeeze(0)  # [H, W]
    elif result.ndim == 3 and result.shape[0] == in_channels:
        result = result.transpose(1, 2, 0)  # [H, W, C]

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Portfolio wrapper
# ---------------------------------------------------------------------------

def run_diffusion_posterior(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Portfolio-compatible wrapper for Diffusion Posterior Sampling.

    Args:
        y: measurements.
        physics: physics operator with ``forward`` and ``adjoint`` methods.
        cfg: configuration dict with optional keys:
            - n_steps (int): diffusion timesteps (default 1000).
            - guidance_scale (float): DPS guidance weight (default 1.0).
            - weights_path (str): path to U-Net checkpoint.
            - device (str): torch device.
            - in_channels (int): image channels (default 1).

    Returns:
        Tuple of (reconstructed image, info dict).
    """
    if not (hasattr(physics, "forward") and hasattr(physics, "adjoint")):
        return y.astype(np.float32), {"error": "operator has no forward/adjoint"}

    n_steps = cfg.get("n_steps", 1000)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    weights_path = cfg.get("weights_path", None)
    device = cfg.get("device", None)
    in_channels = cfg.get("in_channels", 1)

    try:
        x = diffusion_posterior_sample(
            y=y,
            forward_fn=physics.forward,
            adjoint_fn=physics.adjoint,
            weights_path=weights_path,
            n_steps=n_steps,
            guidance_scale=guidance_scale,
            in_channels=in_channels,
            device=device,
        )
        info = {
            "solver": "diffusion_posterior",
            "n_steps": n_steps,
            "guidance_scale": guidance_scale,
            "weights_path": weights_path,
        }
        return x, info

    except Exception as e:
        return y.astype(np.float32), {"error": str(e), "solver": "diffusion_posterior"}
