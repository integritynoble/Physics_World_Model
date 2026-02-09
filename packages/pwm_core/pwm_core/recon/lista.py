"""LISTA: Learned ISTA for Sparse Signal Recovery.

Unrolled sparse coding network for generic linear inverse problems.

References:
- Gregor, K. & LeCun, Y. (2010). "Learning Fast Approximations of
  Sparse Coding", ICML 2010.

Benchmark:
- 10-20x faster convergence than classical ISTA
- Params: ~0.5M, VRAM: <0.5GB
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("LISTA requires PyTorch. Install with: pip install torch")


# --------------------------------------------------------------------------- #
# Weight path
# --------------------------------------------------------------------------- #

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent  # pwm_core package root
_DEFAULT_WEIGHTS = _PKG_ROOT / "weights" / "lista" / "lista.pth"


# --------------------------------------------------------------------------- #
# LISTALayer
# --------------------------------------------------------------------------- #

if HAS_TORCH:

    class LISTALayer(nn.Module):
        """Single LISTA iteration.

        Computes:
            z_{k+1} = shrink(S * z_k + We * y, theta_k)

        where *shrink* is the soft-thresholding (proximal) operator,
        *We* is the encoder matrix, *S* is the lateral inhibition matrix,
        and *theta_k* is a learnable positive threshold.

        Args:
            n: Signal / sparse-code dimension.
            m: Measurement dimension.
        """

        def __init__(self, n: int, m: int) -> None:
            super().__init__()
            self.We = nn.Linear(m, n, bias=False)
            self.S = nn.Linear(n, n, bias=False)
            self.theta = nn.Parameter(torch.ones(n) * 0.01)

        def forward(self, z: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
            """One LISTA step.

            Args:
                z: Current sparse code estimate (B, N).
                y: Measurements (B, M).

            Returns:
                Updated sparse code (B, N).
            """
            pre = self.S(z) + self.We(y)
            # Soft-thresholding with learnable positive threshold
            theta = torch.abs(self.theta)
            return torch.sign(pre) * torch.clamp(torch.abs(pre) - theta, min=0.0)

    class LISTA(nn.Module):
        """Learned ISTA network.

        Stacks *num_layers* independent ``LISTALayer`` modules.  Each layer
        has its own learnable ``(We, S, theta)`` parameters so the network
        can learn an iteration-dependent schedule.

        An optional linear decoder maps the sparse code back to the signal
        domain when the dictionary is not square.

        Args:
            n: Signal / sparse-code dimension.
            m: Measurement dimension.
            num_layers: Number of unrolled ISTA iterations (default 16).
            use_decoder: If ``True`` append a learnable linear decoder
                ``(N -> N)`` after the last layer.  Useful when the
                sparsifying dictionary is not known analytically.
        """

        def __init__(
            self,
            n: int,
            m: int,
            num_layers: int = 16,
            use_decoder: bool = False,
        ) -> None:
            super().__init__()
            self.n = n
            self.m = m
            self.num_layers = num_layers

            self.layers = nn.ModuleList(
                [LISTALayer(n, m) for _ in range(num_layers)]
            )

            self.decoder: Optional[nn.Linear] = None
            if use_decoder:
                self.decoder = nn.Linear(n, n, bias=False)

        # ---- initialisation from measurement matrix ---- #

        def init_from_measurement_matrix(self, A: np.ndarray, step: float = 1.0) -> None:
            """Warm-start weights from a measurement matrix *A*.

            Classical ISTA uses:
                We = step * A^T,   S = I - step * A^T A

            Each layer is initialised identically; training then specialises
            them.

            Args:
                A: Measurement matrix of shape ``(M, N)``.
                step: Step-size (should be ``< 2 / ||A||_2^2``).
            """
            At = A.T  # (N, M)
            We0 = torch.from_numpy((step * At).astype(np.float32))
            S0 = torch.from_numpy(
                (np.eye(self.n) - step * (At @ A)).astype(np.float32)
            )

            for layer in self.layers:
                layer.We.weight.data.copy_(We0)
                layer.S.weight.data.copy_(S0)

        # ---- forward ---- #

        def forward(self, y: "torch.Tensor") -> "torch.Tensor":
            """Run full LISTA forward pass.

            Args:
                y: Measurements ``(B, M)``.

            Returns:
                Sparse code ``(B, N)`` (or decoded signal if decoder is used).
            """
            z = torch.zeros(y.shape[0], self.n, device=y.device, dtype=y.dtype)
            for layer in self.layers:
                z = layer(z, y)

            if self.decoder is not None:
                z = self.decoder(z)
            return z


# --------------------------------------------------------------------------- #
# High-level reconstruction helper
# --------------------------------------------------------------------------- #


def lista_reconstruct(
    y: np.ndarray,
    measurement_matrix: np.ndarray,
    n_components: Optional[int] = None,
    weights_path: Optional[str] = None,
    num_layers: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """Reconstruct a signal from compressed measurements using LISTA.

    This is the main entry point for using LISTA outside the portfolio
    system.  It builds (or loads) a ``LISTA`` model, runs inference, and
    returns the recovered signal as a NumPy array.

    Args:
        y: Measurement vector(s), shape ``(M,)`` or ``(B, M)``.
        measurement_matrix: The ``(M, N)`` measurement / sensing matrix.
        n_components: Sparse-code dimension.  Defaults to ``N``
            (columns of *measurement_matrix*).
        weights_path: Path to a ``.pth`` checkpoint.  When ``None`` the
            model is initialised analytically from *measurement_matrix*
            (no learned weights).
        num_layers: Number of unrolled iterations (ignored when loading
            from checkpoint).
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        Reconstructed signal, shape ``(N,)`` or ``(B, N)``.
    """
    _require_torch()

    M, N = measurement_matrix.shape
    if n_components is None:
        n_components = N

    # Handle 1-D input
    squeeze = False
    if y.ndim == 1:
        y = y[np.newaxis, :]
        squeeze = True

    # Build model
    model = LISTA(n=n_components, m=M, num_layers=num_layers)

    wp = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
    if wp.exists():
        state = torch.load(str(wp), map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        # Analytical warm-start (no learned weights available)
        lip = float(np.linalg.norm(measurement_matrix, ord=2) ** 2)
        step = 1.0 / lip if lip > 0 else 1.0
        model.init_from_measurement_matrix(measurement_matrix, step=step)

    model.to(device)
    model.eval()

    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    with torch.no_grad():
        z = model(y_t)

    x_hat = z.cpu().numpy()
    if squeeze:
        x_hat = x_hat.squeeze(0)

    return x_hat.astype(np.float32)


# --------------------------------------------------------------------------- #
# Quick training helper
# --------------------------------------------------------------------------- #


def lista_train_quick(
    measurement_matrix: np.ndarray,
    y_train: np.ndarray,
    x_train: np.ndarray,
    num_layers: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> np.ndarray:
    """Train LISTA on (y, x) pairs and return reconstructions for y_train.

    Initialises from the measurement matrix analytically, then fine-tunes
    on the provided training pairs.

    Args:
        measurement_matrix: Sensing matrix (M, N).
        y_train: Training measurements (B, M) or (M,).
        x_train: Training ground truth signals (B, N) or (N,).
        num_layers: Number of unrolled iterations.
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device.

    Returns:
        Reconstructed signals for y_train, shape matching x_train.
    """
    _require_torch()

    M, N = measurement_matrix.shape
    squeeze = False
    if y_train.ndim == 1:
        y_train = y_train[np.newaxis, :]
        x_train = x_train[np.newaxis, :]
        squeeze = True

    model = LISTA(n=N, m=M, num_layers=num_layers)
    lip = float(np.linalg.norm(measurement_matrix, ord=2) ** 2)
    step = 1.0 / lip if lip > 0 else 1.0
    model.init_from_measurement_matrix(measurement_matrix, step=step)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_t = torch.from_numpy(x_train.astype(np.float32)).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pred = model(y_t)
        loss = loss_fn(x_pred, x_t)
        loss.backward()
        optimizer.step()
        if epoch == epochs // 2:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1

    model.eval()
    with torch.no_grad():
        result = model(y_t).cpu().numpy()

    if squeeze:
        result = result.squeeze(0)
    return result.astype(np.float32)


# --------------------------------------------------------------------------- #
# Portfolio wrapper
# --------------------------------------------------------------------------- #


def run_lista(
    y: np.ndarray,
    physics: Any,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run LISTA reconstruction (portfolio interface).

    Args:
        y: Measurements.
        physics: Physics operator.  Must expose either an ``A`` attribute
            (the measurement matrix) or ``forward`` / ``adjoint`` methods.
        cfg: Configuration dict with optional keys:

            - ``num_layers`` (int): Unrolled iterations (default 16).
            - ``weights`` (str): Path to checkpoint.
            - ``device`` (str): ``'cpu'`` or ``'cuda'`` (default ``'cpu'``).

    Returns:
        Tuple of ``(reconstructed_signal, info_dict)``.
    """
    num_layers = cfg.get("num_layers", 16)
    weights = cfg.get("weights", None)
    device = cfg.get("device", "cpu")

    info: Dict[str, Any] = {
        "solver": "lista",
        "num_layers": num_layers,
    }

    try:
        _require_torch()

        # Obtain the measurement matrix A
        A: Optional[np.ndarray] = None
        if hasattr(physics, "A") and physics.A is not None:
            A = np.asarray(physics.A, dtype=np.float32)
        elif hasattr(physics, "measurement_matrix"):
            A = np.asarray(physics.measurement_matrix, dtype=np.float32)

        if A is None:
            info["error"] = "no measurement matrix available"
            return y.astype(np.float32), info

        x_hat = lista_reconstruct(
            y=y.reshape(-1) if y.ndim > 1 else y,
            measurement_matrix=A,
            weights_path=weights,
            num_layers=num_layers,
            device=device,
        )

        return x_hat, info

    except ImportError as exc:
        info["error"] = str(exc)
        return y.astype(np.float32), info
    except Exception as exc:
        info["error"] = str(exc)
        # Best-effort fallback: pseudo-inverse
        if hasattr(physics, "adjoint"):
            try:
                x_shape = tuple(getattr(physics, "x_shape", y.shape))
                return physics.adjoint(y).reshape(x_shape).astype(np.float32), info
            except Exception:
                pass
        return y.astype(np.float32), info
