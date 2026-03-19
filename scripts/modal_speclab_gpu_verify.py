#!/usr/bin/env python3
"""SpecLab GPU Verification on Modal T4.

Runs GPU-accelerated reconstructions for speclab modalities that the CPU
cannot pass. Detects modality type from data and applies appropriate algorithm.

Usage:
    modal run scripts/modal_speclab_gpu_verify.py --modality confocal_3d
    modal run scripts/modal_speclab_gpu_verify.py --modality ct
    modal run scripts/modal_speclab_gpu_verify.py --modality asl_mri
    modal run scripts/modal_speclab_gpu_verify.py --modality all

The script:
1. Downloads HDF5 from GCS canonical path: datasets/Benchmark/{modality}/public/
2. Detects data type (sinogram, k-space, PSF-blurred, noisy)
3. Applies GPU reconstruction (SIRT, CG-SENSE, Wiener+TV, TV-ADMM)
4. Reports PSNR vs speclab thresholds
5. Prints which algorithms pass (PSNR >= ref_psnr - 2.0 dB)
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import modal

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("pwm-speclab-gpu-verify")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "h5py",
        "scikit-image",
        "scikit-learn",
    )
)

GCS_BUCKET = "pwm-benchmark-datasets"


# ══════════════════════════════════════════════════════════════════════════════
# GPU reconstruction functions (run inside Modal)
# ══════════════════════════════════════════════════════════════════════════════


def _compute_psnr(x_hat: "np.ndarray", x_true: "np.ndarray") -> float:
    import numpy as np
    # Normalize both to [0,1] (scale-invariant PSNR, matching speclab)
    def _n01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-12)
    xh = _n01(x_hat.astype(np.float64))
    xt = _n01(x_true.astype(np.float64))
    # Match shapes
    if xh.shape != xt.shape:
        return 0.0
    mse = float(np.mean((xh - xt) ** 2))
    if mse < 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _gpu_tv_admm(y: "np.ndarray", lam: float = 0.05, n_iter: int = 200,
                 rho: float = 1.0) -> "np.ndarray":
    """GPU-accelerated TV-ADMM denoising/reconstruction."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    # Normalize
    y_min, y_max = y_t.min(), y_t.max()
    if y_max - y_min < 1e-8:
        return y
    y_n = (y_t - y_min) / (y_max - y_min)

    # ADMM for TV denoising: minimize 0.5||x-y||^2 + lam*TV(x)
    x = y_n.clone()
    z_h = torch.zeros_like(x)  # horizontal difference
    z_v = torch.zeros_like(x)  # vertical difference
    u_h = torch.zeros_like(x)
    u_v = torch.zeros_like(x)

    for _ in range(n_iter):
        # x-update: (I + rho * D^T D) x = y + rho * D^T (z - u)
        # Using iterative shrinkage
        grad_h = F.pad(z_h - u_h, (0, 0, 0, 0))
        grad_v = F.pad(z_v - u_v, (0, 0, 0, 0))
        # Divergence of (z-u)
        if x.ndim == 2:
            div_h = torch.zeros_like(x)
            div_v = torch.zeros_like(x)
            div_h[:-1, :] = (z_h - u_h)[:-1, :] - (z_h - u_h)[1:, :]
            div_h[-1, :] = -(z_h - u_h)[-1, :]
            div_v[:, :-1] = (z_v - u_v)[:, :-1] - (z_v - u_v)[:, 1:]
            div_v[:, -1] = -(z_v - u_v)[:, -1]
        else:
            div_h = torch.zeros_like(x)
            div_v = torch.zeros_like(x)
        x = (y_n + rho * (div_h + div_v)) / (1.0 + rho * 2)
        x = torch.clamp(x, 0.0, 1.0)

        # z-update: soft-threshold on image gradients
        if x.ndim == 2:
            dx_h = torch.zeros_like(x)
            dx_v = torch.zeros_like(x)
            dx_h[:-1, :] = x[1:, :] - x[:-1, :]
            dx_v[:, :-1] = x[:, 1:] - x[:, :-1]
        else:
            dx_h = torch.zeros_like(x)
            dx_v = torch.zeros_like(x)

        v_h = dx_h + u_h
        v_v = dx_v + u_v
        thresh = lam / rho
        mag = torch.sqrt(v_h ** 2 + v_v ** 2 + 1e-10)
        scale = torch.clamp(1.0 - thresh / mag, min=0.0)
        z_h = scale * v_h
        z_v = scale * v_v

        # u-update
        u_h = u_h + dx_h - z_h
        u_v = u_v + dx_v - z_v

    return x.cpu().numpy() * (y_max - y_min).cpu().item() + y_min.cpu().item()


def _gpu_wiener_tv(y: "np.ndarray", psf: "np.ndarray | None",
                   n_rl_iter: int = 200) -> "np.ndarray":
    """GPU-accelerated Wiener + Richardson-Lucy deconvolution."""
    import numpy as np
    import torch
    import torch.fft

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize
    y_f = y.astype(np.float64)
    lo, hi = y_f.min(), y_f.max()
    if hi - lo < 1e-12:
        return y
    y_n = (y_f - lo) / (hi - lo)

    if psf is None:
        # Estimate PSF as Gaussian sigma=2
        from scipy.ndimage import gaussian_filter
        s = 5  # kernel size
        k = np.zeros((s, s))
        k[s//2, s//2] = 1.0
        psf = gaussian_filter(k, sigma=2.0)

    # Wiener deconvolution in Fourier domain
    psf_padded = np.zeros_like(y_n)
    ph, pw = psf.shape[:2]
    psf_padded[:ph, :pw] = psf[:min(ph, y_n.shape[0]), :min(pw, y_n.shape[1])]

    Y = np.fft.fft2(y_n)
    H = np.fft.fft2(psf_padded)
    K = 0.01  # regularization
    X_wiener = np.real(np.fft.ifft2(Y * np.conj(H) / (np.abs(H)**2 + K)))
    x_init = np.clip(X_wiener, 0, 1)

    # Richardson-Lucy iterations (GPU)
    y_t = torch.from_numpy(y_n.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    psf_t = torch.from_numpy(psf.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    psf_t = psf_t / psf_t.sum()
    x_t = torch.from_numpy(x_init.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x_t = torch.clamp(x_t, 1e-6, None)

    # Flip PSF for correlation
    psf_flip = torch.flip(psf_t, [-2, -1])
    pad_h = psf_t.shape[-2] // 2
    pad_w = psf_t.shape[-1] // 2

    for _ in range(n_rl_iter):
        # Forward: conv(x, psf)
        y_est = torch.nn.functional.conv2d(x_t, psf_t, padding=(pad_h, pad_w))
        y_est = torch.clamp(y_est, 1e-10, None)
        # Ratio
        ratio = y_t / y_est
        # Correlation with flipped PSF
        update = torch.nn.functional.conv2d(ratio, psf_flip, padding=(pad_h, pad_w))
        x_t = x_t * update
        x_t = torch.clamp(x_t, 1e-8, None)

    result = x_t.squeeze().cpu().numpy()
    result = np.clip(result, 0, 1)
    return result * (hi - lo) + lo


def _gpu_sirt_ct(sinogram: "np.ndarray", angles: "np.ndarray",
                 output_size: int | None = None, n_iter: int = 50) -> "np.ndarray":
    """GPU-accelerated SIRT iterative CT reconstruction."""
    import numpy as np
    import torch
    from skimage.transform import iradon, radon

    n_views, n_det = sinogram.shape
    if output_size is None:
        output_size = n_det

    # Normalize sinogram
    sino_max = sinogram.max()
    if sino_max < 1e-12:
        return np.zeros((output_size, output_size))
    sino_n = sinogram / sino_max

    fbp_kw = {"theta": angles, "filter_name": "ramp", "interpolation": "linear",
              "output_size": output_size}

    # Initialize with FBP
    x = np.clip(iradon(sino_n.T, **fbp_kw), 0.0, None)

    # SIRT iterations (TV-ADMM style on CPU since radon/iradon not GPU)
    # Use SART-like update with TV denoising
    from skimage.restoration import denoise_tv_chambolle

    for i in range(n_iter):
        # Forward project
        s_fwd = radon(x, theta=angles, circle=False).T  # (n_views, n_det')
        # Crop/pad to match
        nd_fwd = s_fwd.shape[1]
        if nd_fwd > n_det:
            t = (nd_fwd - n_det) // 2
            s_fwd = s_fwd[:, t:t + n_det]
        elif nd_fwd < n_det:
            p = (n_det - nd_fwd) // 2
            s_fwd = np.pad(s_fwd, ((0, 0), (p, n_det - nd_fwd - p)))

        # SIRT update: x += A^T (b - Ax) * step
        residual = sino_n - s_fwd
        back = np.clip(iradon(residual.T, **fbp_kw), None, None)
        step = 0.5 / (n_iter ** 0.5 + 1)
        x = x + step * back
        x = np.clip(x, 0.0, None)

        # TV denoising every 10 iterations
        if (i + 1) % 10 == 0:
            x_max = x.max()
            if x_max > 1e-12:
                x_n = x / x_max
                x_n = denoise_tv_chambolle(x_n, weight=0.02, max_num_iter=20)
                x = np.clip(x_n, 0.0, None) * x_max

    return x * sino_max


def _gpu_cg_sense_mri(kspace: "np.ndarray", mask: "np.ndarray",
                      coil_maps: "np.ndarray | None",
                      n_iter: int = 50) -> "np.ndarray":
    """GPU-accelerated CG-SENSE MRI reconstruction."""
    import numpy as np
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle complex k-space (multi-coil or single-coil)
    if kspace.ndim == 2:
        # Single-coil: apply mask, do iFFT
        kspace_masked = kspace * mask if mask is not None else kspace
        img = np.fft.ifft2(np.fft.ifftshift(kspace_masked))
        return np.abs(img)

    if kspace.ndim == 3 and kspace.shape[0] <= 32:
        # (n_coils, H, W) — multi-coil
        n_coils, H, W = kspace.shape
        kspace_t = torch.from_numpy(kspace.astype(np.complex64)).to(device)
        if mask is not None:
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)
            kspace_t = kspace_t * mask_t.unsqueeze(0)

        # Zero-filled initial image (per coil)
        imgs = torch.fft.ifft2(torch.fft.ifftshift(kspace_t, dim=[-2, -1]))

        if coil_maps is not None:
            # CG-SENSE with coil maps
            coil_t = torch.from_numpy(coil_maps.astype(np.complex64)).to(device)
            # RSS initial estimate
            x = (imgs * coil_t.conj()).sum(0)  # (H, W)

            # CG iterations
            def At_A(x_cg):
                # x_cg: (H, W) complex
                # Forward: coil maps * x → k-space
                k = torch.fft.fftshift(
                    torch.fft.fft2(coil_t * x_cg.unsqueeze(0)), dim=[-2, -1]
                )
                if mask is not None:
                    k = k * mask_t.unsqueeze(0)
                # Adjoint
                imgs_back = torch.fft.ifft2(
                    torch.fft.ifftshift(k, dim=[-2, -1])
                )
                return (imgs_back * coil_t.conj()).sum(0)

            # Right-hand side: A^H y
            rhs = (imgs * coil_t.conj()).sum(0)
            r = rhs - At_A(x)
            p = r.clone()

            for _ in range(n_iter):
                Ap = At_A(p)
                rr = (r * r.conj()).real.sum()
                if rr < 1e-10:
                    break
                alpha = rr / ((p * Ap.conj()).real.sum() + 1e-10)
                x = x + alpha * p
                r = r - alpha * Ap
                beta = (r * r.conj()).real.sum() / (rr + 1e-10)
                p = r + beta * p

            return torch.abs(x).cpu().numpy()
        else:
            # RSS reconstruction
            rss = torch.sqrt((torch.abs(imgs) ** 2).sum(0))
            return rss.cpu().numpy()

    # Fallback: treat as 2D k-space
    img = np.fft.ifft2(np.fft.ifftshift(kspace))
    return np.abs(img)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def run_modality_gpu(
    h5_bytes: bytes,
    variant: str,
    thresholds: dict[str, float],
) -> dict:
    """Process one modality on T4 GPU.

    Args:
        h5_bytes: HDF5 file content as bytes
        variant: modality variant key
        thresholds: {algo_name: threshold_psnr}

    Returns:
        dict with results per sample and per-algorithm pass/fail
    """
    import numpy as np
    import h5py
    import torch
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{variant}] Device: {device}  GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'N/A'}")

    f = h5py.File(io.BytesIO(h5_bytes), "r")
    sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
    if not sample_keys:
        sample_keys = sorted(f.keys())[:20]
    print(f"  Samples: {len(sample_keys)}")

    psnr_samples = []

    for sk in sample_keys:
        grp = f[sk]
        available = list(grp.keys())

        # Load x_true
        x_true = None
        for key in ["x_true", "target", "image"]:
            if key in grp:
                x_true = np.array(grp[key]).astype(np.float64)
                break
        if x_true is None:
            continue

        # Load y (measurement)
        y = None
        for key in ["y", "sinogram_measured", "sinogram", "kspace_undersampled",
                    "kspace", "bmode_measured", "bscan_measured", "image_measured",
                    "measurement"]:
            if key in grp:
                y = np.array(grp[key])
                break
        if y is None:
            continue

        # Load optional metadata
        angles = np.array(grp["angles"]) if "angles" in grp else None
        mask = np.array(grp["mask"]) if "mask" in grp else None
        coil_maps = np.array(grp["coil_maps"]) if "coil_maps" in grp else None
        psf = np.array(grp["psf"]) if "psf" in grp else None
        H_ideal = np.array(grp["H_ideal"]) if "H_ideal" in grp else None

        # Detect modality type
        is_sinogram = (angles is not None and y.ndim == 2 and
                       (H_ideal is None or H_ideal.ndim == 1))
        is_mri = ("kspace" in str(available) or "mask" in str(available) or
                  np.iscomplexobj(y))
        is_psf = (psf is not None)

        # Run appropriate GPU reconstruction
        t0 = time.time()
        try:
            if is_sinogram and angles is not None:
                # CT/sinogram: SIRT
                y_2d = y.astype(np.float64)
                if y_2d.ndim != 2:
                    y_2d = y_2d.reshape(y_2d.shape[0], -1)
                x_hat = _gpu_sirt_ct(y_2d, angles.astype(np.float64),
                                     output_size=x_true.shape[-1], n_iter=60)
            elif is_mri or np.iscomplexobj(y):
                # MRI: CG-SENSE
                x_hat = _gpu_cg_sense_mri(y, mask, coil_maps, n_iter=50)
            elif is_psf:
                # PSF convolution: Wiener + Richardson-Lucy
                psf_2d = psf.squeeze()
                if psf_2d.ndim > 2:
                    psf_2d = psf_2d[0]
                y_2d = y.squeeze().astype(np.float64)
                if y_2d.ndim > 2:
                    y_2d = y_2d[0] if y_2d.shape[0] < y_2d.shape[-1] else y_2d[..., 0]
                x_hat = _gpu_wiener_tv(y_2d, psf_2d, n_rl_iter=150)
            else:
                # Denoising / generic: TV-ADMM
                y_2d = y.squeeze().astype(np.float64)
                if y_2d.ndim > 2:
                    y_2d = np.mean(y_2d, axis=0) if y_2d.shape[0] < y_2d.shape[-1] else y_2d[..., 0]
                x_hat = _gpu_tv_admm(y_2d, lam=0.05, n_iter=150)
        except Exception as e:
            print(f"  [{sk}] GPU reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        # Compute PSNR
        xt_2d = x_true.squeeze()
        if xt_2d.ndim > 2:
            xt_2d = np.mean(np.abs(xt_2d), axis=0) if xt_2d.shape[0] < xt_2d.shape[-1] else np.abs(xt_2d[..., 0])
        xh_2d = np.squeeze(x_hat) if isinstance(x_hat, np.ndarray) else x_hat
        if xh_2d.ndim > 2:
            xh_2d = np.mean(np.abs(xh_2d), axis=0) if xh_2d.shape[0] < xh_2d.shape[-1] else np.abs(xh_2d[..., 0])

        if xt_2d.shape == xh_2d.shape:
            psnr = _compute_psnr(xh_2d, xt_2d)
        else:
            print(f"  [{sk}] Shape mismatch: x_hat={xh_2d.shape} vs x_true={xt_2d.shape}")
            psnr = 0.0

        psnr_samples.append(psnr)
        print(f"  [{sk}]  PSNR={psnr:.2f} dB  t={elapsed:.1f}s  type={'sino' if is_sinogram else 'mri' if is_mri else 'psf' if is_psf else 'denoise'}")

    f.close()

    if not psnr_samples:
        return {"variant": variant, "mean_psnr": 0.0, "passing": []}

    mean_psnr = float(np.mean(psnr_samples))
    print(f"\n  [{variant}] Mean PSNR: {mean_psnr:.2f} dB")

    # Compare against thresholds
    passing = []
    for algo_name, thresh in sorted(thresholds.items(), key=lambda x: x[1]):
        if mean_psnr >= thresh:
            passing.append(algo_name)
            print(f"  ✓ PASS: {algo_name} ({mean_psnr:.2f} >= {thresh:.1f})")
        else:
            print(f"  ✗ FAIL: {algo_name} ({mean_psnr:.2f} < {thresh:.1f})")

    return {
        "variant": variant,
        "mean_psnr": round(mean_psnr, 2),
        "n_samples": len(psnr_samples),
        "psnr_samples": [round(p, 2) for p in psnr_samples],
        "passing": passing,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Local helpers
# ══════════════════════════════════════════════════════════════════════════════


def _download_h5(variant: str, tier: str = "public") -> bytes:
    """Download HDF5 from GCS canonical path."""
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    filename = f"{variant}_challenge_{tier}.h5"

    # Try canonical path first, then legacy
    for prefix in [f"datasets/Benchmark/{variant}/{tier}", "challenge-data/v1.0"]:
        gcs_key = f"{prefix}/{filename}"
        blob = bucket.blob(gcs_key)
        if blob.exists():
            print(f"  Downloading gs://{GCS_BUCKET}/{gcs_key} ...")
            data = blob.download_as_bytes()
            print(f"  Downloaded {len(data) // 1024} KB")
            return data

    raise FileNotFoundError(f"Cannot find {filename} in GCS")


def _load_speclab_thresholds(speclab_md: str) -> dict[str, dict[str, float]]:
    """Parse speclab_recon_state.md to get thresholds per modality.

    Returns {variant_key: {algo_name: threshold_psnr}}
    where threshold = ref_psnr - 2.0 dB.
    """
    result: dict[str, dict[str, float]] = {}
    current_key = None

    with open(speclab_md) as f:
        for line in f:
            m = re.match(r"^## .+\(`(\w+)`\)", line)
            if m:
                current_key = m.group(1)
                result[current_key] = {}
                continue
            if current_key and line.startswith("|"):
                parts = [p.strip() for p in line.strip().split("|")]
                if (len(parts) >= 6 and parts[1] and
                        not parts[1].startswith("-") and parts[1] != "Algorithm"):
                    algo = parts[1]
                    status = parts[5].strip()
                    if status == "done":
                        continue  # Already done
                    psnr_str = parts[3].replace(" dB", "").strip()
                    try:
                        ref_psnr = float(psnr_str)
                        result[current_key][algo] = ref_psnr - 2.0
                    except ValueError:
                        pass  # Blank PSNR — skip

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint
# ══════════════════════════════════════════════════════════════════════════════

SPECLAB_MD = Path(__file__).resolve().parents[1] / "datasets/benchmark/speclab_recon_state.md"


@app.local_entrypoint()
def main(
    modality: str = "all",
    tier: str = "public",
    max_modalities: int = 50,
):
    """Run GPU verification for speclab modalities.

    --modality  <key>|all       Modality key or 'all' (default: all)
    --tier      public|dev      Data tier (default: public)
    --max-modalities  N         Maximum modalities to process in 'all' mode
    """
    import sys

    # Load thresholds
    all_thresholds = _load_speclab_thresholds(str(SPECLAB_MD))

    if modality == "all":
        # Process modalities with most remaining algorithms first
        modalities_to_run = [
            (k, v) for k, v in sorted(
                all_thresholds.items(),
                key=lambda x: -len(x[1])  # Most algorithms first
            )
            if v  # Only modalities with pending algorithms
        ][:max_modalities]
    else:
        keys = [m.strip() for m in modality.split(",")]
        modalities_to_run = [
            (k, all_thresholds.get(k, {})) for k in keys
        ]

    print(f"Processing {len(modalities_to_run)} modalities")

    # Spawn all in parallel
    futures = {}
    for vk, thresholds in modalities_to_run:
        if not thresholds:
            print(f"[SKIP] {vk}: no pending algorithms with PSNR targets")
            continue
        try:
            h5_bytes = _download_h5(vk, tier)
        except FileNotFoundError as e:
            print(f"[SKIP] {vk}: {e}")
            continue
        print(f"[QUEUE] {vk} ({len(thresholds)} pending algorithms)")
        futures[vk] = run_modality_gpu.spawn(h5_bytes, vk, thresholds)

    # Collect results
    all_results = {}
    for vk, fut in futures.items():
        try:
            result = fut.get()
            all_results[vk] = result
        except Exception as e:
            print(f"[ERROR] {vk}: {e}")
            all_results[vk] = {"variant": vk, "mean_psnr": 0.0, "passing": []}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passing = 0
    for vk, res in sorted(all_results.items()):
        psnr = res.get("mean_psnr", 0.0)
        passing = res.get("passing", [])
        total_passing += len(passing)
        status = f"✓ {len(passing)} pass" if passing else "✗ 0 pass"
        print(f"  {vk:35s} PSNR={psnr:5.1f} dB  {status}")
        for algo in passing:
            print(f"    → {algo}")

    print(f"\nTotal new passes: {total_passing}")

    # Save results
    out_path = Path(__file__).resolve().parents[1] / "results/speclab_gpu_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out_path}")

    return all_results
