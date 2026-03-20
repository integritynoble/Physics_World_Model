#!/usr/bin/env python3
"""Targeted solver verification for problem modalities.

Tests each solver with its correct calling convention.
"""
import sys
import os
import json
import time
import traceback
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "benchmark_results" / "targeted_verification.json"
rng = np.random.RandomState(42)
results = {}


def make_phantom(N=64):
    gt = np.zeros((N, N), dtype=np.float32)
    y, x = np.mgrid[-1:1:N*1j, -1:1:N*1j]
    gt += 1.0 * ((x/0.69)**2 + (y/0.92)**2 < 1)
    gt -= 0.8 * ((x/0.66)**2 + (y/0.88)**2 < 1)
    gt += 0.2 * ((x/0.31)**2 + (y/0.11)**2 < 1)
    return gt


def make_blobs(N=64, n=10):
    gt = np.zeros((N, N), dtype=np.float32)
    for _ in range(n):
        cx, cy = rng.randint(5, N-5, 2)
        r = rng.randint(2, 6)
        yy, xx = np.ogrid[-cx:N-cx, -cy:N-cy]
        mask = xx**2 + yy**2 <= r**2
        gt[mask] = rng.uniform(0.3, 1.0)
    return gt


def test_fn(name, fn, timeout=120):
    """Test a function, return result dict."""
    start = time.time()
    try:
        out = fn()
        elapsed = time.time() - start
        if out is None:
            return {"status": "returned_none", "t": round(elapsed, 3)}
        out = np.real(np.asarray(out)).astype(np.float32)
        return {
            "status": "verified",
            "shape": list(out.shape),
            "min": float(np.nanmin(out)),
            "max": float(np.nanmax(out)),
            "t": round(elapsed, 3),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"status": "error", "error": str(e)[:200], "t": round(elapsed, 3)}


def run_all():
    print("=" * 70)
    print("Targeted Solver Verification")
    print("=" * 70)

    N = 64

    # ── 1. MRI solvers (device_cfg_mismatch) ──────────────────────────────
    print("\n--- MRI solvers ---")
    try:
        from pwm_core.recon.mri_solvers import zero_filled_reconstruction, cs_mri_wavelet, sense_reconstruction
        gt = make_phantom(N)
        kspace_full = np.fft.fftshift(np.fft.fft2(gt))
        mask = np.zeros((N, N), dtype=bool)
        center = N // 2
        mask[:, center-3:center+3] = True
        for j in range(N):
            if not mask[0, j] and rng.random() < 0.25:
                mask[:, j] = True
        kspace_us = kspace_full * mask

        results["mri_zero_filled"] = test_fn("MRI zero-filled",
            lambda: zero_filled_reconstruction(kspace_us, mask, device='cpu'))
        results["mri_cs_wavelet"] = test_fn("MRI CS wavelet",
            lambda: cs_mri_wavelet(kspace_us, mask, device='cpu'))
        for k, v in results.items():
            if k.startswith("mri"):
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')}")
    except Exception as e:
        print(f"  MRI import error: {e}")
        results["mri_import"] = {"status": "import_error", "error": str(e)[:200]}

    # ── 2. SPC solvers (operator_interface) ────────────────────────────────
    print("\n--- SPC solvers ---")
    try:
        from pwm_core.recon.spc_solvers import admm_spc, fista_l1
        gt = make_phantom(N).ravel()
        M = N * N // 4
        A = rng.randn(M, N*N).astype(np.float32) / np.sqrt(M)
        y = A @ gt + rng.randn(M).astype(np.float32) * 0.01

        results["spc_admm"] = test_fn("SPC ADMM",
            lambda: admm_spc(y, A, device='cpu'))
        results["spc_fista_l1"] = test_fn("SPC FISTA-L1",
            lambda: fista_l1(y, A, device='cpu'))
        for k, v in results.items():
            if k.startswith("spc"):
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')}")
    except Exception as e:
        print(f"  SPC import error: {e}")
        results["spc_import"] = {"status": "import_error", "error": str(e)[:200]}

    # ── 3. Matrix solvers (operator_interface) ─────────────────────────────
    print("\n--- Matrix/classical solvers ---")
    try:
        from pwm_core.recon.classical import fista_l2
        gt = rng.randn(32*32).astype(np.float64) * 0.3
        gt[np.abs(gt) < 0.15] = 0
        M = 32*32 // 4
        A = rng.randn(M, 32*32).astype(np.float64) / np.sqrt(M)
        y = A @ gt

        results["matrix_fista_l2"] = test_fn("Matrix FISTA-L2",
            lambda: fista_l2(y, A, lam=1e-3, iters=100))
        for k, v in results.items():
            if k.startswith("matrix"):
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')}")
    except Exception as e:
        print(f"  Matrix import error: {e}")
        results["matrix_import"] = {"status": "import_error", "error": str(e)[:200]}

    # ── 4. CACTI/CASSI solvers (return_tuple_mismatch) ─────────────────────
    print("\n--- CACTI/CASSI GAP-TV solvers ---")
    try:
        from pwm_core.recon.gap_tv import gap_tv_cacti, gap_tv_cassi
        import torch

        # CACTI test
        T = 8
        gt_vid = np.stack([make_blobs(N, 5+t) for t in range(T)]).astype(np.float32)
        masks = (rng.rand(T, N, N) > 0.5).astype(np.float32)
        y_cacti = np.sum(gt_vid * masks, axis=0)

        results["cacti_gap_tv"] = test_fn("CACTI GAP-TV",
            lambda: gap_tv_cacti(y_cacti, masks, iterations=10))

        # CASSI test
        n_lambda = 4
        gt_cube = rng.rand(N, N, n_lambda).astype(np.float32) * 0.5 + 0.3
        coded_mask = (rng.rand(N, N) > 0.5).astype(np.float32)
        shift = 2
        y_cassi = np.zeros((N, N + (n_lambda-1)*shift), dtype=np.float32)
        for l in range(n_lambda):
            y_cassi[:, l*shift:l*shift+N] += gt_cube[:,:,l] * coded_mask

        results["cassi_gap_tv"] = test_fn("CASSI GAP-TV",
            lambda: gap_tv_cassi(y_cassi, coded_mask, n_lambda=n_lambda, shift=shift, iterations=10))

        for k, v in results.items():
            if "cassi" in k or "cacti" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  GAP-TV import error: {e}")
        traceback.print_exc()

    # ── 5. Ptychography (return_tuple_mismatch) ───────────────────────────
    print("\n--- Ptychography ---")
    try:
        from pwm_core.recon.ptychography_solver import run_epie
        gt = make_blobs(N)
        # Minimal ptychography data
        n_pos = 16
        scan_pos = [(i*8, j*8) for i in range(4) for j in range(4)]
        probe_size = 16
        probe = np.ones((probe_size, probe_size), dtype=np.complex64)

        # Create diffraction patterns
        patterns = np.zeros((n_pos, probe_size, probe_size), dtype=np.float32)
        for idx, (py, px) in enumerate(scan_pos):
            if py+probe_size <= N and px+probe_size <= N:
                patch = gt[py:py+probe_size, px:px+probe_size]
                exit_wave = patch * probe
                dp = np.abs(np.fft.fft2(exit_wave))**2
                patterns[idx] = dp.astype(np.float32)

        class PtychoOp:
            def __init__(self):
                self.scan_positions = scan_pos
                self.probe = probe
                self.x_shape = (N, N)
            def forward(self, x): return x
            def adjoint(self, y): return y

        op = PtychoOp()
        cfg = {"iterations": 5, "probe_size": probe_size}
        results["ptychography_epie"] = test_fn("ePIE",
            lambda: run_epie(patterns.astype(np.float32), op, cfg))
        for k, v in results.items():
            if "ptycho" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  Ptychography error: {e}")
        results["ptychography_import"] = {"status": "error", "error": str(e)[:200]}

    # ── 6. SIM solvers (return_tuple_mismatch) ─────────────────────────────
    print("\n--- SIM solvers ---")
    try:
        from pwm_core.recon.sim_solver import run_sim_reconstruction, wiener_sim_2d, hifi_sim_2d
        gt = make_blobs(N)
        # 9 SIM patterns (3 orientations x 3 phases)
        n_patterns = 9
        sim_images = np.stack([
            gt + 0.3 * np.sin(2*np.pi*k/3 * np.arange(N)[None,:]/N)
            for k in range(n_patterns)
        ]).astype(np.float32)

        class SIMOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.n_orientations = 3
                self.n_phases = 3
            def forward(self, x): return x
            def adjoint(self, y): return y

        op = SIMOp()
        cfg = {"n_orientations": 3, "n_phases": 3}
        results["sim_wiener"] = test_fn("SIM Wiener",
            lambda: wiener_sim_2d(sim_images, op, cfg))
        results["sim_hifi"] = test_fn("SIM HiFi",
            lambda: hifi_sim_2d(sim_images, op, cfg))

        for k, v in results.items():
            if "sim" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  SIM error: {e}")
        traceback.print_exc()

    # ── 7. FLIM solver (return_tuple_mismatch) ─────────────────────────────
    print("\n--- FLIM solver ---")
    try:
        from pwm_core.recon.flim_solver import run_flim, mle_fit_recon
        # FLIM: time-resolved fluorescence
        n_time = 32
        gt = make_blobs(N)
        decay_times = gt * 3.0 + 1.0  # lifetime in ns
        time_data = np.zeros((n_time, N, N), dtype=np.float32)
        t = np.linspace(0, 10, n_time)
        for i in range(N):
            for j in range(N):
                time_data[:, i, j] = gt[i, j] * np.exp(-t / max(decay_times[i, j], 0.1))
        time_data += rng.normal(0, 0.01, time_data.shape).astype(np.float32)

        class FLIMOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.time_bins = n_time
                self.t = t
            def forward(self, x): return x
            def adjoint(self, y): return y

        op = FLIMOp()
        cfg = {"time_bins": n_time}
        results["flim_mle"] = test_fn("FLIM MLE",
            lambda: mle_fit_recon(time_data.astype(np.float32), op, cfg))
        results["flim_run"] = test_fn("FLIM run_flim",
            lambda: run_flim(time_data.astype(np.float32), op, cfg))

        for k, v in results.items():
            if "flim" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  FLIM error: {e}")
        traceback.print_exc()

    # ── 8. Panorama solver (return_tuple_mismatch) ─────────────────────────
    print("\n--- Panorama solver ---")
    try:
        from pwm_core.recon.panorama_solver import run_panorama_fusion
        # Multiple overlapping views
        n_views = 4
        gt = make_phantom(N)
        views = np.stack([gt + rng.normal(0, 0.05, (N, N)).astype(np.float32)
                         for _ in range(n_views)])

        class PanoOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.n_views = n_views
            def forward(self, x): return x
            def adjoint(self, y): return y

        cfg = {"n_views": n_views, "blend_mode": "average"}
        results["panorama_fusion"] = test_fn("Panorama fusion",
            lambda: run_panorama_fusion(views.astype(np.float32), PanoOp(), cfg))

        for k, v in results.items():
            if "panorama" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  Panorama error: {e}")
        traceback.print_exc()

    # ── 9. Lightsheet solver ──────────────────────────────────────────────
    print("\n--- Lightsheet solver ---")
    try:
        from pwm_core.recon.lightsheet_solver import run_lightsheet, vsnr_destripe
        gt = make_blobs(N)
        from scipy.signal import fftconvolve
        psf = np.zeros((15, 15), dtype=np.float32)
        psf[7, :] = 1.0 / 15
        y = fftconvolve(gt, psf, mode='same').astype(np.float32)
        y += rng.normal(0, 0.02, y.shape).astype(np.float32)

        class LSop:
            def __init__(self):
                self.x_shape = (N, N)
                self.psf = psf
                self.theta = {"sigma": 2.0}
            def forward(self, x):
                return fftconvolve(x, self.psf, mode='same').astype(np.float32)
            def adjoint(self, y):
                return fftconvolve(y, self.psf[::-1,::-1], mode='same').astype(np.float32)

        cfg = {"iters": 10}
        results["lightsheet_run"] = test_fn("Lightsheet run",
            lambda: run_lightsheet(y, LSop(), cfg))
        results["lightsheet_destripe"] = test_fn("Lightsheet destripe",
            lambda: vsnr_destripe(y, LSop(), cfg))

        for k, v in results.items():
            if "lightsheet" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  Lightsheet error: {e}")
        traceback.print_exc()

    # ── 10. PnP solver ────────────────────────────────────────────────────
    print("\n--- PnP solver ---")
    try:
        from pwm_core.recon.pnp import run_pnp
        gt = make_phantom(N)
        noisy = gt + rng.normal(0, 0.1, gt.shape).astype(np.float32)

        class PnPOp:
            def __init__(self):
                self.x_shape = (N, N)
            def forward(self, x):
                return x.copy()
            def adjoint(self, y):
                return y.copy()

        cfg = {"algorithm": "hqs", "denoiser": "nlm", "iters": 10, "sigma": 0.1}
        results["pnp_hqs_nlm"] = test_fn("PnP-HQS (NLM)",
            lambda: run_pnp(noisy, PnPOp(), cfg))

        for k, v in results.items():
            if "pnp" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  PnP error: {e}")
        traceback.print_exc()

    # ── 11. NeRF solver ───────────────────────────────────────────────────
    print("\n--- NeRF solver ---")
    try:
        from pwm_core.recon.nerf_solver import run_nerf
        gt = make_blobs(N)
        n_views = 4
        views = np.stack([gt + rng.normal(0, 0.05, (N, N)).astype(np.float32)
                         for _ in range(n_views)])

        class NeRFOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.n_views = n_views
            def forward(self, x): return x
            def adjoint(self, y): return y

        cfg = {"n_views": n_views, "iterations": 5}
        results["nerf_run"] = test_fn("NeRF run",
            lambda: run_nerf(views.astype(np.float32), NeRFOp(), cfg))

        for k, v in results.items():
            if "nerf" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  NeRF error: {e}")

    # ── 12. Lensless solver ───────────────────────────────────────────────
    print("\n--- Lensless solver ---")
    try:
        from pwm_core.recon.lensless_solver import run_lensless
        gt = make_blobs(N)
        psf = rng.rand(N, N).astype(np.float32)
        psf /= psf.sum()
        from scipy.signal import fftconvolve
        y = fftconvolve(gt, psf, mode='same').astype(np.float32)

        class LenslessOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.psf = psf
            def forward(self, x):
                return fftconvolve(x, self.psf, mode='same').astype(np.float32)
            def adjoint(self, y):
                return fftconvolve(y, self.psf[::-1,::-1], mode='same').astype(np.float32)

        cfg = {"method": "wiener", "reg": 0.01}
        results["lensless_wiener"] = test_fn("Lensless Wiener",
            lambda: run_lensless(y, LenslessOp(), cfg))

        for k, v in results.items():
            if "lensless" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  Lensless error: {e}")

    # ── 13. CT solvers ────────────────────────────────────────────────────
    print("\n--- CT solvers (FBP, SART, PnP) ---")
    try:
        from pwm_core.recon.ct_solvers import run_fbp, run_sart
        ct_data_dir = ROOT / "datasets" / "benchmark" / "ct" / "public" / "sample_00"
        if ct_data_dir.exists():
            meas = np.load(ct_data_dir / "measurement.npy")
            gt_ct = np.load(ct_data_dir / "groundtruth.npy")
            angles = np.load(ct_data_dir / "angles.npy")

            class CTOp:
                def __init__(self):
                    self.angles = np.deg2rad(angles) if angles.max() > 2*np.pi else angles
                    self.x_shape = gt_ct.shape
                    self.output_size = gt_ct.shape[0]
                def forward(self, x): return x
                def adjoint(self, y): return y

            cfg_fbp = {"filter": "ramlak", "output_size": gt_ct.shape[0]}
            cfg_sart = {"iterations": 10, "output_size": gt_ct.shape[0]}

            results["ct_fbp"] = test_fn("CT FBP",
                lambda: run_fbp(meas, CTOp(), cfg_fbp))
            results["ct_sart"] = test_fn("CT SART",
                lambda: run_sart(meas, CTOp(), cfg_sart))
        else:
            print("  No CT data available, using synthetic")
            gt_ct = make_phantom(N)
            from scipy.ndimage import rotate
            n_ang = 60
            angs = np.linspace(0, np.pi, n_ang, endpoint=False)
            sino = np.zeros((n_ang, N), dtype=np.float32)
            for i, a in enumerate(angs):
                sino[i] = rotate(gt_ct, np.degrees(a), reshape=False).sum(axis=0)

            class CTOp:
                def __init__(self):
                    self.angles = angs
                    self.x_shape = (N, N)
                    self.output_size = N
                def forward(self, x): return x
                def adjoint(self, y): return y

            cfg_fbp = {"filter": "ramlak", "output_size": N}
            results["ct_fbp"] = test_fn("CT FBP (synthetic)",
                lambda: run_fbp(sino, CTOp(), cfg_fbp))

        for k, v in results.items():
            if k.startswith("ct"):
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')}")
    except Exception as e:
        print(f"  CT error: {e}")
        traceback.print_exc()

    # ── 14. EfficientSCI ──────────────────────────────────────────────────
    print("\n--- EfficientSCI ---")
    try:
        from pwm_core.recon.efficientsci import efficientsci_recon
        T = 8
        gt_vid = np.stack([make_blobs(N, 5+t) for t in range(T)]).astype(np.float32)
        masks = (rng.rand(T, N, N) > 0.5).astype(np.float32)
        y_cacti = np.sum(gt_vid * masks, axis=0)

        class CACTIOp:
            def __init__(self):
                self.x_shape = (T, N, N)
                self.masks = masks
                self.T = T
            def forward(self, x): return np.sum(x * self.masks, axis=0)
            def adjoint(self, y): return self.masks * y[None, :, :]

        cfg = {"T": T}
        results["efficientsci"] = test_fn("EfficientSCI",
            lambda: efficientsci_recon(y_cacti, CACTIOp(), cfg))

        for k, v in results.items():
            if "efficient" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  EfficientSCI error: {e}")

    # ── 15. Integral solver ───────────────────────────────────────────────
    print("\n--- Integral (light field) ---")
    try:
        from pwm_core.recon.integral_solver import run_integral
        n_views = 16
        gt_lf = np.stack([make_phantom(N) + rng.normal(0, 0.02, (N, N)).astype(np.float32)
                         for _ in range(n_views)])

        class IntegralOp:
            def __init__(self):
                self.x_shape = (N, N)
                self.n_views = n_views
            def forward(self, x): return x
            def adjoint(self, y): return y

        cfg = {"n_views": n_views}
        results["integral"] = test_fn("Integral imaging",
            lambda: run_integral(gt_lf.astype(np.float32), IntegralOp(), cfg))

        for k, v in results.items():
            if "integral" in k:
                print(f"  {k:35s} {v['status']:10s} {v.get('shape','')} {v.get('error','')[:60]}")
    except Exception as e:
        print(f"  Integral error: {e}")

    # ── 16. Check missing functions ───────────────────────────────────────
    print("\n--- Missing function checks ---")
    try:
        from pwm_core.recon.classical import run_fista_l2
        results["endoscopy_fista_l2"] = {"status": "verified"}
        print("  run_fista_l2: EXISTS")
    except (ImportError, AttributeError) as e:
        results["endoscopy_fista_l2"] = {"status": "missing", "error": str(e)[:200]}
        print(f"  run_fista_l2: MISSING ({e})")

    try:
        from pwm_core.recon.noise2void import noise2void_denoise
        results["noise2void"] = {"status": "verified"}
        print("  noise2void_denoise: EXISTS")
    except (ImportError, AttributeError) as e:
        results["noise2void"] = {"status": "missing", "error": str(e)[:200]}
        print(f"  noise2void_denoise: MISSING ({e})")

    # ── 17. DL model solvers (need weights) ───────────────────────────────
    print("\n--- DL model weight checks ---")
    dl_solvers = [
        ("pwm_core.recon.care_unet", "care_restore_2d"),
        ("pwm_core.recon.redcnn", "redcnn_denoise"),
        ("pwm_core.recon.modl", "modl_recon"),
        ("pwm_core.recon.lista", "lista_reconstruct"),
        ("pwm_core.recon.phasenet", "phasenet_reconstruct"),
        ("pwm_core.recon.ptychonn", "ptychonn_reconstruct"),
        ("pwm_core.recon.flatnet", "flatnet_reconstruct"),
        ("pwm_core.recon.ifcnn", "ifcnn_fuse"),
    ]
    for mod_path, func_name in dl_solvers:
        try:
            mod = __import__(mod_path, fromlist=[func_name])
            fn = getattr(mod, func_name)
            results[f"dl_{func_name}"] = {"status": "importable"}
            print(f"  {func_name:30s} IMPORTABLE")
        except Exception as e:
            results[f"dl_{func_name}"] = {"status": "import_error", "error": str(e)[:200]}
            print(f"  {func_name:30s} ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    verified = sum(1 for r in results.values() if r.get("status") in ("verified", "importable"))
    errors = sum(1 for r in results.values() if r.get("status") in ("error", "missing", "import_error"))
    other = len(results) - verified - errors

    print(f"\n{'='*70}")
    print(f"SUMMARY: {verified} verified, {errors} errors, {other} other")

    os.makedirs(RESULTS_PATH.parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_all()
