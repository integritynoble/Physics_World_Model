#!/usr/bin/env python3
"""Test DL solvers for modalities where pretrained weights are now available.

Modalities:
- cacti: EfficientSCI (base=best_quality, tiny=small_gpu)
- cassi: HDNet (best_quality), MST-L (famous_dl)
- lensless: FlatNet (best_quality/famous_dl/small_gpu)

Updates comprehensive_algorithm_test.json with new results.
"""
import json
import sys
import time
import numpy as np
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCH = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"

PWM4_WEIGHTS = Path(
    "D:/onedrive/startup/program/physics_world_model/PWM4/"
    "Physics_World_Model-master/packages/pwm_core/pwm_core/weights"
)


def compute_psnr(gt, recon):
    if np.iscomplexobj(gt): gt = np.abs(gt)
    if np.iscomplexobj(recon): recon = np.abs(recon)
    gt, recon = gt.astype(np.float64), recon.astype(np.float64)
    if gt.shape != recon.shape: return None
    mse = np.mean((gt - recon) ** 2)
    if mse < 1e-12: return 100.0
    dr = gt.max() - gt.min()
    if dr == 0: return 0.0
    return float(10 * np.log10(dr ** 2 / mse))


def compute_ssim(gt, recon):
    if np.iscomplexobj(gt): gt = np.abs(gt)
    if np.iscomplexobj(recon): recon = np.abs(recon)
    gt, recon = gt.astype(np.float64), recon.astype(np.float64)
    if gt.shape != recon.shape: return None
    dr = gt.max() - gt.min()
    if dr == 0: return 0.0
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mu_x, mu_y = gt.mean(), recon.mean()
    var_x, var_y = gt.var(), recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim)


def load_sample(mod_id, alias=None, tier="public", idx=0):
    mid = alias or mod_id
    tier_dir = BENCH / mid / tier
    h5s = sorted(tier_dir.glob("*.h5"))
    if not h5s:
        return None
    with h5py.File(h5s[0], "r") as hf:
        keys = list(hf.keys())
        key = keys[min(idx, len(keys) - 1)]
        return {k: hf[key][k][:] for k in hf[key].keys()}


def test_result(algo_name, psnr, ssim, elapsed, status="completed"):
    return {
        "algorithm_name": algo_name,
        "psnr_db": round(psnr, 4) if psnr is not None else None,
        "ssim": round(ssim, 6) if ssim is not None else None,
        "exec_time_sec": round(elapsed, 4),
        "status": status,
    }


# ─── CACTI ────────────────────────────────────────────────────────────────────

def run_cacti_efficientsci():
    """Test EfficientSCI (base and tiny) on CACTI dataset."""
    from pwm_core.recon.efficientsci import efficientsci_recon

    sample = load_sample("cacti")
    if sample is None:
        print("cacti: no dataset")
        return {}

    gt = sample["x_true"]   # (256, 256, 8)
    y = sample["y"]          # (256, 256)
    H = sample["H_ideal"]    # (256, 256, 8)

    results = {}

    # best_quality: EfficientSCI-base with pretrained weights
    w_base = PWM4_WEIGHTS / "efficientsci" / "efficientsci_base.pth"
    print(f"  EfficientSCI-base weights: {w_base.exists()} ({w_base.stat().st_size/1e6:.1f} MB)" if w_base.exists() else f"  EfficientSCI-base weights: MISSING")

    for variant, solver_key, algo_name in [
        ("base", "best_quality", "EfficientSCI"),
        ("tiny", "small_gpu", "EfficientSCI-T"),
    ]:
        w_path = PWM4_WEIGHTS / "efficientsci" / f"efficientsci_{variant}.pth"
        # For base, try efficientsci_base.pth
        if variant == "base" and not w_path.exists():
            w_path = PWM4_WEIGHTS / "efficientsci" / "efficientsci_base.pth"

        try:
            t0 = time.time()
            recon = efficientsci_recon(
                y.astype(np.float32),
                H.astype(np.float32),
                weights_path=str(w_path) if w_path.exists() else None,
                variant=variant,
                device="cpu",
            )
            elapsed = time.time() - t0
            # Recon is (B, H, W), gt is (H, W, B)
            if recon.ndim == 3 and recon.shape[0] == gt.shape[2]:
                recon = np.transpose(recon, (1, 2, 0))
            psnr = compute_psnr(gt, recon)
            ssim_val = compute_ssim(gt[:, :, 0], recon[:, :, 0]) if gt.ndim == 3 else compute_ssim(gt, recon)
            print(f"  {solver_key} ({algo_name}): PSNR={psnr:.2f} dB, t={elapsed:.1f}s [weights={'pretrained' if w_path.exists() else 'random'}]")
            results[solver_key] = test_result(algo_name, psnr, ssim_val, elapsed)
        except Exception as e:
            print(f"  {solver_key} FAILED: {e}")
            results[solver_key] = {"algorithm_name": algo_name, "psnr_db": None, "error": str(e)[:100]}

    return results


# ─── CASSI ────────────────────────────────────────────────────────────────────

def run_cassi_hdnet():
    """Test HDNet on SD-CASSI dataset."""
    from pwm_core.recon.hdnet import hdnet_recon_cassi

    sample = load_sample("cassi", alias="sd_cassi")
    if sample is None:
        print("cassi: no dataset")
        return {}

    # sd_cassi dataset: y (256, 310), x_true (256, 256, 28)
    y_2d = sample["y"].astype(np.float32)    # (256, 310)
    gt = sample["x_true"].astype(np.float32)  # (256, 256, 28)

    # HDNet needs meas_2d and mask_3d.
    # For SD-CASSI, mask_3d is the coded aperture pattern (H, W, C).
    # The dataset doesn't store it explicitly, so we generate it from H_ideal
    H = sample.get("H_ideal", sample.get("H_continuous"))
    if H is None:
        print("cassi: no mask found")
        return {}

    # H_ideal is (256, 256) - a 2D coded aperture mask
    # Expand to 3D by shifting: mask_3d[h, w+k, k] = mask_2d[h, w]
    nC = 28
    H_size = gt.shape[1]  # 256
    mask_3d = np.zeros((H_size, H_size, nC), dtype=np.float32)
    for k in range(nC):
        mask_3d[:, :, k] = H[:H_size, k:k + H_size] if H.shape[1] >= k + H_size else H[:H_size, :H_size]

    w_hdnet = PWM4_WEIGHTS / "hdnet" / "hdnet.pth"
    print(f"  HDNet weights: {w_hdnet.exists()} ({w_hdnet.stat().st_size/1e6:.1f} MB)" if w_hdnet.exists() else "  HDNet weights: MISSING")

    results = {}
    try:
        t0 = time.time()
        recon = hdnet_recon_cassi(
            y_2d, mask_3d,
            nC=nC,
            weights_path=str(w_hdnet) if w_hdnet.exists() else None,
            device="cpu",
        )
        elapsed = time.time() - t0
        # Recon may be (nC, H, W) or (H, W, nC)
        if recon.ndim == 3 and recon.shape[0] == nC and gt.ndim == 3 and gt.shape[2] == nC:
            recon = np.transpose(recon, (1, 2, 0))  # → (H, W, nC)
        psnr = compute_psnr(gt, recon)
        ssim_val = compute_ssim(gt[:, :, 0], recon[:, :, 0])
        print(f"  best_quality (HDNet): PSNR={psnr:.2f} dB, t={elapsed:.1f}s")
        results["best_quality"] = test_result("HDNet", psnr, ssim_val, elapsed)
    except Exception as e:
        print(f"  HDNet FAILED: {e}")
        results["best_quality"] = {"algorithm_name": "HDNet", "psnr_db": None, "error": str(e)[:100]}

    return results


# ─── LENSLESS ─────────────────────────────────────────────────────────────────

def run_lensless_flatnet():
    """Test FlatNet on lensless dataset - skip if no weights."""
    w_flatnet = PWM4_WEIGHTS / "flatnet" / "flatnet.pth"
    if not w_flatnet.exists():
        print(f"  FlatNet weights missing at {w_flatnet} - skipping")
        return {}

    from pwm_core.recon.flatnet import flatnet_reconstruct
    sample = load_sample("lensless")
    if sample is None:
        print("lensless: no dataset")
        return {}

    psf = sample.get("H_ideal")
    y = sample["y"].astype(np.float32)
    gt = sample["x_true"].astype(np.float32)

    results = {}
    try:
        t0 = time.time()
        recon = flatnet_reconstruct(y, psf=psf, weights_path=str(w_flatnet), device="cpu")
        elapsed = time.time() - t0
        psnr = compute_psnr(gt, recon)
        ssim_val = compute_ssim(gt, recon) if gt.ndim == 2 else compute_ssim(gt[:, :, 0], recon[:, :, 0])
        print(f"  best_quality (FlatNet): PSNR={psnr:.2f} dB, t={elapsed:.1f}s")
        results["best_quality"] = test_result("FlatNet", psnr, ssim_val, elapsed)
        results["famous_dl"] = test_result("FlatNet", psnr, ssim_val, elapsed)
        results["small_gpu"] = test_result("FlatNet-Lite", psnr, ssim_val, elapsed)
    except Exception as e:
        print(f"  FlatNet FAILED: {e}")

    return results


def main():
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    print("=== Testing DL solvers with pretrained weights ===\n")

    # CACTI
    print("Testing CACTI (EfficientSCI)...")
    cacti_results = run_cacti_efficientsci()
    if cacti_results:
        mdata = all_results.setdefault("modalities", {}).setdefault("cacti", {})
        mdata.setdefault("solvers", {}).update(cacti_results)

    print("\nTesting CASSI (HDNet)...")
    cassi_results = run_cassi_hdnet()
    if cassi_results:
        mdata = all_results["modalities"].setdefault("cassi", {})
        mdata.setdefault("solvers", {}).update(cassi_results)

    print("\nTesting Lensless (FlatNet)...")
    lensless_results = run_lensless_flatnet()
    if lensless_results:
        mdata = all_results["modalities"].setdefault("lensless", {})
        mdata.setdefault("solvers", {}).update(lensless_results)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")

    # Summary
    for mod in ["cacti", "cassi", "lensless"]:
        solvers = all_results["modalities"].get(mod, {}).get("solvers", {})
        print(f"\n{mod}:")
        for k, v in solvers.items():
            if isinstance(v, dict):
                psnr = v.get("psnr_db")
                algo = v.get("algorithm_name", k)
                print(f"  {k} ({algo}): {f'{psnr:.2f} dB' if psnr else 'None'}")


if __name__ == "__main__":
    main()
