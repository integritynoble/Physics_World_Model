#!/usr/bin/env python3
"""Final verification of all CASSI algorithms — 10-scene mean PSNR."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import glob
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def load_all_scenes():
    scenes = []
    h5_files = sorted(glob.glob(str(ROOT / "datasets/benchmark/cassi/standard/*.h5")))
    for f in h5_files:
        with h5py.File(f, 'r') as hf:
            scenes.append({
                'name': Path(f).stem,
                'x_true': hf['x_true'][:].astype(np.float32),
                'y': hf['y_ideal'][:].astype(np.float32),
                'mask': hf['mask'][:].astype(np.float32),
            })
    return scenes


def main():
    from pwm_core.recon.cassi_models import cassi_model_recon, MODEL_REGISTRY

    scenes = load_all_scenes()
    print(f"Loaded {len(scenes)} scenes\n")

    # All model keys in registry
    models = list(MODEL_REGISTRY.keys())
    results = {}
    errors = []

    for model_key in models:
        spec = MODEL_REGISTRY[model_key]
        ckpt_path = ROOT / "packages/pwm_core/pwm_core/weights/cassi_checkpoints" / spec["ckpt"]
        if not ckpt_path.exists():
            errors.append((model_key, "no checkpoint"))
            continue

        print(f"[{model_key}] Testing on {len(scenes)} scenes...")
        psnrs = []
        for sc in scenes:
            try:
                t0 = time.time()
                x_hat = cassi_model_recon(
                    y=sc['y'], mask_2d=sc['mask'],
                    model_key=model_key,
                    x_true=sc['x_true'],
                    device='cuda'
                )
                dt = time.time() - t0
                p = psnr(sc['x_true'], x_hat)
                psnrs.append(p)
            except Exception as e:
                print(f"    {sc['name']}: FAILED — {e}")

        if psnrs:
            mean_p = np.mean(psnrs)
            ref_p = spec.get("ref_psnr", 0)
            gap = mean_p - ref_p
            status = "MATCH" if abs(gap) < 3 else ("CLOSE" if abs(gap) < 5 else "GAP")
            results[model_key] = (mean_p, ref_p, gap, status)
            print(f"    Mean: {mean_p:.2f} dB (ref: {ref_p:.1f}, diff: {gap:+.1f}) [{status}]")

    # Summary
    print(f"\n{'='*75}")
    print("FINAL CASSI ALGORITHM SUMMARY")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'PWM PSNR':>10} {'Ref PSNR':>10} {'Diff':>8} {'Status':>8}")
    print("-" * 65)

    # Sort by PWM PSNR descending
    for mk, (pwm, ref, gap, st) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{mk:<25} {pwm:>10.2f} {ref:>10.1f} {gap:>+8.1f} {st:>8}")

    if errors:
        print(f"\n--- Missing checkpoints ({len(errors)}) ---")
        for mk, reason in errors:
            print(f"  {mk}: {reason}")

    print(f"\nTotal verified: {len(results)}/{len(models)}")


if __name__ == "__main__":
    main()
