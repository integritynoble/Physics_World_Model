#!/usr/bin/env python3
"""Full 10-scene verification for new CASSI models."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time
import glob
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
    print(f"Loaded {len(scenes)} scenes")

    models = ["rdluf_mixs2_9stg", "ssr_l"]
    results = {}

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"[{model_key}] Testing on {len(scenes)} scenes...")
        print(f"{'='*60}")
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
                print(f"    {sc['name']}: {p:.2f} dB ({dt:.1f}s)")
            except Exception as e:
                print(f"    {sc['name']}: FAILED — {e}")
                import traceback; traceback.print_exc()

        if psnrs:
            mean_p = np.mean(psnrs)
            ref_p = MODEL_REGISTRY[model_key].get("ref_psnr", 0)
            gap = mean_p - ref_p
            status = "MATCH" if abs(gap) < 3 else ("CLOSE" if abs(gap) < 5 else "GAP")
            print(f"\n  => Mean PSNR: {mean_p:.2f} dB (ref: {ref_p:.1f}, diff: {gap:+.1f} dB) [{status}]")
            results[model_key] = mean_p

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'PWM PSNR':>10} {'Ref PSNR':>10} {'Diff':>8} {'Status':>8}")
    print("-" * 65)
    for mk in models:
        if mk in results:
            p = results[mk]
            ref = MODEL_REGISTRY[mk].get("ref_psnr", 0)
            gap = p - ref
            status = "OK" if abs(gap) < 3 else ("CLOSE" if abs(gap) < 5 else "GAP")
            print(f"{mk:<25} {p:>10.2f} {ref:>10.1f} {gap:>+8.1f} {status:>8}")


if __name__ == "__main__":
    main()
