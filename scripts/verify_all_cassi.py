#!/usr/bin/env python3
"""Verify ALL CASSI reconstruction algorithms against KAIST 10-scene benchmark."""
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
    """Load all KAIST scenes from standard h5 files."""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', type=int, default=1, help='Number of scenes to test (1-10)')
    parser.add_argument('--models', nargs='+', default=None, help='Specific models to test')
    args = parser.parse_args()

    print("=" * 70)
    print("CASSI Algorithm Verification — ALL models")
    print("=" * 70)

    scenes = load_all_scenes()
    if not scenes:
        print("ERROR: No CASSI scenes found")
        return
    print(f"Loaded {len(scenes)} scenes")
    scenes = scenes[:args.scenes]

    from pwm_core.recon.cassi_models import cassi_model_recon, MODEL_REGISTRY, _CKPT_ROOT

    # Models to test
    all_models = [
        "admm_net", "gap_net", "tsa_net", "lambda_net",
        "dgsmp", "birnat", "bisrnet",
        "mst_plus_plus", "cst_l_plus",
        "dauhst_9stg", "dauhst_5stg", "dauhst_3stg",
    ]
    models = args.models or all_models

    results = {}
    for model_key in models:
        if model_key not in MODEL_REGISTRY:
            print(f"\n[{model_key}] SKIP — not in registry")
            continue
        ckpt = _CKPT_ROOT / MODEL_REGISTRY[model_key]["ckpt"]
        if not ckpt.exists():
            print(f"\n[{model_key}] SKIP — checkpoint not found: {ckpt}")
            continue

        print(f"\n[{model_key}] Testing on {len(scenes)} scene(s)...")
        psnrs = []
        for sc in scenes:
            try:
                t0 = time.time()
                x_hat = cassi_model_recon(
                    y=sc['y'], mask_2d=sc['mask'],
                    model_key=model_key,
                    x_true=sc['x_true'],  # for binary mask regeneration
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
            print(f"  => Mean: {mean_p:.2f} dB (ref: {ref_p:.1f}, diff: {gap:+.1f}) [{status}]")
            results[model_key] = mean_p

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'PWM PSNR':>10} {'Ref PSNR':>10} {'Diff':>8} {'Status':>8}")
    print("-" * 60)
    for model_key in models:
        if model_key not in results:
            continue
        p = results[model_key]
        ref = MODEL_REGISTRY.get(model_key, {}).get("ref_psnr", 0)
        gap = p - ref
        status = "OK" if abs(gap) < 3 else ("CLOSE" if abs(gap) < 5 else "GAP")
        print(f"{model_key:<20} {p:>10.2f} {ref:>10.1f} {gap:>+8.1f} {status:>8}")


if __name__ == "__main__":
    main()
