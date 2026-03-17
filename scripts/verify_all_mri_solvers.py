#!/usr/bin/env python3
"""Verify ALL MRI solvers from algorithm_base/mri/solvers.py via run_solver()."""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
# Ensure PWM5 packages
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import glob
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "algorithm_base" / "mri"))


def psnr(x_true, x_hat):
    # Use peak of x_true for PSNR
    peak = np.max(x_true)
    if peak < 1e-10:
        peak = 1.0
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(peak ** 2 / mse)


def ssim_simple(x_true, x_hat):
    """Simple SSIM approximation."""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(x_true, x_hat, data_range=x_true.max() - x_true.min())
    except ImportError:
        return 0.0


def load_scene(idx=0):
    h5_files = sorted(glob.glob(str(ROOT / "datasets/benchmark/mri/standard/*.h5")))
    if not h5_files:
        print("ERROR: No MRI scenes found")
        return None
    f = h5_files[idx]
    with h5py.File(f, 'r') as hf:
        y = hf['y_ideal'][:].astype(np.float32)
        if 'sampling_mask' in hf:
            mask = hf['sampling_mask'][:].astype(np.float32)
        else:
            # Infer mask from k-space
            if y.ndim == 3 and y.shape[-1] == 2:
                kc = y[..., 0] + 1j * y[..., 1]
            else:
                kc = y
            mask = (np.abs(kc) > 1e-10).astype(np.float32)
        return {
            'name': Path(f).stem,
            'x_true': hf['x_true'][:].astype(np.float32),
            'y': y,
            'mask': mask,
        }


def load_all_scenes():
    scenes = []
    h5_files = sorted(glob.glob(str(ROOT / "datasets/benchmark/mri/standard/*.h5")))
    for f in h5_files:
        with h5py.File(f, 'r') as hf:
            y = hf['y_ideal'][:].astype(np.float32)
            if 'sampling_mask' in hf:
                mask = hf['sampling_mask'][:].astype(np.float32)
            else:
                if y.ndim == 3 and y.shape[-1] == 2:
                    kc = y[..., 0] + 1j * y[..., 1]
                else:
                    kc = y
                mask = (np.abs(kc) > 1e-10).astype(np.float32)
            scenes.append({
                'name': Path(f).stem,
                'x_true': hf['x_true'][:].astype(np.float32),
                'y': y,
                'mask': mask,
            })
    return scenes


def main():
    from solvers import SOLVERS, run_solver, MRIOperator

    # Test on scene 0 first for quick verification
    scene = load_scene(0)
    if scene is None:
        return

    x_true = scene['x_true']
    y = scene['y']
    mask = scene['mask']

    operator = MRIOperator(mask, x_true.shape[0])

    print(f"Scene: {scene['name']}, x_true: {x_true.shape}, y: {y.shape}")
    print(f"x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]")
    print(f"mask coverage: {mask.sum()/mask.size:.2%}")
    print(f"Testing {len(SOLVERS)} solvers via run_solver()\n")

    results = {}
    errors = {}

    for solver_key, spec in SOLVERS.items():
        print(f"[{solver_key}] {spec['name']}...", end=" ", flush=True)
        cfg = {'device': 'cuda'}
        try:
            t0 = time.time()
            x_hat = run_solver(solver_key, y.copy(), operator, cfg)
            dt = time.time() - t0
            p = psnr(x_true, x_hat)
            s = ssim_simple(x_true, x_hat)
            results[solver_key] = (p, s, dt)
            print(f"{p:.2f} dB, SSIM={s:.4f} ({dt:.1f}s)")
        except Exception as e:
            errors[solver_key] = str(e)
            print(f"FAILED - {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"MRI SOLVER VERIFICATION SUMMARY - {len(results)}/{len(SOLVERS)} passed")
    print(f"{'='*80}")
    print(f"{'Solver':<22} {'Name':<22} {'PSNR':>8} {'SSIM':>8} {'Time':>8} {'GPU':>5}")
    print("-" * 75)

    for sk, spec in SOLVERS.items():
        if sk in results:
            p, s, dt = results[sk]
            gpu = "GPU" if spec.get("gpu") else "CPU"
            print(f"{sk:<22} {spec['name']:<22} {p:>8.2f} {s:>8.4f} {dt:>7.1f}s {gpu:>5}")
        elif sk in errors:
            print(f"{sk:<22} {spec['name']:<22} {'FAILED':>8} {'--':>8} {'--':>8} {'--':>5}")

    if errors:
        print(f"\n--- ERRORS ({len(errors)}) ---")
        for sk, err in errors.items():
            print(f"  {sk}: {err[:150]}")

    # Now test all scenes for passed solvers
    if len(results) == len(SOLVERS):
        print(f"\n\nAll {len(SOLVERS)} solvers passed! Running 10-scene mean PSNR...\n")
        scenes = load_all_scenes()
        print(f"Loaded {len(scenes)} scenes")

        mean_results = {}
        for solver_key in SOLVERS:
            psnrs = []
            ssims = []
            for sc in scenes:
                op = MRIOperator(sc['mask'], sc['x_true'].shape[0])
                cfg = {'device': 'cuda'}
                try:
                    x_hat = run_solver(solver_key, sc['y'].copy(), op, cfg)
                    psnrs.append(psnr(sc['x_true'], x_hat))
                    ssims.append(ssim_simple(sc['x_true'], x_hat))
                except Exception:
                    pass
            if psnrs:
                mean_results[solver_key] = (np.mean(psnrs), np.mean(ssims))
                print(f"  {solver_key}: {np.mean(psnrs):.2f} dB (SSIM {np.mean(ssims):.4f})")

        print(f"\n{'='*80}")
        print("FINAL 10-SCENE MEAN RESULTS")
        print(f"{'='*80}")
        print(f"{'Solver':<22} {'Name':<22} {'Mean PSNR':>10} {'Mean SSIM':>10}")
        print("-" * 70)
        for sk in sorted(mean_results, key=lambda x: -mean_results[x][0]):
            spec = SOLVERS[sk]
            mp, ms = mean_results[sk]
            print(f"{sk:<22} {spec['name']:<22} {mp:>10.2f} {ms:>10.4f}")


if __name__ == "__main__":
    main()
