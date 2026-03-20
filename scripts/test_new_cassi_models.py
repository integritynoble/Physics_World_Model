#!/usr/bin/env python3
"""Test newly added CASSI models: RDLUF-MixS2, SSR-L."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def psnr(x_true, x_hat):
    mse = np.mean((x_true - x_hat) ** 2)
    if mse < 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def load_scene(idx=0):
    import glob
    h5_files = sorted(glob.glob(str(ROOT / "datasets/benchmark/cassi/standard/*.h5")))
    if not h5_files:
        print("ERROR: No CASSI scenes found")
        return None
    f = h5_files[idx]
    with h5py.File(f, 'r') as hf:
        return {
            'name': Path(f).stem,
            'x_true': hf['x_true'][:].astype(np.float32),
            'y': hf['y_ideal'][:].astype(np.float32),
            'mask': hf['mask'][:].astype(np.float32),
        }


def test_model(model_key, scene):
    from pwm_core.recon.cassi_models import cassi_model_recon
    print(f"\n[{model_key}] Testing on {scene['name']}...")
    try:
        t0 = time.time()
        x_hat = cassi_model_recon(
            y=scene['y'], mask_2d=scene['mask'],
            model_key=model_key,
            x_true=scene['x_true'],
            device='cuda'
        )
        dt = time.time() - t0
        p = psnr(scene['x_true'], x_hat)
        print(f"  {model_key}: PSNR={p:.2f} dB, time={dt:.1f}s")
        return p
    except Exception as e:
        print(f"  {model_key}: FAILED — {e}")
        import traceback; traceback.print_exc()
        return None


def main():
    scene = load_scene(0)
    if scene is None:
        return

    models = ["rdluf_mixs2_9stg", "ssr_l"]

    for mk in models:
        test_model(mk, scene)


if __name__ == "__main__":
    main()
