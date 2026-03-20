#!/usr/bin/env python3
"""Full 20-scene verification of pretrained ReconFormer."""
import sys, os, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import h5py
def _ssim_approx(a, b, win=7):
    """Simple SSIM approximation without skimage."""
    from scipy.ndimage import uniform_filter
    C1, C2 = 0.01**2, 0.03**2
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    s_a = uniform_filter(a*a, win) - mu_a*mu_a
    s_b = uniform_filter(b*b, win) - mu_b*mu_b
    s_ab = uniform_filter(a*b, win) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1)*(2*s_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1)*(s_a + s_b + C2)
    return float(np.mean(num / den))

from pwm_core.recon.mri_solvers import run_reconformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'datasets/benchmark/mri/standard')
h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])

class MockPhysics:
    def __init__(self, mask):
        self.mask = mask

scenes = []
for scene_idx in range(20):
    fname = h5_files[scene_idx]
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    physics = MockPhysics(mask_data)
    result, info = run_reconformer(y, physics, {"device": "cpu"})

    mse = np.mean((x_true - result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    s = _ssim_approx(x_true, result)
    scenes.append({"scene": scene_idx, "psnr": round(psnr, 2), "ssim": round(s, 4),
                    "pretrained": info.get("pretrained", False)})
    gc.collect()

result_json = {
    "solver": "reconformer",
    "pretrained": True,
    "num_iter": 1,
    "extra_dc_steps": 3,
    "n_scenes": 20,
    "avg_psnr": round(np.mean([s["psnr"] for s in scenes]), 2),
    "avg_ssim": round(np.mean([s["ssim"] for s in scenes]), 4),
    "scenes": scenes,
}

out_path = os.path.join(ROOT, "benchmark_results", "reconformer_pretrained_verification.json")
with open(out_path, "w") as f:
    json.dump(result_json, f, indent=2)

print(f"ReconFormer (pretrained): PSNR={result_json['avg_psnr']:.2f} dB, SSIM={result_json['avg_ssim']:.4f}")
print(f"Saved to {out_path}")
