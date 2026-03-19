#!/usr/bin/env python3
"""Fix CT standard dataset sinograms: use circle=False for proper FBP reconstruction.

LoDoPaB-CT images are not zero outside the inscribed circle, so circle=True
gives poor FBP results (~13 dB). circle=False gives ~44 dB FBP quality and
produces 512 detector bins (close to LoDoPaB's native 513).
"""
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio as psnr

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "ct" / "standard"
N_ANGLES = 1000
N_SAMPLES = 20


def rebuild():
    print("Rebuilding CT standard sinograms with circle=False...")
    angles = np.linspace(0, 180, N_ANGLES, endpoint=False)
    results = []

    for idx in range(N_SAMPLES):
        h5_path = OUTDIR / f"standard_ct_{idx:02d}.h5"
        if not h5_path.exists():
            print(f"  SKIP: {h5_path.name} not found")
            continue

        with h5py.File(str(h5_path), "r") as f:
            x_true = np.array(f["x_true"], dtype=np.float32)

        # Recompute sinogram with circle=False
        sino = radon(x_true, theta=angles, circle=False)  # (n_det, n_angles)
        sino_T = sino.T.astype(np.float32)  # (n_angles, n_det)
        n_det = sino_T.shape[1]

        # Verify FBP round-trip
        recon = iradon(sino, theta=angles, circle=False,
                       output_size=x_true.shape[0], filter_name='ramp')
        recon = np.clip(recon.astype(np.float32), 0, 1)
        p = psnr(x_true, recon, data_range=1.0)

        # Rewrite H5 file with correct sinogram
        with h5py.File(str(h5_path), "r+") as f:
            # Delete old sinogram and H_params
            if "y_ideal" in f:
                del f["y_ideal"]
            f.create_dataset("y_ideal", data=sino_T, compression="gzip")

            # Update H_params
            hp = f["H_params"]
            hp.attrs["n_detectors"] = n_det
            hp.attrs["circle"] = False
            hp.attrs["geometry"] = "parallel_beam"

        results.append((idx, x_true.shape, sino_T.shape, p))
        print(f"  sample {idx:02d}: x={x_true.shape} sino={sino_T.shape} FBP_PSNR={p:.2f} dB")

    # Update metadata.json
    if results:
        y_shape = [N_ANGLES, results[0][2][1]]
        x_shape = list(results[0][1])

        meta_path = OUTDIR / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["x_true_shape"] = x_shape
        meta["y_ideal_shape"] = y_shape
        meta["note"] = (f"Real clinical CT from LoDoPaB-CT test split; "
                        f"x_true={x_shape[0]}x{x_shape[1]} normalized [0,1]; "
                        f"y_ideal=sinogram ({N_ANGLES} angles, {y_shape[1]} detectors, parallel beam, circle=False)")
        meta["date_built"] = datetime.now().strftime("%Y-%m-%d")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        spec_path = OUTDIR / "spec.json"
        with open(spec_path) as f:
            spec = json.load(f)
        spec["n_detectors"] = y_shape[1]
        spec["circle"] = False
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)

    mean_psnr = np.mean([r[3] for r in results])
    print(f"\nDone. Mean FBP PSNR: {mean_psnr:.2f} dB ({len(results)} samples)")
    print(f"Sinogram shape: ({N_ANGLES}, {results[0][2][1]})")


if __name__ == "__main__":
    rebuild()
