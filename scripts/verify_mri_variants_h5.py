#!/usr/bin/env python3
"""Verify the 6 MRI variant benchmark HDF5 files."""
import h5py
import json
import numpy as np
import os

root = (
    "D:/onedrive/startup/program/physics_world_model/"
    "PWM5/Physics_World_Model/datasets/benchmark"
)

modalities = [
    "asl_mri", "mrs", "mra", "swi",
    "mr_elastography", "mr_fingerprinting",
]
tiers = {"public": 12, "dev": 20, "hidden": 20}

all_ok = True
rows = []

for mod in modalities:
    for tier, n_exp in tiers.items():
        fpath = os.path.join(root, mod, tier, f"{mod}_challenge_{tier}.h5")
        if not os.path.exists(fpath):
            print(f"MISSING: {fpath}")
            all_ok = False
            continue
        with h5py.File(fpath, "r") as hf:
            n_samples = len([k for k in hf.keys() if k.startswith("sample_")])
            if n_samples != n_exp:
                print(f"FAIL {mod}/{tier}: expected {n_exp} samples, got {n_samples}")
                all_ok = False
                continue

            errors = []
            for idx in [0, n_samples - 1]:
                grp = hf[f"sample_{idx:02d}"]
                xt  = grp["x_true"][:]
                y   = grp["y"][:]
                Hi  = grp["H_ideal"][:]
                bl  = grp["reconstruction_baseline"][:]
                spec      = json.loads(grp.attrs["spec"])
                true_spec = json.loads(grp.attrs["true_spec"])

                if xt.shape != (256, 256):
                    errors.append(f"x_true shape {xt.shape}")
                if y.shape != (256, 256, 2):
                    errors.append(f"y shape {y.shape}")
                if Hi.shape != (256, 256):
                    errors.append(f"H_ideal shape {Hi.shape}")
                if bl.shape != (256, 256):
                    errors.append(f"baseline shape {bl.shape}")
                if xt.dtype != np.float32:
                    errors.append(f"x_true dtype {xt.dtype}")
                if y.dtype != np.float32:
                    errors.append(f"y dtype {y.dtype}")
                if Hi.dtype != np.float32:
                    errors.append(f"H_ideal dtype {Hi.dtype}")
                if bl.dtype != np.float32:
                    errors.append(f"baseline dtype {bl.dtype}")
                if xt.min() < -0.01 or xt.max() > 1.01:
                    errors.append(f"x_true range [{xt.min():.4f},{xt.max():.4f}]")
                if Hi.min() < 0.0 or Hi.max() > 1.01:
                    errors.append(f"H_ideal range [{Hi.min():.4f},{Hi.max():.4f}]")
                if not spec:
                    errors.append("spec is empty")
                if not true_spec:
                    errors.append("true_spec is empty")

            sz_mb = os.path.getsize(fpath) / 1e6
            if errors:
                status = "FAIL"
                all_ok = False
                detail = "; ".join(errors)
            else:
                status = "OK"
                detail = f"spec_keys={list(spec.keys())[:4]}"

            rows.append((status, mod, tier, n_samples, sz_mb, detail))

# Print table
print(f"{'Status':<6} {'Modality':<20} {'Tier':<8} {'N':>3} {'MB':>7}  Detail")
print("-" * 80)
for status, mod, tier, n, mb, detail in rows:
    print(f"{status:<6} {mod:<20} {tier:<8} {n:>3} {mb:>7.2f}  {detail}")

print()
if all_ok:
    print("All 18 HDF5 files verified OK.")
else:
    print("Some checks FAILED - see above.")
