#!/usr/bin/env python3
"""Update algorithm_state.md MRI section with new BrainWeb verification PSNR/SSIM."""
import os, json, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load verification results
with open(os.path.join(ROOT, "benchmark_results", "mri_brainweb_verification.json")) as f:
    vdata = json.load(f)

# Map algorithm names to new PSNR/SSIM values
name_to_results = {}
for key, info in vdata["solvers"].items():
    name_to_results[info["name"]] = (info["mean_psnr"], info["mean_ssim"])

# Also map by common variations
alias_map = {
    "FISTA": "FISTA",
    "CS-MRI (Wavelet)": "CS-MRI (Wavelet)",
    "SENSE": "SENSE",
    "ESPIRiT": "ESPIRiT",
    "ADMM": "ADMM",
    "ISTA": "ISTA",
    "PnP-ADMM": "PnP-ADMM",
    "Zero-Filled IFFT": "Zero-filled IFFT",
    "E2E-VarNet": "E2E-VarNet",
    "Split Bregman": "Split Bregman",
    "POCS": "POCS",
    "Low-Rank": "LORAKS (Low-Rank)",
    "Conjugate Gradient": "Conjugate Gradient",
    "Gradient Descent": "Gradient Descent",
    "Nuclear Norm (SVT)": "Nuclear Norm (SVT)",
    "Landweber Iteration": "Landweber Iteration",
    "Dictionary Learning MRI": "Dictionary Learning MRI",
    "U-Net (fastMRI)": "U-Net (fastMRI)",
    "SPIRiT-like": "SPIRiT-like",
    "Tikhonov Regularization": "Tikhonov Regularization",
    "Proximal Gradient Descent": "Proximal Gradient",
    "GRAPPA-like": "GRAPPA-like",
    "CS-MRI (TV)": "CS-MRI (TV)",
    "Homodyne Detection": "Homodyne Detection",
    "Truncated IFFT": "Truncated IFFT",
    "RED (Regularization by Denoising)": "RED",
    "MoDL": "MoDL",
    "MoDL (5 unrolls)": "MoDL (5 unrolls)",
    "BM3D-MRI": "BM3D-MRI",
    "DC-CNN": "DC-CNN",
    "ISTA-Net+": "ISTA-Net+",
    "Deep ADMM-Net": "Deep ADMM-Net",
    "ALOHA (Hankel Low-Rank)": "ALOHA",
}

# Read algorithm_state.md
state_path = os.path.join(ROOT, "datasets", "benchmark", "algorithm_state.md")
with open(state_path, "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')
in_mri_section = False
updated = 0

for i, line in enumerate(lines):
    if '### 81. Magnetic Resonance Imaging (MRI)' in line:
        in_mri_section = True
        continue
    if in_mri_section and line.startswith('### '):
        in_mri_section = False
        continue

    if not in_mri_section or not line.startswith('|'):
        continue

    # Skip header lines
    if 'Rank' in line or '---' in line:
        continue

    # Parse the table row
    cols = [c.strip() for c in line.split('|')]
    if len(cols) < 10:
        continue

    algo_name = cols[2].strip()  # Algorithm column

    # Find matching result
    result = None
    for solver_name, state_name in alias_map.items():
        if state_name in algo_name or algo_name in state_name:
            if solver_name in name_to_results:
                result = name_to_results[solver_name]
                break

    if result is None:
        # Direct match attempt
        for solver_name, (psnr, ssim) in name_to_results.items():
            if solver_name in algo_name or algo_name in solver_name:
                result = (psnr, ssim)
                break

    if result is not None:
        psnr_val, ssim_val = result
        # Update PWM PSNR (col 7) and PWM SSIM (col 8)
        cols[7] = f" {psnr_val:.2f} "
        cols[8] = f" {ssim_val:.4f} "
        lines[i] = '|'.join(cols)
        updated += 1
        print(f"  Updated: {algo_name} -> PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

with open(state_path, "w", encoding="utf-8") as f:
    f.write('\n'.join(lines))

print(f"\nUpdated {updated} rows in algorithm_state.md MRI section")
