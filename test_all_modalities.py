"""Test all 17 modalities and collect PSNR results."""

import subprocess
import json
import sys
from pathlib import Path

# Define all modality prompts
modalities = [
    ("1. Widefield Basic", "widefield microscopy deconvolution"),
    ("2. Widefield Low-Dose", "widefield low dose high background"),
    ("3. Confocal Live-Cell", "confocal live cell drift"),
    ("4. Confocal 3D", "confocal 3d stack attenuation"),
    ("5. SIM", "SIM structured illumination microscopy"),
    ("6. CASSI", "CASSI spectral hyperspectral coded"),
    ("7. SPC", "spc single pixel camera"),
    ("8. CACTI", "CACTI video sci snapshot"),
    ("9. Lensless", "lensless diffusercam"),
    ("10. Lightsheet", "lightsheet tissue stripes"),
    ("11. CT", "CT tomography low dose"),
    ("12. MRI", "MRI cartesian accelerated"),
    ("13. Ptychography", "ptychography phase retrieval"),
    ("14. Holography", "holography off-axis phase"),
    ("15. NeRF", "nerf multi-view 3d rendering"),
    ("16. Gaussian Splatting", "gaussian splatting 3dgs"),
    ("17. Matrix", "matrix linear inverse"),
]

results = []

print("=" * 70)
print("TESTING ALL 17 MODALITIES WITH PnP-HQS")
print("=" * 70)
print()

for name, prompt in modalities:
    print(f"Testing {name}...", end=" ", flush=True)

    try:
        cmd = [
            ".venv/Scripts/pwm.exe",
            "run",
            "--prompt", prompt,
            "--out-dir", "test_pnp_all"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            output = json.loads(result.stdout)
            recon = output.get("recon", [{}])[0]
            metrics = recon.get("metrics", {})
            solver = recon.get("solver_id", "unknown")

            psnr = metrics.get("psnr", None)
            mse = metrics.get("mse", None)

            if psnr is not None and psnr > 0:
                print(f"PSNR: {psnr:.2f} dB [{solver}]")
                results.append((name, psnr, mse, solver, "OK"))
            elif "shape_mismatch" in metrics:
                print(f"3D output (shape mismatch) [{solver}]")
                results.append((name, None, None, solver, "3D"))
            else:
                print(f"PSNR: {psnr} [{solver}]")
                results.append((name, psnr, mse, solver, "CHECK"))
        else:
            print(f"FAILED: {result.stderr[:100]}")
            results.append((name, None, None, "error", "FAIL"))

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        results.append((name, None, None, "timeout", "TIMEOUT"))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append((name, None, None, "error", str(e)[:50]))

print()
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()
print(f"{'Modality':<25} {'PSNR (dB)':>12} {'MSE':>12} {'Solver':>12} {'Status':>8}")
print("-" * 70)

for name, psnr, mse, solver, status in results:
    psnr_str = f"{psnr:.2f}" if psnr is not None else "N/A"
    mse_str = f"{mse:.6f}" if mse is not None else "N/A"
    print(f"{name:<25} {psnr_str:>12} {mse_str:>12} {solver:>12} {status:>8}")

print("-" * 70)

# Calculate statistics
valid_psnrs = [p for _, p, _, _, s in results if p is not None and p > 0 and s == "OK"]
if valid_psnrs:
    avg_psnr = sum(valid_psnrs) / len(valid_psnrs)
    print(f"\nAverage PSNR (2D modalities): {avg_psnr:.2f} dB")
    print(f"Best PSNR: {max(valid_psnrs):.2f} dB")
    print(f"Worst PSNR: {min(valid_psnrs):.2f} dB")

print(f"\nTotal: {len(results)} modalities tested")
print(f"Success: {sum(1 for _, _, _, _, s in results if s in ['OK', '3D'])} modalities")
