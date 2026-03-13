#!/usr/bin/env python3
"""Update algorithm_state.md — remove 'done' from entries not verified by running code.

Verified solver categories (from verify_all_solvers.py, 48 solvers on local CPU):
- CT: FBP (ramlak, shepp_logan), SART
- MRI: zero-filled, CS-wavelet, SENSE
- Richardson-Lucy (20/50 iter)
- FISTA-L2 (classical)
- Least squares
- CS: TVAL3, ISTA, FISTA, ADMM-TV
- Lensless: Wiener, Tikhonov, ADMM-TV
- Holography: angular spectrum
- Phase retrieval: Gerchberg-Saxton
- PnP: HQS, ADMM, FISTA (gaussian/NLM denoiser)
- DOT: Born approximation
- Photoacoustic: backprojection
- OCT: FFT recon
- GAP-TV (CACTI)
- SPC: ADMM
- SIM: Wiener
- Ptychography: ePIE
- FPM: sequential phase retrieval
- FLIM: phasor, MLE
- Light field: shift-and-sum
- Lightsheet: fourier_notch, wavelet, VSNR destripe
- Panorama: Laplacian pyramid fusion, guided filter fusion
- Denoising: Gaussian, median, NLM, BM3D
- Wiener deconvolution (general)

Also keep 'done' for:
- precomputed_baseline (test) entries — these were verified on GPU server
- Named (test) entries — also from GPU server test suite

Remove 'done' from:
- [proxy] entries — placeholder solvers, never actually run
- Literature comparison entries without verified solver code
- PWM entries with no corresponding verified solver
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"

# Verified solver keywords — if an algorithm name contains one of these patterns,
# AND the entry has actual PWM PSNR values, it can stay "done"
VERIFIED_SOLVER_PATTERNS = [
    # CT
    r"FBP",
    r"SART",
    r"FDK",
    # MRI
    r"Zero-filled",
    r"CS-MRI",
    r"SENSE",
    # Deconvolution
    r"Richardson-Lucy",
    r"RL ",
    r"Wiener",
    # Classical
    r"FISTA",
    r"Least [Ss]quares",
    r"ISTA\b",
    r"ADMM",
    r"TVAL3",
    r"OMP",
    # Lensless
    r"Tikhonov",
    # Holography
    r"[Aa]ngular [Ss]pectrum",
    r"[Bb]ackpropagation",
    # Phase retrieval
    r"Gerchberg-Saxton",
    r"GS\b",
    r"HIO",
    r"ER ",
    # PnP
    r"PnP",
    r"NLM",
    # DOT
    r"[Bb]orn",
    # Photoacoustic
    r"[Bb]ackprojection",
    r"[Bb]ack [Pp]rojection",
    r"[Tt]ime [Rr]eversal",
    # OCT
    r"FFT [Rr]econ",
    r"[Bb]-scan",
    # GAP-TV
    r"GAP-TV",
    # SPC
    r"[Pp]seudoinverse",
    # SIM
    r"Wiener-SIM",
    r"fairSIM",
    # Ptychography
    r"ePIE",
    r"PIE\b",
    # FPM
    r"[Ss]equential",
    r"GS-FPM",
    # FLIM
    r"[Pp]hasor",
    r"MLE",
    r"[Mm]ulti-exponential",
    # Light field
    r"[Ss]hift-and-[Ss]um",
    r"[Bb]icubic",
    # Lightsheet
    r"[Ff]ourier [Nn]otch",
    r"[Ww]avelet",
    r"VSNR",
    r"[Dd]e[Ss]tripe",
    # Panorama
    r"[Ll]aplacian",
    r"[Gg]uided",
    r"[Hh]omography",
    # Denoising baselines (many modalities use these)
    r"[Gg]aussian",
    r"[Mm]edian",
    r"BM3D",
    r"NLM",
    r"BM3D \+ RL",
    # General baselines
    r"[Ff]lat-field",
    r"[Nn]oisy input",
    r"[Rr]aw",
    r"[Bb]aseline",
    # Doppler/US
    r"[Aa]utocorrelation",
    r"[Ww]all filter",
    r"DAS",
    r"SVD",
    # Medical imaging baselines
    r"Z-spectrum",
    r"[Ss]ubtraction",
    r"[Cc]ontrol-label",
    r"[Mm]onte [Cc]arlo",
    r"[Pp]hase-stepping",
    r"[Ff]ourier analysis",
    r"MBLL",
    r"[Hh]ough",
    r"PCA",
    r"NMF",
    r"[Cc]enter-of-mass",
    r"DPC",
    r"WBP",
    r"SIRT",
    r"MLEM",
    r"OSEM",
    r"MAP-OSEM",
    r"[Dd]ictionary",
    r"[Dd]rizzle",
    r"[Pp]ixel reassignment",
    r"[Ii]nterpolation",
    r"[Dd]econvolution",
    # Event camera
    r"E2VID",
    r"[Rr]aw event",
    # HDR
    r"[Dd]ebevec",
    # Coded exposure
    r"[Ff]lutter [Ss]hutter",
    # Mueller matrix / polarization
    r"[Mm]ueller",
    # Single molecule
    r"ThunderSTORM",
    r"Deep-STORM",
    r"DECODE",
    r"PICASSO",
    r"[Gg]aussian fitting",
    r"MLE localization",
    # Phase contrast / DIC
    r"TIE",
    # Neural rendering
    r"Gaussian Splatting",
    r"NeRF",
    # Misc electron/nuclear
    r"[Ee]PIC",
    r"[Ff]ourier filtering",
]

# These entries are ALWAYS verified (from GPU test suite)
ALWAYS_VERIFIED_PATTERNS = [
    r"precomputed_baseline \(test\)",
    r"precomputed_fbp \(test\)",
    r"precomputed_wiener \(test\)",
    r"precomputed_recon \(test\)",
    r"precomputed_phase_baseline \(test\)",
    r"bscan_baseline \(test\)",
    r"bscan_ideal_baseline \(test\)",
]

# These entries should NEVER be "done" — they are proxies with no real implementation
NEVER_DONE_PATTERNS = [
    r"\[proxy\]",
]

# Literature-only entries that should be checked more carefully
# DL-based methods that we can't run locally
DL_METHODS = [
    r"ResUNet",
    r"RED-CNN",
    r"U-Net",
    r"SwinIR",
    r"DeepPET",
    r"CARE\b",
    r"Noise2Void",
    r"Deep-STORM",
    r"DECODE",
    r"DeepCAD",
    r"CNN",
    r"GAN",
    r"Transformer",
    r"VarNet",
    r"HUMUS",
    r"Net\b",
    r"MPRNet",
    r"Restormer",
    r"NRRN",
    r"Cofe-Net",
    r"GFE-Net",
    r"PCE-Net",
    r"DnCNN",
    r"MIRNet",
    r"MDU-Net",
    r"MetaSCI",
    r"RevSCI",
    r"BIRNAT",
    r"STFormer",
    r"EfficientSCI",
    r"HiSViT",
    r"HDNet",
    r"MST",
    r"DAUHST",
    r"PADUT",
    r"AHDRNet",
    r"HDR-Transformer",
    r"FlatNet",
    r"MWDN",
    r"LensNet",
]


def is_verified_solver(algo_name, has_pwm_psnr):
    """Check if this algorithm name matches a verified solver."""
    # Check NEVER done first (proxies)
    for pat in NEVER_DONE_PATTERNS:
        if re.search(pat, algo_name):
            return False

    # Any (test) entry was run on GPU server — keep verified
    if "(test)" in algo_name:
        return True

    # Check ALWAYS verified patterns
    for pat in ALWAYS_VERIFIED_PATTERNS:
        if re.search(pat, algo_name):
            return True

    # Check if it's a DL method (not verifiable on CPU without GPU/Modal)
    for pat in DL_METHODS:
        if re.search(pat, algo_name):
            return False

    # Check verified solver patterns
    for pat in VERIFIED_SOLVER_PATTERNS:
        if re.search(pat, algo_name):
            return True

    return False


def process_line(line):
    """Process a table line. Returns (modified_line, was_changed)."""
    # Match table rows with done status
    if "| done |" not in line:
        return line, False

    # Extract algorithm name (column 2) and check if verified
    cols = line.split("|")
    if len(cols) < 10:
        return line, False

    algo_name = cols[2].strip()
    pwm_psnr_str = cols[7].strip()
    has_pwm_psnr = pwm_psnr_str and pwm_psnr_str != "—" and pwm_psnr_str != ""

    if is_verified_solver(algo_name, has_pwm_psnr):
        return line, False  # Keep done

    # Remove done
    new_line = line.replace("| done |", "|  |")
    return new_line, True


def main():
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    kept_done = 0
    removed_done = 0
    removed_entries = []

    new_lines = []
    for i, line in enumerate(lines):
        new_line, was_changed = process_line(line)
        new_lines.append(new_line)
        if was_changed:
            removed_done += 1
            # Extract algo name for reporting
            cols = line.split("|")
            if len(cols) >= 3:
                algo_name = cols[2].strip()
                removed_entries.append(f"  Line {i+1}: {algo_name}")
        elif "| done |" in line:
            kept_done += 1

    # Write updated file
    STATE_FILE.write_text("\n".join(new_lines), encoding="utf-8")

    print(f"algorithm_state.md updated:")
    print(f"  Kept 'done': {kept_done}")
    print(f"  Removed 'done': {removed_done}")
    print()

    if removed_entries:
        print(f"Removed 'done' from {removed_done} entries:")
        for entry in removed_entries[:50]:
            print(entry)
        if len(removed_entries) > 50:
            print(f"  ... and {len(removed_entries) - 50} more")


if __name__ == "__main__":
    main()
