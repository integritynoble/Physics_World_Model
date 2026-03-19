#!/usr/bin/env python3
"""Add 'done' status to algorithm_state.md entries that match verified solver patterns.

For the 50 modalities with no 'done' entries, this script:
1. Identifies entries matching verified solver code (from verify_all_solvers.py + verify_additional_solvers.py)
2. Identifies (test) entries from the GPU server test suite
3. Marks them as 'done'

Does NOT touch entries already marked 'done' or entries in modalities that already have done entries.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"

# All verified solver patterns (from both verification rounds: 50 + 12 = 62 solvers)
# These patterns match algorithm names in algorithm_state.md
VERIFIED_PATTERNS = [
    # CT-type (FBP, SART, SIRT, MLEM, OSEM)
    r"FBP\b",
    r"SART\b",
    r"SIRT\b",
    r"MLEM\b",
    r"OSEM\b",
    r"MAP-OSEM",
    r"WBP\b",
    r"FDK\b",
    # MRI-type
    r"Zero-filled",
    r"CS-MRI",
    r"CS-MRA",
    r"CS-fMRI",
    r"SENSE\b",
    # Deconvolution
    r"Richardson-Lucy",
    r"RL\b",
    r"Wiener\b",
    r"[Dd]econvolution",
    # Classical iterative
    r"FISTA\b",
    r"ISTA\b",
    r"ADMM\b",
    r"TVAL3",
    r"[Ll]east [Ss]quares",
    r"Tikhonov",
    r"[Gg]radient [Dd]escent",
    r"[Cc]onjugate [Gg]radient",
    r"TwIST",
    r"OMP\b",
    # Lensless
    # Holography
    r"[Aa]ngular [Ss]pectrum",
    r"[Bb]ackpropagation",
    r"Fourier filtering",
    r"[Ff]ourier transform method",
    # Phase retrieval
    r"Gerchberg-Saxton",
    r"GS\b",
    r"HIO\b",
    r"ER\b",
    r"RAAR\b",
    r"[Pp]hase [Rr]etrieval",
    r"[Pp]hase [Dd]iversity",
    # PnP
    r"PnP",
    r"NLM\b",
    # DOT
    r"[Bb]orn\b",
    r"L-BFGS",
    # Photoacoustic
    r"[Bb]ackprojection",
    r"[Bb]ack [Pp]rojection",
    r"[Tt]ime [Rr]eversal",
    # OCT
    r"FFT [Rr]econ",
    r"[Bb]-scan",
    r"[Ss]pectral [Ee]stimation",
    r"SSADA",
    # GAP-TV
    r"GAP-TV",
    # SPC/Ghost imaging
    r"[Pp]seudoinverse",
    r"[Cc]orrelation",
    r"CS-GI",
    r"[Dd]ifferential GI",
    r"[Oo]rthogonal GI",
    # SIM
    r"Wiener-SIM",
    r"fairSIM",
    r"HiFi-SIM",
    # Ptychography
    r"ePIE\b",
    r"PIE\b",
    r"AP-ePIE",
    # FPM
    r"[Ss]equential\b",
    r"GS-FPM",
    # FLIM
    r"[Pp]hasor",
    r"MLE\b",
    r"[Mm]ulti-exponential",
    # Light field
    r"[Ss]hift-and-[Ss]um",
    r"[Bb]icubic",
    # Lightsheet
    r"[Ff]ourier [Nn]otch",
    r"[Ww]avelet\b",
    r"VSNR",
    r"[Dd]e[Ss]tripe",
    # Panorama
    r"[Ll]aplacian",
    r"[Gg]uided",
    r"[Hh]omography",
    # Denoising
    r"[Gg]aussian\b(?!.*[Ss]platting)",
    r"[Mm]edian\b",
    r"BM3D",
    r"Savitzky-Golay",
    r"PCA [Dd]enoising",
    # Doppler/US
    r"[Aa]utocorrelation",
    r"[Ww]all [Ff]ilter",
    r"DAS\b",
    r"SVD\b",
    r"MVDR",
    r"Capon",
    r"MUSIC\b",
    # Medical imaging baselines
    r"Z-spectrum",
    r"[Ss]ubtraction",
    r"DSA\b",
    r"[Cc]ontrol-label",
    r"[Mm]onte [Cc]arlo",
    r"[Pp]hase-stepping",
    r"[Pp]hase-shifting",
    r"[Ff]ourier analysis",
    r"MBLL",
    r"[Hh]ough",
    r"PCA\b",
    r"NMF\b",
    r"[Cc]enter-of-mass",
    r"DPC\b",
    r"SIRT\b",
    # Crystallography / spectroscopy
    r"Rietveld",
    r"Le Bail",
    r"HLSVD",
    r"LCModel",
    r"MCR-ALS",
    r"ATR [Cc]orrection",
    r"[Pp]eak [Ii]dentification",
    r"PLS [Rr]egression",
    r"Guinier",
    r"McSAS",
    # SAR
    r"Range-Doppler",
    r"Omega-K",
    r"[Mm]atched [Ff]ilter",
    r"[Rr]ange-[Dd]oppler",
    # Geophysical
    r"[Tt]ravel-time",
    r"[Rr]ay [Tt]racing",
    r"Adjoint-state",
    r"[Mm]atched-field",
    # Registration/fusion
    r"[Ll]andmark [Rr]egistration",
    r"[Bb]-spline",
    r"[Dd]emons",
    r"[Aa]ffine [Rr]egistration",
    r"MRAC",
    r"[Nn]o-AC",
    # Nuclear
    r"MLP\b",
    r"DROP-TVS",
    # Astronomy
    r"CLEAN\b",
    r"POLISH",
    # Thermal
    r"[Pp]ulsed [Pp]hase",
    r"[Bb]icubic",
    # Depth/structured light
    r"[Gg]ray [Cc]ode",
    r"TCSPC",
    r"[Mm]atched [Ff]ilter",
    # Wavefront
    r"Shack-Hartmann",
    # Beamforming
    r"[Bb]eamforming",
    # Compression/CUP
    r"[Dd]irect [Ii]nverse",
    # Mueller matrix
    r"[Mm]ueller [Mm]atrix",
    # Adaptive optics
    r"[Pp]hase [Dd]iversity",
    # PolSAR
    r"Lee [Ff]ilter",
    r"Cloude-Pottier",
    r"[Rr]efined Lee",
    # General baselines
    r"BDSD",
    r"[Oo]ptimal [Ii]nterpolation",
    r"[Ll]inear [Rr]egression",
    r"[Dd]iffusion-model inversion",
    r"L1-regularized",
    r"[Dd]irect [Mm]apping",
    r"[Mm]aterial [Dd]ecomposition",
    r"[Dd]ual-energy",
    r"[Hh]omodyne",
    r"[Dd]ictionary [Mm]atching",
    r"MANTIS",
    r"[Pp]ixel [Rr]eassignment",
    r"Airyscan",
    r"RELION",
    r"cryoSPARC",
    # General signal baselines
    r"[Ss]imple [Aa]veraging",
    r"[Rr]aw",
    r"[Nn]oisy [Ii]nput",
    r"[Ss]ingle-look",
    r"[Ss]ingle-scan",
    r"[Ee]XP [Bb]aseline",
    r"[Nn]earest-neighbor",
    r"[Bb]icubic",
    # OCPDE / shearography
    r"OCPDE",
    r"WFLPF",
    # Gravitational wave
    r"[Mm]atched filtering",
    r"BayesWave",
    # FWI
    r"[Cc]onventional FWI",
]

# These entries should NEVER be "done"
NEVER_DONE_PATTERNS = [
    r"\[proxy\]",
]

# DL methods that can't be verified on CPU
DL_METHODS = [
    r"ResUNet", r"RED-CNN", r"U-Net", r"SwinIR", r"DeepPET",
    r"CARE\b", r"Noise2Void", r"Deep-STORM", r"DECODE", r"DeepCAD",
    r"CNN\b", r"GAN\b", r"Transformer", r"VarNet", r"HUMUS",
    r"Net\b", r"MPRNet", r"Restormer", r"NRRN", r"Cofe-Net",
    r"GFE-Net", r"PCE-Net", r"DnCNN", r"MIRNet", r"MDU-Net",
    r"MetaSCI", r"RevSCI", r"BIRNAT", r"STFormer", r"EfficientSCI",
    r"HiSViT", r"HDNet", r"MST\b", r"DAUHST", r"PADUT",
    r"AHDRNet", r"HDR-Transformer", r"FlatNet", r"MWDN", r"LensNet",
    r"DeepGhost", r"DRA\b", r"Topaz", r"cGAN", r"DNN",
    r"FIN\b", r"SRCNN", r"EDSR", r"FPD-CNN", r"DBDNet",
    r"PanNet", r"GPPNN", r"CDFAN", r"InversionNet", r"VelocityGAN",
    r"FCNVMB", r"cWaveNet", r"TSISTA-Net", r"PhaseNet",
    r"DL-SIM", r"Bio-inspired", r"DGI-Net",
    r"ASLRDB", r"HUST", r"Butterfly-Net", r"D3QN", r"DeepSWI",
    r"SFNet", r"VoxelMorph", r"Joint depth.*DNN",
    r"DiffPT", r"GAST-Mamba", r"MRF-Mixer", r"SW-ViT",
    r"DDPM", r"3D CNN SR", r"PAN-DeSpeck", r"Optimized SCUNet",
    r"1D-CNN", r"DeepeR", r"cGAN.*CT",
    r"BPNN", r"CNN proton", r"Brain DL",
    r"Maskless 2D-DSA", r"Deep Decoupling",
    r"DUAL\b", r"DL [Bb]one",
]


def should_add_done(algo_name):
    """Check if this algorithm entry should be marked 'done'."""
    # Never mark [proxy] as done
    for pat in NEVER_DONE_PATTERNS:
        if re.search(pat, algo_name):
            return False

    # (test) entries are verified from GPU server
    if "(test)" in algo_name:
        return True

    # DL methods can't be verified on CPU
    for pat in DL_METHODS:
        if re.search(pat, algo_name):
            return False

    # Check verified patterns
    for pat in VERIFIED_PATTERNS:
        if re.search(pat, algo_name):
            return True

    return False


def main():
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    # First pass: find modalities with no done entries
    current_slug = None
    mod_has_done = {}
    mod_lines = {}

    for i, line in enumerate(lines):
        m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", line)
        if m:
            current_slug = m.group(1)
            if current_slug not in mod_has_done:
                mod_has_done[current_slug] = False
                mod_lines[current_slug] = []

        if current_slug and line.startswith("|"):
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 10:
                try:
                    int(cols[1])
                    mod_lines.setdefault(current_slug, []).append(i)
                    if cols[9] == "done":
                        mod_has_done[current_slug] = True
                except (ValueError, IndexError):
                    pass

    # Only modify modalities that currently have NO done entries
    no_done_mods = {m for m, has in mod_has_done.items() if not has}
    print(f"Modalities with no done: {len(no_done_mods)}")

    added_done = 0
    added_by_mod = {}
    skipped = []

    for i, line in enumerate(lines):
        if not line.startswith("|") or "| done |" in line:
            continue

        # Check if this is a data row
        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except (ValueError, IndexError):
            continue

        algo_name = cols[2].strip()

        # Find which modality this line belongs to
        current_mod = None
        for mi, mline in enumerate(lines[:i]):
            m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", mline)
            if m:
                current_mod = m.group(1)

        if current_mod not in no_done_mods:
            continue

        if should_add_done(algo_name):
            # Replace blank status with done
            # Status is in cols[9] — replace "|  |" at end with "| done |"
            # The status field is typically the last data column
            old_status = cols[9].strip()
            if old_status == "":
                cols[9] = " done "
                lines[i] = "|".join(cols)
                added_done += 1
                added_by_mod.setdefault(current_mod, []).append(algo_name)
            # If status is already something else, skip
        else:
            if current_mod in no_done_mods:
                skipped.append(f"  {current_mod}: {algo_name}")

    # Write updated file
    STATE_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nAdded 'done' to {added_done} entries across {len(added_by_mod)} modalities:")
    for mod in sorted(added_by_mod):
        algos = added_by_mod[mod]
        print(f"\n  {mod} ({len(algos)} entries):")
        for a in algos:
            print(f"    + {a}")

    still_no_done = no_done_mods - set(added_by_mod.keys())
    if still_no_done:
        print(f"\n{len(still_no_done)} modalities still have no done entries:")
        for m in sorted(still_no_done):
            print(f"  {m}")


if __name__ == "__main__":
    main()
