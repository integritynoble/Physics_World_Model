#!/usr/bin/env python3
"""
PWM Benchmark Algorithm Verifier

Runs reconstruction algorithms on the public-tier benchmark datasets and records
PSNR/SSIM metrics to benchmarks/learn/{modality}/check.md. Also updates the
PWM PSNR/SSIM columns and status in datasets/benchmark/algorithm_state.md.

Physics context (operator family, forward model formula, canonical reference) is
read from datasets/benchmark/standard_references.md and embedded in every test
result block, making each entry scientifically self-documenting.

Operation Modes
---------------
Mode 1 — Run ALL algorithms for all modalities (CPU only by default):
    python verify_benchmark.py --all

Mode 2 — Run algorithms for a SPECIFIC modality:
    python verify_benchmark.py --modality cassi
    python verify_benchmark.py --modality mri --solver traditional_cpu
    python verify_benchmark.py --modality ct --solver traditional_cpu best_quality

Mode 3 — Run all modalities in a named GROUP (keyword match on modality ID):
    python verify_benchmark.py --group mri
    python verify_benchmark.py --group ct
    python verify_benchmark.py --group microscopy

Filter (combinable with any mode):
    --unverified-only   Skip algorithms already recorded in check.md

Other options:
    --include-gpu       Also run GPU-required solvers (default: CPU-only)
    --max-samples N     Use only the first N samples (default: all)
    --timeout SEC       Per-solver timeout in seconds (default: 120)
    --dry-run           Print what would run without executing anything
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import time
import threading
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np

warnings.filterwarnings("ignore")

# ─── Repository paths ────────────────────────────────────────────────────────
# This file lives at: datasets/benchmark/verify_benchmark.py
BENCH_DIR = Path(__file__).resolve().parent         # datasets/benchmark/
ROOT = BENCH_DIR.parent.parent                       # Physics_World_Model/
ALGO_BASE_DIR = ROOT / "algorithm_base"
LEARN_DIR = ROOT / "benchmarks" / "learn"
ALGO_STATE_PATH = BENCH_DIR / "algorithm_state.md"
REFS_PATH = BENCH_DIR / "standard_references.md"
PWM_CORE_DIR = ROOT / "packages" / "pwm_core"

DEFAULT_TIMEOUT = 120  # seconds per solver call

# Inject package paths once at import time
for _p in (str(ROOT), str(PWM_CORE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── Standard References — physics metadata per modality ─────────────────────

# Manual family overrides for modalities where the keyword classifier applied
# to the "Measurement Matrix H" column of standard_references.md produces the
# wrong family.  Each entry records why the override is needed.
_FAMILY_OVERRIDES: Dict[str, str] = {
    # Radon / tomographic reconstruction (emission CT, muon, proton)
    "pet":                     "radon",   # emission CT via Anger camera + sinogram
    "pet_mr":                  "radon",   # PET subsystem is still Radon-based
    "muon_tomo":               "radon",   # muon transmission — line integrals
    "proton_radiography":      "radon",   # proton beam — line integrals
    # Fourier / complex-domain sensing (keywords absent from matrix column)
    "holography":              "fourier", # Fresnel propagation ≡ Fourier optics
    "electron_holography":     "fourier", # off-axis carrier-fringe hologram
    "electron_diffraction":    "fourier", # y = |F{V(r)}|², kinematical
    "ptychography":            "fourier", # pupil-weighted shifted Fourier
    "phase_retrieval":         "fourier", # oversampled Fourier magnitude
    "xray_crystallography":    "fourier", # structure factors F_hkl
    "xfel_sfx":                "fourier", # SFX diffraction patterns
    "sar":                     "fourier", # synthetic aperture (complex FFT)
    "insar":                   "fourier", # interferometric phase (complex)
    "fpm":                     "fourier", # Fourier ptychographic microscopy
    "mr_fingerprinting":       "fourier", # fingerprint k-space acquisition
    "mrs":                     "fourier", # MR spectroscopy (k-space)
    # PSF convolution (nonlinear or specialised microscopy, no "PSF" keyword)
    "sim":                     "psf_conv", # structured illumination (sinusoidal)
    "two_photon":              "psf_conv", # nonlinear (two-photon) PSF
    "three_photon":            "psf_conv", # nonlinear (three-photon) PSF
    "tem":                     "psf_conv", # CTF-modulated projection
    "stem":                    "psf_conv", # probe-function convolution
    "stm":                     "psf_conv", # tip-shape dilation
    "oct":                     "psf_conv", # coherence-gated depth PSF
    "octa":                    "psf_conv", # OCT angiography PSF
    "phase_contrast":          "psf_conv", # phase-contrast PSF
    "shg":                     "psf_conv", # second-harmonic generation PSF
    "srs":                     "psf_conv", # stimulated Raman scattering PSF
    "cars":                    "psf_conv", # coherent anti-Stokes Raman PSF
    "minflux":                 "psf_conv", # PSF-based nanoscale localisation
    # Coded-mask / compressive sensing (keywords absent or matrix says "Kronecker")
    "spc":                     "coded_mask", # single-pixel camera random patterns
    "spc_kronecker":           "coded_mask", # Kronecker structured patterns
    # Wave-equation / acoustic / seismic (matrix says e.g. "Green's function" ≠ wave)
    "ultrasound":              "wave_eq",
    "photoacoustic":           "wave_eq",
    "seismic_tomo":            "wave_eq",
    "ocean_acoustic_tomo":     "wave_eq",
    "sonar":                   "wave_eq",
    "ultrasonic_phased_array": "wave_eq",
    "gpr":                     "wave_eq",  # override from diffusion → wave_eq
}


def _classify_family(matrix_desc: str) -> str:
    """Classify operator family from keywords in the Measurement Matrix H column."""
    m = matrix_desc.lower()
    if any(k in m for k in ("radon", "line integral", "cone-beam",
                             "tilt-series", "projection p_")):
        return "radon"
    if any(k in m for k in ("fourier", "k-space", "fft",
                             "sparse fourier", "visibility")):
        return "fourier"
    if any(k in m for k in ("psf", "convolution", "point spread")):
        return "psf_conv"
    if any(k in m for k in ("mask", "coded aperture", "diagonal mask",
                             "temporal code", "random pattern")):
        return "coded_mask"
    if any(k in m for k in ("diffusion", "green's function",
                             "jacobian j", "weight matrix")):
        return "diffusion"
    if any(k in m for k in ("helmholtz", "impedance", "wave equation")):
        return "wave_eq"
    return "other"


def parse_standard_references() -> Dict[str, Dict[str, str]]:
    """Parse standard_references.md and return per-modality physics metadata.

    Reads the Reference Table (| # | **modality** | citation | link | y=H(x) | H |)
    and auto-classifies each modality's operator family from keywords in the
    Measurement Matrix H column.  Manual overrides in _FAMILY_OVERRIDES correct
    the ~25 modalities where keyword matching alone is insufficient.

    Returns:
        {modality_id: {
            'reference':          canonical citation string,
            'forward_model':      y = H(x) formula string,
            'measurement_matrix': description of H,
            'family':             one of radon | fourier | psf_conv |
                                  coded_mask | diffusion | wave_eq | other,
        }}
    """
    if not REFS_PATH.exists():
        return {}

    text = REFS_PATH.read_text(encoding="utf-8")
    table: Dict[str, Dict[str, str]] = {}

    # Pattern:  | N | **modality** | citation | [link](url) | forward_model | matrix |
    # The link cell may be "[text](url)" or plain text — we discard it entirely.
    row_re = re.compile(
        r"^\|\s*\d+\s*\|"
        r"\s*\*\*(\w+)\*\*\s*\|"    # group 1: modality ID
        r"\s*([^|]+?)\s*\|"          # group 2: canonical reference
        r"\s*[^|]*\|"                # link cell (discarded)
        r"\s*([^|]+?)\s*\|"          # group 3: forward model  y = H(x)
        r"\s*([^|]+?)\s*\|",         # group 4: measurement matrix H
    )

    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        modality = m.group(1).strip()
        reference = m.group(2).strip()
        forward_model = m.group(3).strip()
        matrix = m.group(4).strip()
        family = _FAMILY_OVERRIDES.get(modality, _classify_family(matrix))
        table[modality] = {
            "reference":          reference,
            "forward_model":      forward_model,
            "measurement_matrix": matrix,
            "family":             family,
        }

    return table


# Module-level cache: parsed once on first call to get_modality_family()
_REFS_TABLE: Dict[str, Dict[str, str]] = {}


def _load_refs_table() -> Dict[str, Dict[str, str]]:
    """Return the cached reference table, parsing the file on first call."""
    global _REFS_TABLE
    if not _REFS_TABLE:
        _REFS_TABLE = parse_standard_references()
    return _REFS_TABLE


def get_modality_family(modality: str) -> str:
    """Return the operator family for a modality.

    Family is read from standard_references.md (auto-classified from the
    Measurement Matrix H description, with manual overrides applied).

    Returns one of: radon | fourier | psf_conv | coded_mask | diffusion |
                    wave_eq | other
    """
    return _load_refs_table().get(modality, {}).get("family", "other")


def get_modality_ref_info(modality: str) -> Dict[str, str]:
    """Return the full physics metadata dict for a modality.

    Keys: reference, forward_model, measurement_matrix, family.
    Returns an empty dict if the modality is not in standard_references.md.
    """
    return _load_refs_table().get(modality, {})


# ─── Modality groups for --group flag ─────────────────────────────────────────

MODALITY_GROUPS: Dict[str, List[str]] = {
    "mri": [
        "mri", "asl_mri", "diffusion_mri", "mr_fingerprinting",
        "mr_elastography", "fmri", "us_mri", "swi", "cest_mri", "mrs", "mra",
    ],
    "ct": [
        "ct", "cbct", "industrial_ct", "spectral_ct", "digital_breast_tomo",
        "xray_ndt",
    ],
    "xray": [
        "ct", "cbct", "industrial_ct", "xray_radiography", "xray_ndt",
        "xray_crystallography", "xfel_sfx", "mammography", "dexa",
        "fluoroscopy", "spectral_ct", "dark_field",
    ],
    "microscopy": [
        "sem", "tem", "stem", "afm", "stm", "cryo_em", "cryo_et",
        "fib_sem", "confocal_3d", "confocal_livecell", "confocal_endomicroscopy",
        "two_photon", "three_photon", "lightsheet", "lattice_lightsheet",
        "widefield", "widefield_lowdose", "sted", "palm_storm",
        "sim", "tirf", "spinning_disk", "ism", "expansion", "minflux", "dna_paint",
    ],
    "ultrasound": [
        "ultrasound", "doppler_ultrasound", "ceus", "ivus",
        "elastography", "photoacoustic",
    ],
    "optical": [
        "holography", "lensless", "ptychography", "phase_retrieval", "odt",
        "adaptive_optics", "lucky_imaging", "coronagraphy", "ghost_imaging",
        "fpm", "coded_exposure", "hdr_imaging", "light_field",
        "integral", "panorama", "streak_camera", "polarization",
    ],
    "spectral": [
        "cassi", "hyperspectral_remote", "raman_imaging", "ftir_imaging",
        "cars", "srs", "shg", "ebsd", "edx_mapping", "eels",
        "xrf_imaging", "xrf_tomo", "sims", "libs", "desi", "maldi_msi",
        "spc", "spc_kronecker",
    ],
    "radio": ["radio_astronomy", "radio_interferometry", "eht_imaging"],
    "pet": ["pet", "pet_ct", "pet_mr", "spect", "spect_ct"],
    "astronomy": [
        "coronagraphy", "eht_imaging", "lucky_imaging", "solar_imaging",
        "radio_astronomy", "radio_interferometry", "gravitational_wave",
    ],
    "ndt": [
        "acoustic_emission", "active_thermography", "eddy_current",
        "shearography", "ultrasonic_phased_array", "xray_ndt",
        "machine_vision", "gpr",
    ],
    "compressive": [
        "cassi", "cacti", "spc", "spc_kronecker", "coded_exposure",
        "streak_camera", "cup", "ghost_imaging",
    ],
}


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-12)


def compute_psnr(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute PSNR after normalizing both arrays to [0, 1]."""
    xh = _norm01(np.real(x_hat).astype(np.float64))
    xt = _norm01(np.real(x_true).astype(np.float64))
    if xh.shape != xt.shape:
        try:
            from scipy.ndimage import zoom
            factors = [xt.shape[i] / xh.shape[i]
                       for i in range(min(len(xt.shape), len(xh.shape)))]
            xh = zoom(xh, factors, order=1)
        except Exception:
            return 0.0
    mse = float(np.mean((xh - xt) ** 2))
    if mse < 1e-15:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute SSIM after normalizing both arrays to [0, 1]."""
    from skimage.metrics import structural_similarity as ssim_fn
    xh = _norm01(np.real(x_hat).astype(np.float64))
    xt = _norm01(np.real(x_true).astype(np.float64))
    if xh.shape != xt.shape:
        try:
            from scipy.ndimage import zoom
            factors = [xt.shape[i] / xh.shape[i]
                       for i in range(min(len(xt.shape), len(xh.shape)))]
            xh = zoom(xh, factors, order=1)
        except Exception:
            return 0.0
    try:
        if xh.ndim == 2:
            return float(ssim_fn(xh, xt, data_range=1.0))
        elif xh.ndim >= 3:
            # Average SSIM over the last (spectral/temporal) dimension
            n = xh.shape[-1]
            scores = [float(ssim_fn(xh[..., i], xt[..., i], data_range=1.0))
                      for i in range(min(n, 28))]
            return float(np.mean(scores))
        return 0.0
    except Exception:
        return 0.0


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_public_samples(modality: str,
                        max_samples: Optional[int] = None) -> Tuple[List[Dict], str]:
    """Load samples from the local public-tier HDF5 file.

    Returns:
        (samples, h5_path) — each sample is a dict of numpy arrays.
        Returns ([], "") when no HDF5 file is found.
    """
    pub_dir = BENCH_DIR / modality / "public"
    if not pub_dir.exists():
        return [], ""
    h5_files = sorted(pub_dir.glob("*.h5"))
    if not h5_files:
        return [], ""

    h5_path = str(h5_files[0])
    samples: List[Dict] = []

    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        if max_samples is not None:
            keys = keys[:max_samples]
        for sk in keys:
            grp = f[sk]
            sample: Dict[str, Any] = {"_key": sk}
            for k in grp.keys():
                try:
                    sample[k] = grp[k][()]
                except Exception:
                    pass
            samples.append(sample)

    return samples, h5_path


def get_measurement_and_ground_truth(
    sample: Dict, modality: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract measurement y and ground truth x_true from a sample dict.

    Measurement key selection is driven by the operator family read from
    standard_references.md, covering all 169 modalities:

      fourier  → prefer 'kspace_undersampled'  (complex, e.g. MRI, holography)
      radon    → prefer 'sinogram_measured'     (float, e.g. CT, PET, CBCT)
      others   → prefer 'y'                     (generic measurement key)

    Falls back to 'y' in all cases when the preferred key is absent.

    Returns:
        (y, x_true).  y keeps its native dtype (complex for fourier family).
    """
    x_true = sample.get("x_true")
    if x_true is None:
        return None, None

    family = get_modality_family(modality)

    if family == "fourier":
        y = sample.get("kspace_undersampled", sample.get("y"))
    elif family == "radon":
        y = sample.get("sinogram_measured", sample.get("y"))
    else:
        y = sample.get("y", sample.get("y_ideal", sample.get("measurement")))

    return y, x_true


def build_solver_cfg(sample: Dict, modality: str, base_cfg: Dict) -> Dict:
    """Build a cfg dict that supplements each solver's own defaults.

    Injects modality-specific metadata from the H5 sample so that each
    solver can auto-create its physics operator without needing an explicit
    operator object from outside.  Routing uses the operator family derived
    from standard_references.md rather than hardcoded modality name sets.
    """
    cfg = dict(base_cfg)
    family = get_modality_family(modality)

    # Always pass H_ideal as 'mask' and 'H_ideal' when present — many solvers
    # read one or both names depending on their design.
    if "H_ideal" in sample:
        h = sample["H_ideal"]
        cfg.setdefault("mask", h)
        cfg.setdefault("H_ideal", h)

    # ── Radon family: output image size + angle array ─────────────────────────
    if family == "radon" and "x_true" in sample:
        cfg.setdefault("output_size", int(sample["x_true"].shape[0]))
        if "angles_nominal" in sample:
            cfg.setdefault("angles", sample["angles_nominal"])

    # ── Fourier family: undersampling mask + coil sensitivity maps ───────────
    if family == "fourier":
        if "mask" in sample:
            cfg.setdefault("mask", sample["mask"])
        if "coil_maps" in sample:
            cfg.setdefault("coil_maps", sample["coil_maps"])

    # ── CASSI: spectral dispersion parameters ─────────────────────────────────
    # Handled explicitly because the n_bands / step calculation depends on the
    # specific shape relationship between y and x_true.
    if modality in ("cassi", "sd_cassi"):
        x_true_arr = sample.get("x_true")
        y_arr = sample.get("y")
        if x_true_arr is not None and y_arr is not None:
            if x_true_arr.ndim >= 3:
                # Full 3D spectral cube
                n_bands = x_true_arr.shape[2]
                step = (max(1, (y_arr.shape[1] - x_true_arr.shape[1])
                            // max(n_bands - 1, 1))
                        if y_arr.ndim == 2 else 2)
            elif y_arr.shape == x_true_arr.shape:
                # Simplified 2D benchmark: single-band masked sensing
                n_bands = 1
                step = 0
            else:
                n_bands = 28
                step = (max(1, (y_arr.shape[1] - x_true_arr.shape[1])
                            // max(n_bands - 1, 1))
                        if y_arr.ndim == 2 else 2)
            cfg.setdefault("n_bands", n_bands)
            cfg.setdefault("step", step)

    # ── CACTI: temporal mask stack ────────────────────────────────────────────
    if modality == "cacti" and "H_ideal" in sample:
        cfg.setdefault("masks", sample["H_ideal"])
        cfg.setdefault("n_frames",
                       sample["H_ideal"].shape[-1]
                       if sample["H_ideal"].ndim >= 3 else 8)

    cfg.setdefault("device", "cpu")
    return cfg


# ─── Solver execution with timeout ───────────────────────────────────────────

def run_solver_with_timeout(
    modality: str, solver_key: str,
    y: np.ndarray, cfg: Dict,
    timeout: int,
) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
    """Run algorithm_base.{modality}.solvers.run_solver with a wall-clock timeout.

    Returns:
        (x_hat, elapsed_seconds, error_string)
        x_hat is None on error or timeout; error_string is None on success.
    """
    result_box: List[Any] = [None, 0.0, None]

    def _target() -> None:
        try:
            mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
            run_fn = getattr(mod, "run_solver")
            t0 = time.time()
            res = run_fn(solver_key, y, None, cfg)   # operator=None → auto-create
            elapsed = time.time() - t0
            recon = res[0] if isinstance(res, tuple) else res
            if recon is not None:
                recon = np.real(recon).astype(np.float32)
                recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
            result_box[0] = recon
            result_box[1] = elapsed
        except Exception as exc:
            result_box[2] = f"{type(exc).__name__}: {exc}"[:300]

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None, float(timeout), "timeout exceeded"
    return result_box[0], result_box[1], result_box[2]


# ─── Collapse multi-dim output to 2D for metrics ─────────────────────────────

def _to_2d_for_metrics(arr: np.ndarray) -> np.ndarray:
    """Reduce a multi-dimensional array to 2D for PSNR/SSIM computation.

    (H, W, C) → mean over C channels.
    4D+ → squeeze leading dims first.
    1D → reshape to nearest square.
    """
    a = np.real(arr).astype(np.float32)
    if a.ndim == 1:
        side = int(np.sqrt(a.size))
        a = a[:side * side].reshape(side, side)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3:
        a = np.mean(a, axis=-1)
    return a


# ─── Algorithm state parser ───────────────────────────────────────────────────

def parse_algorithm_state() -> Dict[str, List[Dict]]:
    """Parse algorithm_state.md and return {modality_id: [algorithm_row, ...]}."""
    if not ALGO_STATE_PATH.exists():
        return {}

    text = ALGO_STATE_PATH.read_text(encoding="utf-8")
    result: Dict[str, List[Dict]] = {}
    current_modality: Optional[str] = None
    in_table = False

    section_re = re.compile(r"^###\s+\d+\.\s+.+\(`([a-z0-9_]+)`\)")
    row_re = re.compile(
        r"^\|\s*(\d+)\s*\|"   # rank
        r"\s*(.+?)\s*\|"      # name
        r"\s*(\S+)\s*\|"      # year
        r"\s*(.+?)\s*\|"      # reference
        r"\s*(\S+)\s*\|"      # ref_psnr
        r"\s*(\S+)\s*\|"      # ref_ssim
        r"\s*(\S+)\s*\|"      # pwm_psnr
        r"\s*(\S+)\s*\|"      # pwm_ssim
        r"\s*(\S+)\s*\|"      # status
        r"\s*(\S+)\s*\|"      # organized
    )

    def _parse_float(s: str) -> Optional[float]:
        s = s.strip()
        if s in ("—", "-", "", "N/A", "–"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    for line in text.splitlines():
        m = section_re.match(line)
        if m:
            current_modality = m.group(1)
            result.setdefault(current_modality, [])
            in_table = False
            continue

        if current_modality:
            if "| Rank | Algorithm |" in line:
                in_table = True
                continue
            if line.startswith("|---"):
                continue
            if in_table:
                row_m = row_re.match(line)
                if row_m:
                    result[current_modality].append({
                        "rank":      int(row_m.group(1)),
                        "name":      row_m.group(2).strip(),
                        "year":      row_m.group(3).strip(),
                        "reference": row_m.group(4).strip(),
                        "ref_psnr":  _parse_float(row_m.group(5)),
                        "ref_ssim":  _parse_float(row_m.group(6)),
                        "pwm_psnr":  _parse_float(row_m.group(7)),
                        "pwm_ssim":  _parse_float(row_m.group(8)),
                        "status":    row_m.group(9).strip(),
                        "organized": row_m.group(10).strip(),
                    })
                elif line.startswith("|"):
                    pass  # skip malformed rows silently
                else:
                    in_table = False  # blank line ends table

    return result


def _determine_status(pwm_psnr: float, ref_psnr: Optional[float]) -> str:
    """Map (pwm_psnr, ref_psnr) to algorithm_state.md status token."""
    if ref_psnr is None:
        return "ran"
    gap = ref_psnr - pwm_psnr
    if gap <= 3.0:
        return "done"
    elif gap <= 10.0:
        return "partial"
    else:
        return "gap"


# ─── algorithm_state.md updater ──────────────────────────────────────────────

def update_algorithm_state_md(
    modality: str, solver_name: str,
    pwm_psnr: float, pwm_ssim: float,
    ref_psnr: Optional[float],
) -> bool:
    """Update one algorithm row in algorithm_state.md with new PWM metrics.

    Returns True if a row was updated, False otherwise.
    """
    if not ALGO_STATE_PATH.exists():
        return False

    new_status = _determine_status(pwm_psnr, ref_psnr)
    psnr_str = f"{pwm_psnr:.1f}"
    ssim_str = f"{pwm_ssim:.4f}"

    text = ALGO_STATE_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")
    new_lines: List[str] = []

    modality_header_re = re.compile(
        rf"^###\s+\d+\.\s+.+\(`{re.escape(modality)}`\)"
    )
    in_modality = False
    updated = False
    name_lower = solver_name.lower()

    for line in lines:
        if modality_header_re.match(line):
            in_modality = True
            new_lines.append(line)
            continue
        if re.match(r"^###\s+\d+\.", line) and in_modality:
            in_modality = False

        if in_modality and "|" in line and not line.startswith("|---") and not updated:
            if name_lower in line.lower():
                parts = [p.strip() for p in line.split("|")]
                # Column layout after split:
                #   parts[0]="" | [1]=rank | [2]=name | [3]=year | [4]=ref
                #   | [5]=ref_psnr | [6]=ref_ssim | [7]=pwm_psnr
                #   | [8]=pwm_ssim | [9]=status | [10]=organized |
                if len(parts) >= 11:
                    parts[7] = psnr_str
                    parts[8] = ssim_str
                    parts[9] = new_status
                    line = "| " + " | ".join(p.strip() for p in parts[1:11]) + " |"
                    updated = True

        new_lines.append(line)

    if updated:
        ALGO_STATE_PATH.write_text("\n".join(new_lines), encoding="utf-8")
    return updated


# ─── check.md writer ─────────────────────────────────────────────────────────

def append_result_to_check_md(
    modality: str, solver_key: str, solver_name: str,
    solver_info: Dict, psnr_mean: float, ssim_mean: float,
    runtime_mean: float, n_samples: int,
    test_status: str, note: str = "",
    ref_info: Optional[Dict[str, str]] = None,
) -> None:
    """Append one test result block to benchmarks/learn/{modality}/check.md.

    When ref_info is provided (read from standard_references.md) three extra
    lines are added per result:
        **Operator Family:** <family>
        **Forward Model:**   <y = H(x) formula>
        **Canonical Reference:** <citation>
    """
    check_dir = LEARN_DIR / modality
    check_dir.mkdir(parents=True, exist_ok=True)
    check_path = check_dir / "check.md"

    today = date.today().isoformat()
    gpu_required = "Yes" if solver_info.get("gpu", False) else "No"
    algo_reference = solver_info.get("reference") or "—"
    algo_type = "GPU" if solver_info.get("gpu", False) else "Classical CPU"

    block = (
        f"\n---\n\n"
        f"## CPU Algorithm Test Results\n\n"
        f"**Algorithm:** {solver_name}\n"
        f"**Solver Key:** {solver_key}\n"
        f"**Type:** {algo_type}\n"
        f"**GPU Required:** {gpu_required}\n"
        f"**Test Date:** {today}\n"
        f"**Dataset:** public tier, {n_samples} sample(s)\n"
        f"**Status:** {test_status}\n"
        f"**Reference:** {algo_reference}\n"
    )

    # Physics annotations from standard_references.md
    if ref_info:
        family = ref_info.get("family", "other")
        fwd_model = ref_info.get("forward_model", "—")
        canonical_ref = ref_info.get("reference", "—")
        block += (
            f"**Operator Family:** {family}\n"
            f"**Forward Model:** {fwd_model}\n"
            f"**Canonical Reference:** {canonical_ref}\n"
        )

    if note:
        block += f"**Note:** {note}\n"

    block += (
        f"\n| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| PSNR (mean, {n_samples} samples) | {psnr_mean:.2f} dB |\n"
        f"| SSIM (mean, {n_samples} samples) | {ssim_mean:.4f} |\n"
        f"| Runtime | {runtime_mean:.2f} s/sample |\n"
        f"\n**Result: {test_status}**\n"
    )

    with open(check_path, "a", encoding="utf-8") as fh:
        fh.write(block)


# ─── Verified-solver detection ────────────────────────────────────────────────

def get_verified_solvers(modality: str) -> Set[str]:
    """Parse check.md and return lowercased names/keys of already-verified solvers.

    Matches both '**Algorithm:** name' and '**Solver Key:** key' patterns,
    handling the colon either inside or outside the bold markers.
    """
    check_path = LEARN_DIR / modality / "check.md"
    if not check_path.exists():
        return set()

    text = check_path.read_text(encoding="utf-8")
    verified: Set[str] = set()
    for m in re.finditer(
        r"\*\*(?:Algorithm|Solver Key):?\*\*:?\s*(.+?)[\n\r]",
        text,
    ):
        verified.add(m.group(1).strip().lower())
    return verified


def is_already_verified(solver_key: str, solver_name: str,
                        verified_set: Set[str]) -> bool:
    return (solver_key.lower() in verified_set
            or solver_name.lower() in verified_set)


# ─── Modality resolution ──────────────────────────────────────────────────────

def resolve_group(keyword: str, all_modalities: List[str]) -> List[str]:
    """Return modality IDs matching a group keyword.

    First checks MODALITY_GROUPS; falls back to substring match on modality ID.
    """
    kw = keyword.lower()
    if kw in MODALITY_GROUPS:
        return [m for m in MODALITY_GROUPS[kw] if m in all_modalities]
    return [m for m in all_modalities if kw in m.lower()]


def discover_all_modalities() -> List[str]:
    """Return sorted list of modality IDs that have a solvers.py in algorithm_base/."""
    return sorted(
        d.name
        for d in ALGO_BASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "solvers.py").exists()
    )


# ─── Core verification routine ────────────────────────────────────────────────

def verify_modality(
    modality: str,
    solver_keys: Optional[List[str]],
    unverified_only: bool,
    include_gpu: bool,
    max_samples: Optional[int],
    timeout: int,
    dry_run: bool,
    algo_state: Dict[str, List[Dict]],
) -> Dict:
    """Run verification for one modality.

    Args:
        modality:        Modality identifier string.
        solver_keys:     Explicit list of solver keys to run; None = all CPU.
        unverified_only: If True, skip solvers already in check.md.
        include_gpu:     If True, also run GPU-required solvers.
        max_samples:     Maximum number of H5 samples to process.
        timeout:         Per-solver wall-clock timeout in seconds.
        dry_run:         If True, print plan but do not execute.
        algo_state:      Pre-parsed algorithm_state.md data.

    Returns:
        Summary dict: {modality, status, results}.
    """
    print(f"\n{'='*64}")
    print(f"  Modality: {modality}")
    print(f"{'='*64}")

    # Physics metadata for this modality (from standard_references.md)
    ref_info = get_modality_ref_info(modality)
    family = ref_info.get("family", "other") if ref_info else "other"
    print(f"  Physics  : {family}"
          + (f"  [{ref_info.get('measurement_matrix', '')[:60]}]"
             if ref_info else ""))

    # Load solver registry from algorithm_base
    try:
        mod = importlib.import_module(f"algorithm_base.{modality}.solvers")
        solvers: Dict = getattr(mod, "SOLVERS", {})
    except Exception as exc:
        print(f"  [SKIP] Cannot import algorithm_base.{modality}.solvers — {exc}")
        return {"modality": modality, "status": "import_error", "results": {}}

    if not solvers:
        print(f"  [SKIP] SOLVERS dict is empty")
        return {"modality": modality, "status": "no_solvers", "results": {}}

    # Determine which solver keys to attempt
    if solver_keys:
        unknown = [k for k in solver_keys if k not in solvers]
        if unknown:
            print(f"  [WARN] Unknown solver keys (will skip): {unknown}")
        keys_to_run = [k for k in solver_keys if k in solvers]
    else:
        keys_to_run = [
            k for k, v in solvers.items()
            if include_gpu or not v.get("gpu", False)
        ]

    if not keys_to_run:
        reason = ("no CPU solvers (use --include-gpu to also run GPU solvers)"
                  if not include_gpu else "no solvers defined")
        print(f"  [SKIP] {reason}")
        return {"modality": modality, "status": "no_runnable_solvers", "results": {}}

    # Filter already-verified if requested
    if unverified_only:
        verified_set = get_verified_solvers(modality)
        keys_to_run = [
            k for k in keys_to_run
            if not is_already_verified(k, solvers[k].get("name", k), verified_set)
        ]
        if not keys_to_run:
            print(f"  [SKIP] All solvers already verified in check.md")
            return {"modality": modality, "status": "all_verified", "results": {}}

    # Load public dataset
    samples, h5_path = load_public_samples(modality, max_samples)
    if not samples:
        print(f"  [SKIP] No HDF5 dataset found at "
              f"datasets/benchmark/{modality}/public/")
        return {"modality": modality, "status": "no_data", "results": {}}

    print(f"  Dataset  : {Path(h5_path).name}  ({len(samples)} samples)")
    print(f"  Solvers  : {keys_to_run}")

    modality_algo_rows = algo_state.get(modality, [])
    results: Dict[str, Dict] = {}

    for solver_key in keys_to_run:
        solver_info = solvers[solver_key]
        solver_name = solver_info.get("name", solver_key)
        is_gpu = solver_info.get("gpu", False)

        print(f"\n  ─── {solver_name}  [{solver_key}] "
              f"{'(GPU)' if is_gpu else '(CPU)'} ───")

        if dry_run:
            print(f"      [DRY RUN] would run {solver_name} on {len(samples)} sample(s)")
            continue

        psnrs: List[float] = []
        ssims: List[float] = []
        times: List[float] = []
        last_error: Optional[str] = None

        for sample in samples:
            sample_key = sample["_key"]
            y, x_true = get_measurement_and_ground_truth(sample, modality)
            if y is None or x_true is None:
                print(f"      {sample_key}: [SKIP] missing y or x_true")
                continue

            # Build cfg: solver defaults + H5 metadata via physics-aware routing
            base_cfg = dict(solver_info.get("cfg_override", {}))
            cfg = build_solver_cfg(sample, modality, base_cfg)

            # Keep complex dtype for fourier-family modalities; float32 elsewhere
            if family != "fourier":
                y = np.asarray(y, dtype=np.float32)
            x_true_f = np.asarray(np.real(x_true), dtype=np.float32)

            x_hat, elapsed, error = run_solver_with_timeout(
                modality, solver_key, y, cfg, timeout
            )

            if error or x_hat is None:
                last_error = error or "returned None"
                print(f"      {sample_key}: ERROR — {last_error[:80]}")
                break  # abort remaining samples for this solver

            xh_2d = _to_2d_for_metrics(x_hat)
            xt_2d = _to_2d_for_metrics(x_true_f)

            psnr = compute_psnr(xh_2d, xt_2d)
            ssim = compute_ssim(xh_2d, xt_2d)
            psnrs.append(psnr)
            ssims.append(ssim)
            times.append(elapsed)
            print(f"      {sample_key}: PSNR={psnr:.2f} dB  "
                  f"SSIM={ssim:.4f}  t={elapsed:.2f}s")

        # Aggregate across samples
        if psnrs:
            mean_psnr = float(np.mean(psnrs))
            mean_ssim = float(np.mean(ssims))
            mean_time = float(np.mean(times))
            test_status = "PASS"
            note = f"{len(psnrs)} sample(s) measured."
        elif last_error:
            mean_psnr = 0.0
            mean_ssim = 0.0
            mean_time = 0.0
            test_status = "FAIL"
            note = f"Error: {last_error[:150]}"
        else:
            print(f"      No valid samples — skipping.")
            continue

        print(f"      => Mean PSNR={mean_psnr:.2f} dB  "
              f"SSIM={mean_ssim:.4f}  t={mean_time:.2f}s  [{test_status}]")

        results[solver_key] = {
            "name":      solver_name,
            "psnr":      mean_psnr,
            "ssim":      mean_ssim,
            "time":      mean_time,
            "n_samples": len(psnrs),
            "status":    test_status,
        }

        # Write result block to check.md (with physics annotations)
        append_result_to_check_md(
            modality=modality,
            solver_key=solver_key,
            solver_name=solver_name,
            solver_info=solver_info,
            psnr_mean=mean_psnr,
            ssim_mean=mean_ssim,
            runtime_mean=mean_time,
            n_samples=len(psnrs) if psnrs else 0,
            test_status=test_status,
            note=note,
            ref_info=ref_info or None,
        )

        # Update algorithm_state.md on success
        if test_status == "PASS":
            ref_psnr: Optional[float] = None
            for row in modality_algo_rows:
                if solver_name.lower() in row["name"].lower():
                    ref_psnr = row.get("ref_psnr")
                    break
            updated = update_algorithm_state_md(
                modality, solver_name, mean_psnr, mean_ssim, ref_psnr
            )
            if updated:
                algo_status = _determine_status(mean_psnr, ref_psnr)
                print(f"      algorithm_state.md updated: status={algo_status}")

    if dry_run:
        return {"modality": modality, "status": "dry_run", "results": {}}

    return {
        "modality": modality,
        "status":   "done" if results else "no_results",
        "results":  results,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_benchmark.py",
        description="PWM Benchmark Algorithm Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
Mode 1 — all modalities, all CPU solvers:
  python verify_benchmark.py --all

Mode 2 — specific modality (all CPU solvers):
  python verify_benchmark.py --modality cassi
  python verify_benchmark.py --modality mri --solver traditional_cpu

Mode 2 — specific modality, multiple solvers:
  python verify_benchmark.py --modality cassi --solver traditional_cpu twist

Mode 3 — named group (keyword matched against modality IDs):
  python verify_benchmark.py --group mri
  python verify_benchmark.py --group ct
  python verify_benchmark.py --group microscopy

Filter — only unverified algorithms (works with any mode):
  python verify_benchmark.py --all --unverified-only
  python verify_benchmark.py --group mri --unverified-only

Other options:
  python verify_benchmark.py --group mri --max-samples 3
  python verify_benchmark.py --modality cassi --include-gpu
  python verify_benchmark.py --all --dry-run

Available group keywords:
  mri, ct, xray, microscopy, ultrasound, optical, spectral,
  radio, pet, astronomy, ndt, compressive
  (or any substring of a modality ID)
""",
    )

    mode_group = p.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all", action="store_true",
        help="Run all algorithms for all modalities",
    )
    mode_group.add_argument(
        "--modality", metavar="MODALITY_ID",
        help="Run algorithms for a single modality (e.g. cassi, mri, ct)",
    )
    mode_group.add_argument(
        "--group", metavar="KEYWORD",
        help="Run all modalities in a named group or matching a substring",
    )

    p.add_argument(
        "--solver", metavar="SOLVER_KEY", nargs="+",
        help="One or more solver keys to run (only with --modality)",
    )
    p.add_argument(
        "--unverified-only", action="store_true",
        help="Skip solvers already recorded in benchmarks/learn/{modality}/check.md",
    )
    p.add_argument(
        "--include-gpu", action="store_true",
        help="Also run GPU-required solvers (default: CPU-only)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Use only the first N samples per modality (default: all)",
    )
    p.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT, metavar="SEC",
        help=f"Per-solver wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing any solver",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.solver and not args.modality:
        parser.error("--solver can only be used together with --modality")

    all_modalities = discover_all_modalities()
    print(f"Discovered {len(all_modalities)} modalities in algorithm_base/")

    # Warm up the reference table so every modality lookup below is instant
    refs = _load_refs_table()
    print(f"Loaded standard_references.md — {len(refs)} modality entries "
          f"across {len(set(v['family'] for v in refs.values()))} operator families")

    # Resolve target modality list
    if args.all:
        modalities_to_run = all_modalities
        solver_keys = None
    elif args.modality:
        if args.modality not in all_modalities:
            parser.error(
                f"Modality '{args.modality}' not found in algorithm_base/.\n"
                f"First few available: {all_modalities[:8]} ..."
            )
        modalities_to_run = [args.modality]
        solver_keys = args.solver
    else:  # --group
        modalities_to_run = resolve_group(args.group, all_modalities)
        if not modalities_to_run:
            parser.error(
                f"No modalities matched group '{args.group}'.\n"
                f"Known groups: {sorted(MODALITY_GROUPS.keys())}\n"
                f"Or use any substring of a modality ID."
            )
        solver_keys = None
        print(f"Group '{args.group}' → {len(modalities_to_run)} modalities: "
              f"{modalities_to_run}")

    if args.dry_run:
        print("\n[DRY RUN — no solvers will be executed]\n")

    print("Parsing algorithm_state.md …")
    algo_state = parse_algorithm_state()
    print(f"  Loaded {sum(len(v) for v in algo_state.values())} algorithm rows "
          f"across {len(algo_state)} modalities.")

    all_results: Dict[str, Dict] = {}
    t_start = time.time()

    for modality in modalities_to_run:
        result = verify_modality(
            modality=modality,
            solver_keys=solver_keys,
            unverified_only=args.unverified_only,
            include_gpu=args.include_gpu,
            max_samples=args.max_samples,
            timeout=args.timeout,
            dry_run=args.dry_run,
            algo_state=algo_state,
        )
        all_results[modality] = result

    total_time = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"SUMMARY  ({len(modalities_to_run)} modalities, {total_time:.1f}s total)")
    print(f"{'='*64}")

    n_pass = n_fail = n_skip = 0
    for mod, res in all_results.items():
        solver_results = res.get("results", {})
        if solver_results:
            passed = sum(1 for r in solver_results.values() if r["status"] == "PASS")
            failed = sum(1 for r in solver_results.values() if r["status"] == "FAIL")
            print(f"  {mod:<32}  {passed:3d} PASS  {failed:3d} FAIL"
                  f"  ({len(solver_results)} solvers)")
            n_pass += passed
            n_fail += failed
        else:
            status_label = res.get("status", "unknown")
            print(f"  {mod:<32}  [{status_label}]")
            n_skip += 1

    print(f"\nTotal:  {n_pass} PASS,  {n_fail} FAIL,  {n_skip} skipped/no-data")
    if not args.dry_run:
        for modality in modalities_to_run:
            print(f"\nResults written to : benchmarks/learn/{{modality}}/check.md")
        print(f"State updated in   : datasets/benchmark/algorithm_state.md")


if __name__ == "__main__":
    main()
