#!/usr/bin/env python3
"""Shared real-data loaders for Phase 2 hardware validation.

Provides functions to load real experimental datasets for:
- Ultrasound: PICMUS experimental UFF + DeepUS CIRS040GSE
- Cryo-EM: EMDB 3D maps projected to 2D
- CT: LoDoPaB-CT, FIPS walnut, HTC 2022

These loaders reuse existing infrastructure from the benchmark dataset
generators but expose a simplified API for the multiphantom scripts.
"""
from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "datasets" / "benchmark"
REAL_CT_DIR = PROJECT_ROOT / "datasets" / "real_ct"
EMDB_CACHE = Path("/tmp/emdb_cache")
EMDB_CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ultrasound: PICMUS + DeepUS
# ---------------------------------------------------------------------------

PICMUS_SCENE_NAMES = {
    "PICMUS_experiment_resolution_distortion": "exp_resolution",
    "PICMUS_experiment_contrast_speckle": "exp_contrast",
    "PICMUS_carotid_cross": "invivo_carotid_cross",
    "PICMUS_carotid_long": "invivo_carotid_long",
}

# Only experimental UFF files (exclude simulations)
PICMUS_EXPERIMENTAL = [
    "PICMUS_experiment_resolution_distortion",
    "PICMUS_experiment_contrast_speckle",
    "PICMUS_carotid_cross",
    "PICMUS_carotid_long",
]


def load_picmus_real(n_samples: int = 5) -> list[tuple[str, dict]]:
    """Load real PICMUS experimental RF data + 1 DeepUS sample.

    Returns list of (name, {"rf_75": ndarray, "fs": float, "c": float,
    "n_elements": int, "pitch": float, "source": str}).
    Selects 4 experimental PICMUS UFF files + 1 DeepUS MAT file.
    """
    picmus_dir = BENCHMARK_DIR / "ultrasound" / "picmus_src"
    deepus_dir = BENCHMARK_DIR / "ultrasound" / "deepus_src" / "CIRS040GSE"

    results = []

    # Load experimental PICMUS UFF files (skip simulations)
    if picmus_dir.is_dir():
        for uff_path in sorted(picmus_dir.glob("*.uff")):
            stem = uff_path.stem
            if stem not in PICMUS_EXPERIMENTAL:
                continue
            scene_name = PICMUS_SCENE_NAMES.get(stem, stem)
            try:
                with h5py.File(uff_path, "r") as f:
                    cd = f["channel_data"]
                    rf = cd["data"][:].astype(np.float32)
                    fs = float(cd["sampling_frequency"][0, 0])
                    c = float(cd["sound_speed"][0, 0])
                    n_elem = int(cd["probe/N"][0, 0])
                    pitch = float(cd["probe/pitch"][0, 0])
                results.append((scene_name, {
                    "rf_75": rf,
                    "fs": fs,
                    "c": c,
                    "n_elements": n_elem,
                    "pitch": pitch,
                    "source": f"PICMUS/{uff_path.name}",
                }))
            except Exception as e:
                print(f"  [WARN] {uff_path.name} skipped: {e}")

    # Load 1 DeepUS sample (low attenuation, first file)
    if deepus_dir.is_dir() and len(results) < n_samples:
        lo_dir = deepus_dir / "low_attenuation"
        if lo_dir.is_dir():
            mat_files = sorted(lo_dir.glob("USDATA_*.mat"))
            if mat_files:
                mat_path = mat_files[0]
                try:
                    with h5py.File(mat_path, "r") as f:
                        rf = f["USDATA"][:].astype(np.float32)
                    rf = rf[:, 0, :, :]  # (75, 128, N_samples)
                    results.append(("deepus_cirs_lo", {
                        "rf_75": rf,
                        "fs": 31.25e6,
                        "c": 1540.0,
                        "n_elements": 128,
                        "pitch": 0.195e-3,
                        "source": f"DeepUS/{mat_path.name}",
                    }))
                except Exception as e:
                    print(f"  [WARN] DeepUS skipped: {e}")

    return results[:n_samples]


# ---------------------------------------------------------------------------
# Cryo-EM: EMDB structures
# ---------------------------------------------------------------------------

# 5 well-known, diverse EMDB structures
EMDB_STRUCTURES = [
    ("EMD-5778", "trpv1", "TRPV1 ion channel"),
    ("EMD-2984", "beta_galactosidase", "Beta-galactosidase 2.2 A"),
    ("EMD-6287", "t20s_proteasome", "T20S proteasome 2.8 A"),
    ("EMD-11103", "apoferritin", "Apoferritin 1.25 A atomic res"),
    ("EMD-21375", "sars_cov2_spike", "SARS-CoV-2 spike glycoprotein"),
]


def _download_emdb(emd_id: str) -> np.ndarray:
    """Download an EMDB map and return the 3D volume."""
    numeric = emd_id.replace("EMD-", "")
    gz_path = EMDB_CACHE / f"emd_{numeric}.map.gz"
    map_path = EMDB_CACHE / f"emd_{numeric}.map"

    if not map_path.exists():
        url = (f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
               f"{emd_id}/map/emd_{numeric}.map.gz")
        print(f"    Downloading {emd_id} from EMDB...")
        urllib.request.urlretrieve(url, str(gz_path))
        with gzip.open(str(gz_path), "rb") as f_in:
            with open(str(map_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)
    else:
        print(f"    Using cached {emd_id}")

    import mrcfile
    with mrcfile.open(str(map_path), mode="r", permissive=True) as mrc:
        vol = mrc.data.copy()
    return vol


def _project_and_resize(vol: np.ndarray, shape: tuple[int, int],
                        proj_axis: int = 0) -> np.ndarray:
    """Project 3D volume along an axis and resize to target shape."""
    proj = vol.sum(axis=proj_axis).astype(np.float64)
    zy = shape[0] / proj.shape[0]
    zx = shape[1] / proj.shape[1]
    proj_resized = zoom(proj, (zy, zx), order=3)
    proj_resized -= proj_resized.min()
    pmax = proj_resized.max()
    if pmax > 1e-8:
        proj_resized /= pmax
    return proj_resized.astype(np.float32)


def _random_rotation_3d(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random 3D rotation via sequential axis rotations."""
    from scipy.ndimage import rotate
    alpha = rng.uniform(0, 360)
    beta = rng.uniform(0, 360)
    gamma = rng.uniform(0, 360)
    v = rotate(vol, alpha, axes=(1, 2), reshape=False, order=1, mode="constant")
    v = rotate(v, beta, axes=(0, 2), reshape=False, order=1, mode="constant")
    v = rotate(v, gamma, axes=(0, 1), reshape=False, order=1, mode="constant")
    return v


def load_emdb_real_phantoms(
    size: int = 128,
    emdb_ids: list[tuple[str, str, str]] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Download and project 5 EMDB structures to 2D phantoms.

    Returns list of (short_name, phantom_2d) where phantom_2d is (size, size)
    normalized to [0, 1].
    """
    if emdb_ids is None:
        emdb_ids = EMDB_STRUCTURES

    results = []
    for idx, (emd_id, short_name, _desc) in enumerate(emdb_ids):
        rng = np.random.default_rng(42 + idx)
        vol = _download_emdb(emd_id)
        vol = np.clip(vol, 0, None)
        vol = _random_rotation_3d(vol, rng)
        phantom = _project_and_resize(vol, (size, size))
        results.append((short_name, phantom))
        print(f"    {emd_id} ({short_name}): vol={vol.shape} -> phantom={phantom.shape}")

    return results


# ---------------------------------------------------------------------------
# CT: LoDoPaB-CT, FIPS walnut, HTC 2022
# ---------------------------------------------------------------------------

def load_lodopab_real_slices(n_slices: int = 3, size: int = 128) -> list[tuple[str, np.ndarray]]:
    """Load real CT slices from ct_challenge_public.h5.

    The H5 file contains sample groups (sample_00, sample_01, ...) each with
    'x_true' (362x362 ground truth), 'sinogram_ideal', 'sinogram_measured'.

    Returns list of (name, image) where image is (size, size) normalized.
    """
    h5_path = BENCHMARK_DIR / "ct" / "public" / "ct_challenge_public.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"CT challenge H5 not found: {h5_path}")

    results = []
    with h5py.File(h5_path, "r") as f:
        sample_keys = sorted([k for k in f.keys() if k.startswith("sample_")])
        # Select evenly spaced samples
        indices = np.linspace(0, len(sample_keys) - 1, n_slices, dtype=int)
        for i, idx in enumerate(indices):
            key = sample_keys[idx]
            img = f[key]["x_true"][:].astype(np.float64)
            if img.shape[0] != size or img.shape[1] != size:
                zy = size / img.shape[0]
                zx = size / img.shape[1]
                img = zoom(img, (zy, zx), order=3)
            img -= img.min()
            pmax = img.max()
            if pmax > 1e-8:
                img /= pmax
            results.append((f"lodopab_{key}", img.astype(np.float64)))

    return results[:n_slices]


def load_fips_walnut(size: int = 128) -> tuple[str, np.ndarray]:
    """Load FIPS walnut central slice from GroundTruthReconstruction.mat.

    Returns (name, image) where image is (size, size) normalized.
    """
    import scipy.io as sio

    mat_path = REAL_CT_DIR / "GroundTruthReconstruction.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"FIPS walnut not found: {mat_path}")

    try:
        data = sio.loadmat(str(mat_path))
    except NotImplementedError:
        # v7.3 MAT file - use h5py
        with h5py.File(mat_path, "r") as f:
            key = [k for k in f.keys() if not k.startswith("__")][0]
            vol = f[key][:].astype(np.float64)
    else:
        key = [k for k in data.keys() if not k.startswith("__")][0]
        vol = data[key].astype(np.float64)

    # Extract central slice
    if vol.ndim == 3:
        mid = vol.shape[0] // 2
        img = vol[mid]
    else:
        img = vol

    if img.shape[0] != size or img.shape[1] != size:
        zy = size / img.shape[0]
        zx = size / img.shape[1]
        img = zoom(img, (zy, zx), order=3)

    img -= img.min()
    pmax = img.max()
    if pmax > 1e-8:
        img /= pmax

    return ("fips_walnut", img.astype(np.float64))


def load_htc2022(size: int = 128) -> tuple[str, np.ndarray]:
    """Load HTC 2022 sample A FBP reconstruction from htc2022_ta_full_recon_fbp.mat.

    The raw htc2022_ta_full.mat contains only sinogram data (structured array).
    We use the pre-computed FBP reconstruction instead.

    Returns (name, image) where image is (size, size) normalized.
    """
    import scipy.io as sio

    # Use the FBP reconstruction file (512x512 float64 image)
    recon_path = REAL_CT_DIR / "htc2022_ta_full_recon_fbp.mat"
    if not recon_path.exists():
        raise FileNotFoundError(f"HTC 2022 FBP recon not found: {recon_path}")

    try:
        data = sio.loadmat(str(recon_path))
    except NotImplementedError:
        with h5py.File(recon_path, "r") as f:
            key = [k for k in f.keys() if not k.startswith("__")][0]
            img = f[key][:].astype(np.float64)
    else:
        key = [k for k in data.keys() if not k.startswith("__")][0]
        img = data[key].astype(np.float64)

    # If 3D, take central slice
    if img.ndim == 3:
        mid = img.shape[0] // 2
        img = img[mid]

    if img.shape[0] != size or img.shape[1] != size:
        zy = size / img.shape[0]
        zx = size / img.shape[1]
        img = zoom(img, (zy, zx), order=3)

    img -= img.min()
    pmax = img.max()
    if pmax > 1e-8:
        img /= pmax

    return ("htc2022_sampleA", img.astype(np.float64))


def load_real_ct_phantoms(n_lodopab: int = 3, size: int = 128) -> list[tuple[str, np.ndarray]]:
    """Load all real CT phantoms: LoDoPaB slices + FIPS walnut + HTC 2022.

    Returns list of (name, image) with 5 total phantoms.
    """
    results = load_lodopab_real_slices(n_slices=n_lodopab, size=size)

    try:
        results.append(load_fips_walnut(size=size))
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    try:
        results.append(load_htc2022(size=size))
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    return results


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def compute_self_ref_metrics(
    recon_ref: np.ndarray,
    recon_test: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> dict:
    """Compute PSNR/SSIM using either true GT or self-reference.

    If ground_truth is None, uses recon_ref as pseudo-GT (self-reference).
    """
    ref = ground_truth if ground_truth is not None else recon_ref
    ref64 = ref.astype(np.float64)
    test64 = recon_test.astype(np.float64)

    # PSNR
    mse = np.mean((ref64 - test64) ** 2)
    if mse < 1e-15:
        p = 100.0
    else:
        max_val = float(np.max(ref64) - np.min(ref64))
        p = float(10.0 * np.log10(max_val ** 2 / mse)) if max_val > 1e-10 else 0.0

    # SSIM
    from scipy.ndimage import uniform_filter
    win = 7
    L = float(ref64.max() - ref64.min())
    if L < 1e-10:
        s = 0.0
    else:
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        mu_x = uniform_filter(ref64, win)
        mu_y = uniform_filter(test64, win)
        sigma_x2 = uniform_filter(ref64 ** 2, win) - mu_x ** 2
        sigma_y2 = uniform_filter(test64 ** 2, win) - mu_y ** 2
        sigma_xy = uniform_filter(ref64 * test64, win) - mu_x * mu_y
        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
        s = float(np.mean(num / den))

    return {"psnr": p, "ssim": s, "mode": "true_gt" if ground_truth is not None else "self_ref"}
