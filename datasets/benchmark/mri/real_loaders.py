"""Real brain MRI loaders for the PWM MRI benchmark.

Provides two loaders that return scene tuples compatible with build_dataset.py:

    (scene_name: str, x_true: np.ndarray[float32, (H,W)], recipe_str: str)

IXI T2w  →  dev tier  (healthy anatomy, multi-site, multi-field-strength)
BraTS T2w →  hidden tier (real pathology: GBM, LGG, meningioma, metastases)

Both loaders:
  - Accept a directory path (set via env vars IXI_T2_ROOT / BRATS_ROOT)
  - Normalise intensities to [0, 1]
  - Bicubic-resize slices to target shape (default 320×320)
  - Skip edge/empty slices
  - Return None if the directory is empty or nibabel unavailable
  - Fall back gracefully — callers should handle None return
"""

from __future__ import annotations

import os
import glob
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as _zoom

# nibabel is optional — loaders return None if not available
try:
    import nibabel as nib
    _NIBABEL_OK = True
except ImportError:
    _NIBABEL_OK = False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _require_nibabel(caller: str) -> bool:
    if not _NIBABEL_OK:
        print(f"  [WARNING] {caller}: nibabel not installed — cannot load NIfTI files.")
        print("  [WARNING]   Install with:  pip install nibabel")
        return False
    return True


def _load_nifti_volume(fpath: str) -> np.ndarray | None:
    """Load a NIfTI file and return float32 (H, W, D) array, or None on error."""
    try:
        img = nib.load(fpath)
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]  # drop 4th dim if present
        if data.ndim != 3:
            return None
        return data
    except Exception as exc:
        print(f"  [WARNING] NIfTI load failed for {os.path.basename(fpath)}: {exc}")
        return None


def _percentile_norm(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Robust percentile normalisation to [0, 1]."""
    lo = float(np.percentile(vol, p_low))
    hi = float(np.percentile(vol, p_high))
    if hi - lo < 1e-8:
        return np.zeros_like(vol)
    return ((vol - lo) / (hi - lo)).clip(0.0, 1.0)


def _extract_axial_slices(
    vol: np.ndarray,
    n_slices: int,
    target_shape: tuple[int, int],
    rng: np.random.Generator,
    depth_range: tuple[float, float] = (0.30, 0.70),
    min_mean: float = 0.06,
) -> list[tuple[int, np.ndarray]]:
    """Extract up to n_slices diverse axial slices from a 3D volume.

    Returns list of (slice_idx, slice_2d) pairs where slice_2d is float32 (H, W).
    """
    H_tgt, W_tgt = target_shape
    n_z = vol.shape[2]
    z0 = max(0, int(n_z * depth_range[0]))
    z1 = min(n_z, int(n_z * depth_range[1]))
    if z1 <= z0:
        return []

    # Sample evenly-spaced slice indices (with small jitter)
    candidate_z = np.linspace(z0, z1 - 1, n_slices * 3).astype(int)
    candidate_z = np.unique(np.clip(candidate_z, z0, z1 - 1))
    rng.shuffle(candidate_z)

    results = []
    for zi in candidate_z:
        if len(results) >= n_slices:
            break
        sl = vol[:, :, int(zi)].astype(np.float32)
        if float(sl.mean()) < min_mean:
            continue  # mostly empty
        if sl.shape != (H_tgt, W_tgt):
            sl = _zoom(sl, (H_tgt / sl.shape[0], W_tgt / sl.shape[1]), order=3)
            sl = sl.clip(0.0, 1.0).astype(np.float32)
        results.append((int(zi), sl))
    return results


# ── IXI T2w loader ────────────────────────────────────────────────────────────

def load_ixi_t2_slices(
    ixi_dir: str,
    n_samples: int = 20,
    target_shape: tuple[int, int] = (320, 320),
    seed: int = 42,
) -> list[tuple[str, np.ndarray, str]] | None:
    """Load n_samples axial T2w slices from the IXI dataset.

    IXI dataset layout (after extracting IXI-T2.tar):
        IXI002-Guys-0828-T2.nii.gz    (Guy's Hospital, 1.5 T, Philips)
        IXI012-HH-1211-T2.nii.gz      (Hammersmith, 3 T, Philips)
        IXI025-IOP-0888-T2.nii.gz     (IOP, 1.5 T, GE)

    Interleaves subjects from all three sites for diversity.

    Parameters
    ----------
    ixi_dir      : path to directory containing *T2*.nii.gz files
    n_samples    : number of (scene_name, x_true, recipe_str) tuples to return
    target_shape : (H, W) of output images
    seed         : RNG seed for reproducibility

    Returns
    -------
    List of (scene_name, x_true, recipe_str) or None if no data found.
    recipe_str encodes the acquisition site and field strength.
    """
    if not _require_nibabel("load_ixi_t2_slices"):
        return None

    ixi_dir = os.path.expanduser(ixi_dir)
    # Collect NIfTI files — support both flat and nested layout
    files = sorted(
        glob.glob(os.path.join(ixi_dir, "*T2*.nii.gz")) +
        glob.glob(os.path.join(ixi_dir, "**", "*T2*.nii.gz"), recursive=True)
    )
    files = sorted(set(files))

    if not files:
        print(f"  [IXI] No T2 NIfTI files found in {ixi_dir}")
        print("  [IXI]   Run: python download_datasets.py --ixi-dir " + ixi_dir)
        return None

    print(f"  [IXI] Found {len(files)} T2 NIfTI files in {ixi_dir}")

    # Sort by site: interleave HH (3T) > Guys (1.5T) > IOP (1.5T)
    site_order = {"HH": 0, "Guys": 1, "IOP": 2}
    def _site_key(f):
        bn = os.path.basename(f)
        for site in site_order:
            if f"-{site}-" in bn:
                return site_order[site]
        return 9
    hh_files   = [f for f in files if "-HH-"   in os.path.basename(f)]
    guys_files = [f for f in files if "-Guys-"  in os.path.basename(f)]
    iop_files  = [f for f in files if "-IOP-"   in os.path.basename(f)]
    other_files = [f for f in files if f not in hh_files + guys_files + iop_files]

    # Interleave sites
    interleaved: list[str] = []
    max_len = max(len(hh_files), len(guys_files), len(iop_files), len(other_files))
    for i in range(max_len):
        if i < len(hh_files):   interleaved.append(hh_files[i])
        if i < len(guys_files): interleaved.append(guys_files[i])
        if i < len(iop_files):  interleaved.append(iop_files[i])
        if i < len(other_files): interleaved.append(other_files[i])

    rng = np.random.default_rng(seed)
    scenes: list[tuple[str, np.ndarray, str]] = []

    for fpath in interleaved:
        if len(scenes) >= n_samples:
            break
        fname = Path(fpath).name.replace(".nii.gz", "")

        # Determine site / field strength for recipe label
        if "-HH-" in fname:
            site, tesla = "hammersmith", "3T"
        elif "-Guys-" in fname:
            site, tesla = "guys", "1.5T"
        elif "-IOP-" in fname:
            site, tesla = "iop", "1.5T"
        else:
            site, tesla = "unknown", "?"
        recipe_str = f"ixi_t2_{site}_{tesla}"

        vol = _load_nifti_volume(fpath)
        if vol is None:
            continue
        vol = _percentile_norm(vol)

        slices = _extract_axial_slices(
            vol,
            n_slices=1,
            target_shape=target_shape,
            rng=rng,
            depth_range=(0.30, 0.70),
        )

        for zi, sl in slices:
            scene_name = f"ixi_{fname}_sl{zi:03d}"
            scenes.append((scene_name, sl, recipe_str))
            print(f"  [IXI] {len(scenes)-1:02d} {scene_name}: "
                  f"shape={sl.shape}  mean={sl.mean():.3f}  site={site}/{tesla}")
            if len(scenes) >= n_samples:
                break

    if not scenes:
        print("  [IXI] No usable slices extracted — check NIfTI files.")
        return None

    print(f"  [IXI] Loaded {len(scenes)} slices from "
          f"{len(set(s[0].split('_sl')[0] for s in scenes))} subjects.")
    return scenes


# ── BraTS T2w loader ──────────────────────────────────────────────────────────

# Filename suffix patterns for each BraTS edition
_BRATS_T2_SUFFIXES = [
    "-t2w.nii.gz",    # BraTS 2024 GLI / SSA / PED / MEN / MET
    "-t2f.nii.gz",    # BraTS 2024 T2-FLAIR (fallback)
    "_t2.nii.gz",     # BraTS 2020 / 2021
    "_t2w.nii.gz",    # some BraTS 2023 variants
]

# Map folder prefixes to recipe labels
_BRATS_RECIPE_MAP = {
    "BraTS-GLI": "brats_glioma",
    "BraTS-MEN": "brats_meningioma",
    "BraTS-MET": "brats_metastases",
    "BraTS-PED": "brats_pediatric_glioma",
    "BraTS-SSA": "brats_glioma_ssa",
    "BraTS20_Training": "brats2020_glioma",
    "BraTS21_Training": "brats2021_glioma",
    "BRATS_": "brats_glioma",
}


def _brats_recipe(subject_dir: str) -> str:
    bn = os.path.basename(subject_dir)
    for prefix, recipe in _BRATS_RECIPE_MAP.items():
        if bn.startswith(prefix):
            return recipe
    return "brats_t2w"


def load_brats_t2_slices(
    brats_dir: str,
    n_samples: int = 20,
    target_shape: tuple[int, int] = (320, 320),
    seed: int = 42,
) -> list[tuple[str, np.ndarray, str]] | None:
    """Load n_samples axial T2w slices from a BraTS dataset.

    Compatible with BraTS 2020, 2021, 2023, and 2024 directory structures.
    Preferentially loads T2w; falls back to T2-FLAIR if T2w unavailable.
    Focuses on tumour-bearing slices (depth 40–70 % of the volume).

    Parameters
    ----------
    brats_dir    : root directory containing per-subject subdirectories
    n_samples    : number of (scene_name, x_true, recipe_str) tuples to return
    target_shape : (H, W) of output images
    seed         : RNG seed for reproducibility

    Returns
    -------
    List of (scene_name, x_true, recipe_str) or None if no data found.
    recipe_str encodes the tumour sub-type (glioma, meningioma, etc.)
    """
    if not _require_nibabel("load_brats_t2_slices"):
        return None

    brats_dir = os.path.expanduser(brats_dir)

    # Collect all T2w NIfTI files across the directory tree
    t2_files: list[str] = []
    for suffix in _BRATS_T2_SUFFIXES:
        # Prefer T2w over T2-FLAIR (first suffix wins)
        pattern = os.path.join(brats_dir, "**", f"*{suffix}")
        t2_files.extend(glob.glob(pattern, recursive=True))
    # Deduplicate; prefer true T2w over FLAIR (earlier in suffix list = lower index)
    seen_subjects: set[str] = set()
    ordered: list[str] = []
    for fpath in t2_files:
        subj = os.path.basename(os.path.dirname(fpath))
        if subj not in seen_subjects:
            seen_subjects.add(subj)
            ordered.append(fpath)
        # else: already have a T2 file for this subject

    if not ordered:
        print(f"  [BraTS] No T2w NIfTI files found under {brats_dir}")
        print("  [BraTS]   Download BraTS from https://www.synapse.org/brats2024")
        print("  [BraTS]   Then set: export BRATS_ROOT=" + brats_dir)
        return None

    print(f"  [BraTS] Found {len(ordered)} T2w files across "
          f"{len(seen_subjects)} subjects in {brats_dir}")

    rng = np.random.default_rng(seed)
    # Shuffle subjects for diversity (deterministic)
    order = rng.permutation(len(ordered)).tolist()
    ordered = [ordered[i] for i in order]

    scenes: list[tuple[str, np.ndarray, str]] = []

    for fpath in ordered:
        if len(scenes) >= n_samples:
            break

        subject_dir = os.path.dirname(fpath)
        subj_name   = os.path.basename(subject_dir)
        recipe_str  = _brats_recipe(subject_dir)

        vol = _load_nifti_volume(fpath)
        if vol is None:
            continue
        # BraTS volumes can have background zeros and large white matter
        # Use robust normalisation relative to p99 of foreground voxels
        fg = vol[vol > vol.max() * 0.01]
        if fg.size == 0:
            continue
        p99 = float(np.percentile(fg, 99))
        if p99 < 1e-8:
            continue
        vol = (vol / p99).clip(0.0, 1.0).astype(np.float32)

        # BraTS volumes are typically 240×240×155 (axial orientation)
        # Tumor most visible between 40–70% depth
        slices = _extract_axial_slices(
            vol,
            n_slices=min(2, n_samples - len(scenes)),
            target_shape=target_shape,
            rng=rng,
            depth_range=(0.40, 0.70),
            min_mean=0.04,
        )

        for zi, sl in slices:
            scene_name = f"brats_{subj_name}_sl{zi:03d}"
            scenes.append((scene_name, sl, recipe_str))
            print(f"  [BraTS] {len(scenes)-1:02d} {scene_name}: "
                  f"shape={sl.shape}  mean={sl.mean():.3f}  recipe={recipe_str}")
            if len(scenes) >= n_samples:
                break

    if not scenes:
        print("  [BraTS] No usable slices extracted — check directory structure.")
        return None

    print(f"  [BraTS] Loaded {len(scenes)} slices from "
          f"{len(set(s[0].split('_sl')[0] for s in scenes))} subjects.")
    return scenes
