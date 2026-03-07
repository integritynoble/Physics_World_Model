"""Format-specific download and conversion functions.

All converters return ``np.ndarray`` in ``[0, 1]`` float32, saved as
``.npy`` in the local cache.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Cache root lives alongside benchmark results
CACHE_ROOT = Path(__file__).parent.parent / "results" / ".data_cache"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file from *url* to *dest* with progress logging.

    Returns *dest* on success.  Raises on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest

    logger.info("Downloading %s -> %s", url, dest.name)
    req = urllib.request.Request(url, headers={"User-Agent": "PWM-Benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = resp.headers.get("Content-Length")
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded % (1 << 20) < chunk_size:
                pct = 100.0 * downloaded / int(total)
                logger.debug("  %.0f%% (%d / %s bytes)", pct, downloaded, total)
    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalise to ``[0, 1]`` float32."""
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo)
    return arr


def crop_or_resize(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Crop centre or zero-pad *arr* to *target_shape*.

    Does NOT interpolate — crops from the centre, pads with zeros.
    """
    if arr.shape == target_shape:
        return arr

    out = np.zeros(target_shape, dtype=arr.dtype)
    slices_src = []
    slices_dst = []
    for i in range(min(arr.ndim, len(target_shape))):
        src_size = arr.shape[i]
        tgt_size = target_shape[i]
        if src_size >= tgt_size:
            # Centre crop
            start = (src_size - tgt_size) // 2
            slices_src.append(slice(start, start + tgt_size))
            slices_dst.append(slice(0, tgt_size))
        else:
            # Zero-pad
            start = (tgt_size - src_size) // 2
            slices_src.append(slice(0, src_size))
            slices_dst.append(slice(start, start + src_size))

    # Handle dimension mismatch
    if arr.ndim < len(target_shape):
        for _ in range(len(target_shape) - arr.ndim):
            arr = arr[..., np.newaxis]
            slices_src.append(slice(0, 1))
            slices_dst.append(slice(0, 1))
        out[tuple(slices_dst)] = arr[tuple(slices_src)]
        # Broadcast along new axes
        if len(target_shape) > 0 and target_shape[-1] > 1 and arr.shape[-1] == 1:
            for c in range(1, target_shape[-1]):
                out[..., c] = out[..., 0]
    elif arr.ndim > len(target_shape):
        squeezed = arr
        for _ in range(arr.ndim - len(target_shape)):
            squeezed = squeezed[..., 0]
        slices_src = slices_src[: len(target_shape)]
        out[tuple(slices_dst)] = squeezed[tuple(slices_src)]
    else:
        out[tuple(slices_dst)] = arr[tuple(slices_src)]

    return out


# ---------------------------------------------------------------------------
# .mat
# ---------------------------------------------------------------------------

def convert_mat(
    path: Path,
    key: Optional[str] = None,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load a MATLAB ``.mat`` file and return the image array.

    Auto-detects the data key if *key* is ``None``.
    """
    import scipy.io as sio

    try:
        mat = sio.loadmat(str(path))
    except NotImplementedError:
        # v7.3 (HDF5-backed) .mat file
        return convert_hdf5(path, key=key, target_shape=target_shape)

    if key and key in mat:
        arr = np.asarray(mat[key])
    else:
        # Auto-detect: skip MATLAB meta keys
        candidates = {
            k: v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray)
        }
        if not candidates:
            raise ValueError(f"No array found in {path}")
        # Pick the largest array
        arr = max(candidates.values(), key=lambda a: a.size)

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def convert_hdf5(
    path: Path,
    key: Optional[str] = None,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load an HDF5 file and return the image array."""
    import h5py

    with h5py.File(str(path), "r") as f:
        if key and key in f:
            arr = np.asarray(f[key])
        else:
            # Auto-detect: find the largest dataset
            datasets = []
            def _visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append((name, obj.shape, obj.size))
            f.visititems(_visitor)
            if not datasets:
                raise ValueError(f"No datasets found in {path}")
            best = max(datasets, key=lambda t: t[2])
            arr = np.asarray(f[best[0]])

    # For LoDoPaB: shape is (N, H, W) — pick first slice if batch
    if arr.ndim == 3 and arr.shape[0] > 4 and arr.shape[1] == arr.shape[2]:
        arr = arr[0]

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# TIFF
# ---------------------------------------------------------------------------

def convert_tiff(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load a single TIFF or multi-page TIFF stack."""
    try:
        import tifffile
        arr = tifffile.imread(str(path))
    except ImportError:
        from PIL import Image
        img = Image.open(str(path))
        arr = np.asarray(img)

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------

def convert_nifti(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load a NIfTI (.nii / .nii.gz) volume, extract central slice."""
    import nibabel as nib

    img = nib.load(str(path))
    vol = np.asarray(img.dataobj)

    # For 3-D volumes, extract the central axial slice
    if vol.ndim >= 3:
        mid = vol.shape[2] // 2
        arr = vol[:, :, mid].astype(np.float32)
    else:
        arr = vol.astype(np.float32)

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# PNG stack
# ---------------------------------------------------------------------------

def convert_png_stack(
    directory: Path,
    n_images: int = 1,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load the first *n_images* PNG/JPG files from a directory.

    Returns a single 2-D image (first found) or a 3-D stack.
    """
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in exts
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {directory}")

    if n_images == 1:
        img = Image.open(image_files[0]).convert("L")
        arr = np.asarray(img, dtype=np.float32)
    else:
        frames = []
        for p in image_files[:n_images]:
            img = Image.open(p).convert("L")
            frames.append(np.asarray(img, dtype=np.float32))
        arr = np.stack(frames, axis=-1) if len(frames) > 1 else frames[0]

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# MRC (cryo-EM)
# ---------------------------------------------------------------------------

def convert_mrc(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load an MRC file (cryo-EM micrograph)."""
    try:
        import mrcfile
        with mrcfile.open(str(path), permissive=True) as mrc:
            arr = np.copy(mrc.data)
    except ImportError:
        # Fallback: read raw binary (MRC header is 1024 bytes)
        raw = np.fromfile(str(path), dtype=np.float32, offset=1024)
        side = int(np.sqrt(raw.size))
        arr = raw[:side * side].reshape(side, side)

    if arr.ndim == 3:
        arr = arr[0]
    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def extract_zip(
    zip_path: Path,
    extract_dir: Optional[Path] = None,
) -> Path:
    """Extract a ZIP archive to *extract_dir* (defaults to sibling dir).

    Returns the extraction directory.
    """
    if extract_dir is None:
        extract_dir = zip_path.parent / zip_path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(extract_dir))

    logger.info("Extracted %s -> %s", zip_path.name, extract_dir)
    return extract_dir


# ---------------------------------------------------------------------------
# Generated surface (scanning probe fallback)
# ---------------------------------------------------------------------------

def generate_surface(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic surface topography for scanning-probe benchmarks.

    Combines a fractal background with step edges and point defects.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Fractal background via FFT
    freq_x = np.fft.fftfreq(W)
    freq_y = np.fft.fftfreq(H)
    FX, FY = np.meshgrid(freq_x, freq_y)
    radius = np.sqrt(FX**2 + FY**2)
    radius[0, 0] = 1.0  # avoid divide by zero

    # 1/f^beta noise (beta ~ 2 for Brownian surface)
    power = 1.0 / (radius ** 2)
    phase = rng.uniform(0, 2 * np.pi, (H, W))
    fft_data = np.sqrt(power) * np.exp(1j * phase)
    surface = np.real(np.fft.ifft2(fft_data)).astype(np.float32)

    # Add step edge
    surface[:, W // 2:] += 0.3

    # Add point defects
    n_defects = 10
    for _ in range(n_defects):
        cx, cy = rng.randint(0, W), rng.randint(0, H)
        r = rng.randint(2, 6)
        yy, xx = np.ogrid[-cy:H - cy, -cx:W - cx]
        mask = xx**2 + yy**2 <= r**2
        surface[mask] += rng.uniform(-0.2, 0.2)

    return normalize_array(surface)


# ---------------------------------------------------------------------------
# BrainWeb PET phantom (simplified synthetic)
# ---------------------------------------------------------------------------

def convert_brainweb(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load BrainWeb raw data or generate a simple brain PET phantom.

    If the raw file is available, load it; otherwise generate a simple
    elliptical phantom mimicking PET uptake patterns.
    """
    if path.exists() and path.stat().st_size > 0:
        raw = np.fromfile(str(path), dtype=np.uint8)
        side = int(np.cbrt(raw.size))
        if side ** 3 == raw.size:
            vol = raw.reshape(side, side, side)
            arr = vol[side // 2].astype(np.float32)
        else:
            side2 = int(np.sqrt(raw.size))
            arr = raw[:side2 * side2].reshape(side2, side2).astype(np.float32)
    else:
        # Generate simple PET-like phantom
        H = target_shape[0] if target_shape else 256
        W = target_shape[1] if target_shape and len(target_shape) > 1 else H
        yy = np.linspace(-1, 1, H)
        xx = np.linspace(-1, 1, W)
        X, Y = np.meshgrid(xx, yy)
        arr = np.zeros((H, W), dtype=np.float32)
        # Skull outline
        arr[(X**2 + Y**2) < 0.8] = 0.2
        # Gray matter (higher uptake)
        arr[((X**2 + Y**2) < 0.6) & ((X**2 + Y**2) > 0.35)] = 0.8
        # White matter
        arr[(X**2 + Y**2) < 0.35] = 0.4
        # Ventricles (low uptake)
        arr[((X / 0.08)**2 + ((Y + 0.05) / 0.15)**2) < 1] = 0.05
        arr[((X / 0.08)**2 + ((Y - 0.05) / 0.15)**2) < 1] = 0.05

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# LiDAR binary (KITTI format)
# ---------------------------------------------------------------------------

def convert_lidar_bin(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Convert KITTI-format LiDAR .bin to a 2-D range image."""
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    # Spherical projection
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    H = target_shape[0] if target_shape else 64
    W = target_shape[1] if target_shape and len(target_shape) > 1 else 256

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(np.clip(z / np.clip(r, 1e-8, None), -1, 1))

    # Bin into image
    az_bins = np.linspace(-np.pi, np.pi, W + 1)
    el_bins = np.linspace(-0.5, 0.3, H + 1)  # typical Velodyne range

    range_image = np.zeros((H, W), dtype=np.float32)
    az_idx = np.digitize(azimuth, az_bins) - 1
    el_idx = np.digitize(elevation, el_bins) - 1
    valid = (az_idx >= 0) & (az_idx < W) & (el_idx >= 0) & (el_idx < H)
    range_image[el_idx[valid], az_idx[valid]] = r[valid]

    return normalize_array(range_image)


# ---------------------------------------------------------------------------
# MAT v7.3 (HDF5-backed MATLAB files)
# ---------------------------------------------------------------------------

def convert_mat_v73(
    path: Path,
    key: Optional[str] = None,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Load a MATLAB v7.3 (HDF5) ``.mat`` file.

    Tries h5py first, then mat73, then falls back to convert_mat (which
    will itself try HDF5).
    """
    # Try h5py
    try:
        return convert_hdf5(path, key=key, target_shape=target_shape)
    except Exception:
        pass

    # Try mat73
    try:
        import mat73
        data = mat73.loadmat(str(path))
        if key and key in data:
            arr = np.asarray(data[key])
        else:
            # Pick the largest array
            candidates = {k: np.asarray(v) for k, v in data.items()
                          if isinstance(v, np.ndarray)}
            if not candidates:
                raise ValueError(f"No arrays in {path}")
            arr = max(candidates.values(), key=lambda a: a.size)
        arr = normalize_array(arr)
        if target_shape:
            arr = crop_or_resize(arr, target_shape)
        return arr
    except ImportError:
        pass

    # Final fallback
    return convert_mat(path, key=key, target_shape=target_shape)


# ---------------------------------------------------------------------------
# NIfTI from ZIP archive (COVID CT etc.)
# ---------------------------------------------------------------------------

def convert_nifti_from_zip(
    zip_path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Extract a ZIP containing NIfTI volumes, load the first one."""
    extract_dir = extract_zip(zip_path)

    # Find .nii or .nii.gz files
    nifti_files = sorted(
        list(extract_dir.rglob("*.nii.gz")) + list(extract_dir.rglob("*.nii"))
    )
    if not nifti_files:
        # Fallback: try loading as PNG stack
        return convert_png_stack(extract_dir, n_images=1, target_shape=target_shape)

    try:
        return convert_nifti(nifti_files[0], target_shape=target_shape)
    except ImportError:
        logger.warning("nibabel not installed; trying raw NIfTI load for %s", nifti_files[0].name)
        return _load_nifti_raw(nifti_files[0], target_shape=target_shape)


def _load_nifti_raw(
    path: Path,
    target_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Minimal NIfTI-1 loader without nibabel (uncompressed only)."""
    import gzip
    import struct

    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(str(path), "rb") as f:
        header = f.read(348)
        # NIfTI-1: dims at offset 40, datatype at 70
        dims = struct.unpack_from("<8h", header, 40)
        ndim = dims[0]
        shape = tuple(dims[1:ndim + 1])
        datatype = struct.unpack_from("<h", header, 70)[0]
        vox_offset = struct.unpack_from("<f", header, 108)[0]

        dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
        dt = dtype_map.get(datatype, np.float32)

        f.seek(int(vox_offset))
        data = np.frombuffer(f.read(), dtype=dt)

    total_elements = 1
    for s in shape:
        total_elements *= s
    vol = data[:total_elements].reshape(shape)

    # Extract central axial slice
    if vol.ndim >= 3:
        mid = vol.shape[2] // 2
        arr = vol[:, :, mid].astype(np.float32)
    else:
        arr = vol.astype(np.float32)

    arr = normalize_array(arr)
    if target_shape:
        arr = crop_or_resize(arr, target_shape)
    return arr


# ---------------------------------------------------------------------------
# Generated OCT phantom
# ---------------------------------------------------------------------------

def generate_oct_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic OCT B-scan phantom.

    Creates a multi-layer retinal structure with speckle noise.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Layer boundaries with slight curvature
    x_coord = np.linspace(0, np.pi, W)
    base_curve = 0.05 * np.sin(x_coord) + 0.02 * np.sin(3 * x_coord)

    # Define retinal layers (fraction of height from top)
    layer_fracs = [0.25, 0.30, 0.35, 0.42, 0.50, 0.55, 0.65, 0.72, 0.80]
    layer_intensities = [0.9, 0.3, 0.7, 0.2, 0.8, 0.3, 0.6, 0.2, 0.85]

    for i, (frac, intensity) in enumerate(zip(layer_fracs, layer_intensities)):
        boundary = (frac + base_curve) * H
        thickness = max(2, int(0.03 * H))
        for col in range(W):
            row = int(boundary[col])
            r0 = max(0, row - thickness // 2)
            r1 = min(H, row + thickness // 2)
            arr[r0:r1, col] = intensity

    # Background tissue between layers
    for row in range(H):
        for i in range(len(layer_fracs) - 1):
            b_top = int((layer_fracs[i] + 0.03 + base_curve.mean()) * H)
            b_bot = int((layer_fracs[i + 1] - 0.03 + base_curve.mean()) * H)
            if b_top <= row < b_bot:
                arr[row, :] = np.clip(arr[row, :] + 0.15, 0, 1)

    # Speckle noise (multiplicative, characteristic of OCT)
    speckle = rng.exponential(1.0, (H, W)).astype(np.float32)
    arr = arr * (0.7 + 0.3 * speckle)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated SMLM phantom (PALM/STORM)
# ---------------------------------------------------------------------------

def generate_smlm_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a sparse single-molecule localization microscopy image.

    Creates a ground-truth emitter field: sparse point sources convolved
    with Gaussian PSFs to simulate super-resolution microscopy data.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Random point emitters
    n_emitters = 200
    y_pos = rng.randint(10, H - 10, n_emitters)
    x_pos = rng.randint(10, W - 10, n_emitters)
    intensities = rng.uniform(0.5, 1.0, n_emitters)

    # Gaussian PSF
    sigma = 1.5  # pixels
    radius = int(3 * sigma)
    for yi, xi, intensity in zip(y_pos, x_pos, intensities):
        y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
        x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        psf = np.exp(-((yy - yi)**2 + (xx - xi)**2) / (2 * sigma**2))
        arr[y0:y1, x0:x1] += intensity * psf.astype(np.float32)

    # Add a few structured clusters (filaments)
    for _ in range(5):
        t = np.linspace(0, 1, 50)
        cx = rng.randint(30, W - 30)
        cy = rng.randint(30, H - 30)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(20, 60)
        for ti in t:
            xi = int(cx + length * ti * np.cos(angle) + rng.randn() * 1.5)
            yi = int(cy + length * ti * np.sin(angle) + rng.randn() * 1.5)
            if 0 <= xi < W and 0 <= yi < H:
                y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
                x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                psf = np.exp(-((yy - yi)**2 + (xx - xi)**2) / (2 * sigma**2))
                arr[y0:y1, x0:x1] += 0.8 * psf.astype(np.float32)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated medical phantom (ultrasound, DOT, photoacoustic, etc.)
# ---------------------------------------------------------------------------

def generate_medical_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a Shepp-Logan variant with tissue contrast.

    Used for ultrasound, photoacoustic, DOT, endoscopy, and other
    medical modalities that lack dedicated public datasets.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    arr = np.zeros((H, W), dtype=np.float32)

    # Body outline
    arr[(X / 0.85)**2 + (Y / 0.95)**2 < 1] = 0.15
    # Organs (varying contrast)
    arr[((X - 0.2) / 0.25)**2 + ((Y + 0.1) / 0.35)**2 < 1] = 0.6
    arr[((X + 0.25) / 0.20)**2 + ((Y + 0.05) / 0.30)**2 < 1] = 0.45
    arr[((X + 0.05) / 0.15)**2 + ((Y - 0.35) / 0.20)**2 < 1] = 0.7
    # Vessels
    for _ in range(8):
        cx, cy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        r = rng.uniform(0.02, 0.05)
        arr[(X - cx)**2 + (Y - cy)**2 < r**2] = rng.uniform(0.3, 0.9)
    # Speckle texture (characteristic of ultrasound)
    speckle = rng.rayleigh(0.3, (H, W)).astype(np.float32)
    arr = arr * (0.7 + 0.3 * speckle / speckle.max())

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated EM phantom (nanoparticle field)
# ---------------------------------------------------------------------------

def generate_em_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic electron microscopy-like image.

    Nanoparticles on amorphous carbon with varying contrast.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Amorphous carbon background
    arr = 0.15 + 0.05 * rng.randn(H, W).astype(np.float32)

    # Nanoparticles of various sizes
    for _ in range(80):
        cx, cy = rng.randint(5, W - 5), rng.randint(5, H - 5)
        r = rng.uniform(2, 8)
        intensity = rng.uniform(0.5, 1.0)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        particle = np.exp(-dist**2 / (2 * (r / 2.5)**2))
        arr += (intensity * particle).astype(np.float32)

    # Lattice fringes in a few particles
    for _ in range(5):
        cx, cy = rng.randint(20, W - 20), rng.randint(20, H - 20)
        r = rng.uniform(5, 12)
        freq = rng.uniform(0.5, 1.5)
        angle = rng.uniform(0, np.pi)
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        mask = dist < r
        fringes = 0.15 * np.cos(2 * np.pi * freq * (
            (np.arange(W) - cx) * np.cos(angle) +
            (np.arange(H)[:, None] - cy) * np.sin(angle)
        ))
        arr[mask] += fringes[mask].astype(np.float32)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated depth map
# ---------------------------------------------------------------------------

def generate_depth_map(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic depth map of a simple room scene."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Background wall at far depth
    arr = np.full((H, W), 0.8, dtype=np.float32)

    # Floor gradient (closer at bottom)
    floor_start = H // 2
    for row in range(floor_start, H):
        depth = 0.8 - 0.6 * (row - floor_start) / (H - floor_start)
        arr[row, :] = depth

    # Place rectangular objects at various depths
    for _ in range(6):
        x0 = rng.randint(0, W - 40)
        y0 = rng.randint(0, H - 40)
        w = rng.randint(20, 60)
        h = rng.randint(20, 60)
        depth = rng.uniform(0.1, 0.6)
        arr[y0:min(y0 + h, H), x0:min(x0 + w, W)] = depth

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated test scene (edges, gradients, textures)
# ---------------------------------------------------------------------------

def generate_test_scene(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a high-contrast test scene with varied features.

    Used for computational photography, event cameras, HDR, etc.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Checkerboard quadrant
    qh, qw = H // 2, W // 2
    block = 16
    for i in range(qh // block):
        for j in range(qw // block):
            if (i + j) % 2 == 0:
                arr[i * block:(i + 1) * block, j * block:(j + 1) * block] = 0.9

    # Gradient quadrant
    arr[:qh, qw:] = np.linspace(0, 1, W - qw)[np.newaxis, :]

    # Sinusoidal quadrant (varying frequency)
    x = np.linspace(0, 1, W - qw)
    y = np.linspace(0, 1, H - qh)
    X, Y = np.meshgrid(x, y)
    freq = 5 + 30 * X
    arr[qh:, qw:] = 0.5 + 0.5 * np.sin(2 * np.pi * freq * Y)

    # Random shapes quadrant
    for _ in range(15):
        cx = rng.randint(0, qw)
        cy = rng.randint(qh, H)
        r = rng.randint(5, 25)
        intensity = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        arr[mask] = intensity

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated star field (astronomy)
# ---------------------------------------------------------------------------

def generate_star_field(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic astronomical field with point sources and nebula."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Background sky noise
    arr += 0.02 * rng.randn(H, W).astype(np.float32)

    # Point stars with Airy-like PSF
    n_stars = 50
    for _ in range(n_stars):
        cx, cy = rng.randint(5, W - 5), rng.randint(5, H - 5)
        mag = rng.uniform(0.1, 1.0)  # brightness
        sigma = rng.uniform(0.8, 2.0)  # PSF width
        radius = int(4 * sigma)
        y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        psf = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))
        arr[y0:y1, x0:x1] += mag * psf.astype(np.float32)

    # Extended source (nebula/galaxy)
    cx, cy = W // 2, H // 2
    sigma_x, sigma_y = W // 6, H // 8
    yy, xx = np.mgrid[:H, :W]
    nebula = 0.3 * np.exp(-(
        (xx - cx)**2 / (2 * sigma_x**2) +
        (yy - cy)**2 / (2 * sigma_y**2)
    ))
    arr += nebula.astype(np.float32)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated resolution target with phase (coherent imaging)
# ---------------------------------------------------------------------------

def generate_resolution_target(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate USAF-like resolution target for coherent imaging."""
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.ones((H, W), dtype=np.float32) * 0.1

    # Groups of bar patterns at increasing frequencies
    y_offset = 10
    for group in range(6):
        freq = 4 + group * 3  # increasing line density
        bar_height = max(3, H // (12 + group * 2))
        for element in range(3):  # 3 elements per group
            x_start = 10 + group * (W // 7)
            y_start = y_offset
            # Horizontal bars
            for bar in range(3):
                bar_width = max(1, W // (7 * freq))
                y0 = y_start + bar * 2 * bar_width
                if y0 + bar_width > H:
                    break
                arr[y0:y0 + bar_width, x_start:x_start + bar_height] = 0.9
            y_offset += bar_height + 5

        if y_offset > H - 20:
            break

    # Central circle
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - W // 2)**2 + (yy - H // 2)**2 < (min(H, W) // 8)**2
    arr[mask] = 0.7

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated diffraction pattern
# ---------------------------------------------------------------------------

def generate_diffraction_pattern(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate simulated X-ray/neutron diffraction pattern."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    yy, xx = np.mgrid[:H, :W]
    cx, cy = W // 2, H // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)

    arr = np.zeros((H, W), dtype=np.float32)

    # Debye-Scherrer rings
    ring_radii = [20, 35, 50, 65, 85, 100, 120]
    for radius in ring_radii:
        if radius > min(H, W) // 2:
            break
        ring_width = rng.uniform(1.5, 3.0)
        intensity = rng.uniform(0.3, 0.8)
        arr += intensity * np.exp(-(r - radius)**2 / (2 * ring_width**2))

    # Bragg peaks on rings
    for radius in ring_radii[:4]:
        n_peaks = rng.randint(4, 8)
        for _ in range(n_peaks):
            angle = rng.uniform(0, 2 * np.pi)
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            spot = np.exp(-((xx - px)**2 + (yy - py)**2) / 8.0)
            arr += rng.uniform(0.5, 1.0) * spot.astype(np.float32)

    # Central beam stop
    arr[r < 8] = 0.0

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated elemental map (XRF, MALDI)
# ---------------------------------------------------------------------------

def generate_elemental_map(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic elemental distribution map."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Matrix background
    arr += 0.1

    # Elemental domains (different concentrations)
    n_domains = 12
    for _ in range(n_domains):
        cx, cy = rng.randint(20, W - 20), rng.randint(20, H - 20)
        rx, ry = rng.randint(15, 50), rng.randint(15, 50)
        concentration = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:H, :W]
        mask = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2 < 1
        arr[mask] = concentration

    # Grain boundaries (thin lines of different composition)
    for _ in range(5):
        x0, x1 = rng.randint(0, W, 2)
        y0, y1 = rng.randint(0, H, 2)
        n_pts = max(abs(x1 - x0), abs(y1 - y0))
        if n_pts == 0:
            continue
        xs = np.linspace(x0, x1, n_pts).astype(int)
        ys = np.linspace(y0, y1, n_pts).astype(int)
        valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        arr[ys[valid], xs[valid]] = rng.uniform(0.5, 0.9)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated Acoustic Emission (AE) source energy map
# ---------------------------------------------------------------------------

def generate_ae_source_map(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate acoustic emission source energy map.

    Models the 2-D distribution of acoustic energy released during crack
    initiation and propagation in a structural component.  The ground truth
    is a *source intensity map* — not a defect geometry map — because the
    AE inverse problem is to localise *where energy is being released*, given
    multi-sensor time-domain waveforms.

    Physics basis
    -------------
    Crack-tip AE events are impulsive point sources (high local energy release);
    delamination fronts and fibre-breakage produce line/arc sources; background
    dislocation activity contributes a low-level diffuse field.  Source
    amplitudes follow a power-law magnitude distribution (Gutenberg-Richter
    analogue), consistent with real AE data from steel, concrete and CFRP
    structures (Grosse & Ohtsu, 2008).

    References
    ----------
    Grosse, C.U. & Ohtsu, M. (2008). *Acoustic Emission Testing*. Springer.
    Ebrahimkhanlou & Salamone (2019). *Structural Health Monitoring*, 18(2):636-651.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # --- Point-source AE events (crack initiation hits) ---
    # Power-law amplitude distribution: small events are frequent, large rare
    n_events = rng.randint(8, 25)
    for _ in range(n_events):
        cx = rng.randint(5, W - 5)
        cy = rng.randint(5, H - 5)
        amplitude = rng.power(0.4)  # power-law: many weak, few strong events
        sigma = rng.uniform(1.5, 4.0)  # Gaussian spread per event
        yy, xx = np.ogrid[:H, :W]
        arr += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    # --- Line sources (crack propagation fronts) ---
    n_cracks = rng.randint(1, 4)
    for _ in range(n_cracks):
        x0 = rng.randint(10, W - 10)
        y0 = rng.randint(10, H - 10)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(20, 60)
        amp_crack = rng.uniform(0.3, 0.8)
        for t in range(length):
            xi = int(x0 + t * np.cos(angle))
            yi = int(y0 + t * np.sin(angle))
            if 0 <= xi < W and 0 <= yi < H:
                # Each point along the crack is a weak Gaussian source
                sigma_c = rng.uniform(1.0, 2.5)
                yy, xx = np.ogrid[:H, :W]
                arr += amp_crack * 0.15 * np.exp(
                    -((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma_c ** 2)
                )

    # --- Low-level diffuse background (dislocation activity) ---
    bg = rng.uniform(0.0, 0.05, (H, W)).astype(np.float32)
    # Smooth to remove high-frequency noise from the background
    from scipy.ndimage import gaussian_filter
    bg = gaussian_filter(bg, sigma=3.0)
    arr += bg

    return normalize_array(arr.clip(0, None))


# ---------------------------------------------------------------------------
# Generated NDT phantom (material with defects)
# ---------------------------------------------------------------------------

def generate_ndt_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate phantom with embedded defects for NDT inspection."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Uniform material
    arr = np.full((H, W), 0.5, dtype=np.float32)

    # Circular voids
    for _ in range(5):
        cx, cy = rng.randint(20, W - 20), rng.randint(20, H - 20)
        r = rng.uniform(3, 10)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        arr[mask] = 0.05

    # Cracks (thin lines)
    for _ in range(3):
        x0 = rng.randint(0, W)
        y0 = rng.randint(0, H)
        angle = rng.uniform(0, np.pi)
        length = rng.randint(20, 80)
        for t in range(length):
            xi = int(x0 + t * np.cos(angle))
            yi = int(y0 + t * np.sin(angle))
            if 0 <= xi < W and 0 <= yi < H:
                arr[yi, max(0, xi - 1):min(W, xi + 2)] = 0.1

    # Inclusions (different material)
    for _ in range(4):
        cx, cy = rng.randint(15, W - 15), rng.randint(15, H - 15)
        r = rng.uniform(4, 12)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        arr[mask] = rng.uniform(0.7, 0.95)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated velocity model (seismic)
# ---------------------------------------------------------------------------

def generate_velocity_model(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate layered seismic velocity model."""
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # Horizontal layers with gentle undulation
    n_layers = 8
    boundaries = sorted(rng.randint(10, H - 10, n_layers - 1))
    boundaries = [0] + list(boundaries) + [H]
    velocities = np.linspace(0.2, 0.9, n_layers)

    x_coord = np.linspace(0, 2 * np.pi, W)
    for i in range(n_layers):
        top = boundaries[i]
        bot = boundaries[i + 1]
        # Add undulation
        undulation = (5 * np.sin(x_coord + rng.uniform(0, np.pi))).astype(int)
        for col in range(W):
            t = max(0, top + undulation[col])
            b = min(H, bot + undulation[col])
            arr[t:b, col] = velocities[i]

    # Fault (vertical displacement)
    fault_x = W // 2 + rng.randint(-20, 20)
    shift = rng.randint(10, 30)
    arr[:, fault_x:] = np.roll(arr[:, fault_x:], shift, axis=0)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# High-level: download + convert a registry entry
# ---------------------------------------------------------------------------

def acquire_dataset(
    entry,  # DatasetEntry from registry
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[Path, np.ndarray]:
    """Download and convert a single ``DatasetEntry``.

    Returns ``(npy_path, array)`` where *npy_path* is the cached ``.npy`` file.
    """
    cache_dir = cache_dir or CACHE_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path = cache_dir / f"{entry.id}.npy"

    # Check cache
    if npy_path.exists() and not force:
        logger.info("Using cached: %s", npy_path)
        arr = np.load(npy_path)
        return npy_path, arr

    target_shape = tuple(entry.x_shape) if entry.x_shape else None

    # Handle generated datasets (no download needed)
    _generated_converters = {
        "generate_surface": lambda: generate_surface(target_shape=target_shape),
        "generate_oct_phantom": lambda: generate_oct_phantom(target_shape=target_shape),
        "generate_smlm_phantom": lambda: generate_smlm_phantom(target_shape=target_shape),
        "generate_medical_phantom": lambda: generate_medical_phantom(target_shape=target_shape),
        "generate_em_phantom": lambda: generate_em_phantom(target_shape=target_shape),
        "generate_depth_map": lambda: generate_depth_map(target_shape=target_shape),
        "generate_test_scene": lambda: generate_test_scene(target_shape=target_shape),
        "generate_star_field": lambda: generate_star_field(target_shape=target_shape),
        "generate_resolution_target": lambda: generate_resolution_target(target_shape=target_shape),
        "generate_diffraction_pattern": lambda: generate_diffraction_pattern(target_shape=target_shape),
        "generate_elemental_map": lambda: generate_elemental_map(target_shape=target_shape),
        "generate_ndt_phantom": lambda: generate_ndt_phantom(target_shape=target_shape),
        "generate_velocity_model": lambda: generate_velocity_model(target_shape=target_shape),
        "generate_ae_source_map": lambda: generate_ae_source_map(target_shape=target_shape),
    }
    gen_fn = _generated_converters.get(entry.converter)
    if gen_fn is not None:
        arr = gen_fn()
        np.save(npy_path, arr)
        return npy_path, arr

    if entry.converter == "convert_brainweb" and not entry.url:
        arr = convert_brainweb(Path("/dev/null"), target_shape=target_shape)
        np.save(npy_path, arr)
        return npy_path, arr

    # Download
    if not entry.url:
        raise ValueError(f"No URL for dataset {entry.id}")

    url_hash = hashlib.sha256(entry.url.encode()).hexdigest()[:16]
    suffix_map = {
        "mat": ".mat", "mat_v73": ".mat", "hdf5": ".hdf5", "tiff": ".tiff",
        "npy": ".npy", "nifti": ".nii.gz", "zip": ".zip",
        "png": ".zip", "mrc": ".mrc", "raw": ".raw",
    }
    ext = suffix_map.get(entry.format, "")
    raw_path = cache_dir / f"{entry.id}_{url_hash}{ext}"

    download_file(entry.url, raw_path)

    # Convert based on format
    converter_map = {
        "convert_mat": lambda: convert_mat(raw_path, key=entry.mat_key, target_shape=target_shape),
        "convert_mat_v73": lambda: convert_mat_v73(raw_path, key=entry.mat_key, target_shape=target_shape),
        "convert_hdf5": lambda: convert_hdf5(raw_path, key=entry.mat_key, target_shape=target_shape),
        "convert_tiff": lambda: convert_tiff(raw_path, target_shape=target_shape),
        "convert_nifti": lambda: convert_nifti(raw_path, target_shape=target_shape),
        "convert_nifti_from_zip": lambda: convert_nifti_from_zip(raw_path, target_shape=target_shape),
        "convert_mrc": lambda: convert_mrc(raw_path, target_shape=target_shape),
        "convert_brainweb": lambda: convert_brainweb(raw_path, target_shape=target_shape),
        "convert_lidar_bin": lambda: convert_lidar_bin(raw_path, target_shape=target_shape),
        "convert_png_stack": lambda: _convert_png_from_archive(raw_path, target_shape),
        "generate_surface": lambda: generate_surface(target_shape=target_shape),
        "generate_oct_phantom": lambda: generate_oct_phantom(target_shape=target_shape),
        "generate_smlm_phantom": lambda: generate_smlm_phantom(target_shape=target_shape),
        "generate_medical_phantom": lambda: generate_medical_phantom(target_shape=target_shape),
        "generate_em_phantom": lambda: generate_em_phantom(target_shape=target_shape),
        "generate_depth_map": lambda: generate_depth_map(target_shape=target_shape),
        "generate_test_scene": lambda: generate_test_scene(target_shape=target_shape),
        "generate_star_field": lambda: generate_star_field(target_shape=target_shape),
        "generate_resolution_target": lambda: generate_resolution_target(target_shape=target_shape),
        "generate_diffraction_pattern": lambda: generate_diffraction_pattern(target_shape=target_shape),
        "generate_elemental_map": lambda: generate_elemental_map(target_shape=target_shape),
        "generate_ndt_phantom": lambda: generate_ndt_phantom(target_shape=target_shape),
        "generate_velocity_model": lambda: generate_velocity_model(target_shape=target_shape),
    }

    convert_fn = converter_map.get(entry.converter)
    if convert_fn is None:
        raise ValueError(f"Unknown converter: {entry.converter}")

    arr = convert_fn()
    np.save(npy_path, arr)
    logger.info("Saved converted: %s  shape=%s", npy_path.name, arr.shape)
    return npy_path, arr


def _convert_png_from_archive(
    archive_path: Path,
    target_shape: Optional[Tuple[int, ...]],
) -> np.ndarray:
    """Extract a ZIP and load the first image as a PNG stack."""
    extract_dir = extract_zip(archive_path)
    return convert_png_stack(extract_dir, n_images=1, target_shape=target_shape)
