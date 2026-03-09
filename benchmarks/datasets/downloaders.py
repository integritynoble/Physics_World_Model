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
# Generated Scanning Acoustic Microscopy (SAM) C-scan phantom
# ---------------------------------------------------------------------------

def generate_sam_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic SAM C-scan reflectivity map.

    Models a 2-D acoustic reflectivity map (C-scan slice at a fixed depth)
    of a layered electronic or composite structure.  Features are calibrated
    to match the appearance of real SAM images from microelectronic packages
    and CFRP laminate specimens.

    Physics basis
    -------------
    The reflectivity R at an interface is given by the acoustic impedance
    mismatch:  R = (Z2 - Z1)/(Z2 + Z1), where Z = rho * c_s.  Voids and
    delaminations (Z2 ≈ 0) give R ≈ -1; inclusions with Z2 > Z1 give
    positive R.  The benchmark ground truth is the 2-D map R(x,y).

    Features generated
    ------------------
    - Uniform background (bulk material, Z ≈ Z1, R ≈ 0)
    - Delamination regions (elliptical, low reflectivity R ≈ -0.8 to -0.5)
    - Voids (small circular, very low R ≈ -0.95)
    - Inclusions / wire bonds (small bright spots, R ≈ +0.4 to +0.7)
    - Die-attach boundary (rectilinear bright-edge gradient)

    References
    ----------
    Guo, S. et al. (2022). Acoustic microscopy for electronic package inspection.
    Ultrasonics, 122, 106679.
    Rigby et al. (2023). Deep learning for SAM defect detection. NDT&E Int. 138.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)

    # --- Die boundary (rectilinear bright rectangle - die-attach perimeter) ---
    margin_x = rng.randint(15, 40)
    margin_y = rng.randint(15, 40)
    bw = rng.randint(2, 5)
    arr[margin_y:margin_y + bw, margin_x:W - margin_x] = rng.uniform(0.4, 0.6)
    arr[H - margin_y - bw:H - margin_y, margin_x:W - margin_x] = rng.uniform(0.4, 0.6)
    arr[margin_y:H - margin_y, margin_x:margin_x + bw] = rng.uniform(0.4, 0.6)
    arr[margin_y:H - margin_y, W - margin_x - bw:W - margin_x] = rng.uniform(0.4, 0.6)

    # --- Delamination regions (elliptical, low reflectivity) ---
    yy, xx = np.ogrid[:H, :W]
    n_delam = rng.randint(1, 4)
    for _ in range(n_delam):
        cx = rng.randint(margin_x + 10, W - margin_x - 10)
        cy = rng.randint(margin_y + 10, H - margin_y - 10)
        rx = rng.uniform(15, 45)
        ry = rng.uniform(10, 35)
        angle = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = (xx - cx) * cos_a + (yy - cy) * sin_a
        dy = -(xx - cx) * sin_a + (yy - cy) * cos_a
        mask = (dx / rx) ** 2 + (dy / ry) ** 2 < 1.0
        arr[mask] = rng.uniform(-0.80, -0.50)

    # --- Voids (small circular, very negative) ---
    n_voids = rng.randint(2, 8)
    for _ in range(n_voids):
        cx = rng.randint(margin_x + 5, W - margin_x - 5)
        cy = rng.randint(margin_y + 5, H - margin_y - 5)
        r = rng.uniform(3, 8)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        arr[mask] = rng.uniform(-0.95, -0.80)

    # --- Inclusions / wire-bond pads (small bright spots) ---
    n_incl = rng.randint(3, 12)
    for _ in range(n_incl):
        cx = rng.randint(margin_x + 3, W - margin_x - 3)
        cy = rng.randint(margin_y + 3, H - margin_y - 3)
        r = rng.uniform(2, 6)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        arr[mask] = rng.uniform(0.40, 0.70)

    # Slight smoothing (acoustic PSF blurring at fabrication level)
    arr = gaussian_filter(arr, sigma=0.8)

    # Normalise to [0, 1] for downstream benchmark framework
    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated Active/Pulsed Thermography phantom (thermal diffusivity map)
# ---------------------------------------------------------------------------

def generate_thermography_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate thermal diffusivity map for pulsed thermography NDT.

    Models the 2-D thermal diffusivity distribution of a composite panel
    containing subsurface defects (delaminations, voids).  The ground truth
    is a defect contrast map where uniform material background (high thermal
    diffusivity) is interrupted by flat circular low-diffusivity regions
    representing subsurface air/resin-rich pockets.

    Physics basis
    -------------
    Pulsed thermography: a short heat pulse (flash lamp, ~1 ms) heats the
    surface; heat diffuses as G_D(x,y,t) = (4πDt)^{-1}exp(-(r²)/(4Dt)).
    A defect at depth d has a characteristic blind-time t_blind ≈ d²/(πD)
    after which it first becomes detectable; shallow defects appear early and
    dark (strong thermal contrast), deep defects appear later and lighter
    (reduced contrast).  Defect intensity is encoded as:
      arr_defect ≈ 0.1 + 0.1 * (depth_fraction)    (shallow=dark, deep=lighter)

    Features
    --------
    - Uniform background: arr = 0.5 (bulk material thermal diffusivity)
    - 3-6 circular defects of varying radius (r=8-30 px) and depth
    - Defect depth encoded as intensity: shallow → ~0.10, deep → ~0.20
    - Slight Gaussian smoothing (sigma=1.5) to simulate lateral thermal diffusion

    References
    ----------
    Maldague, X.P.V. (2001). *Theory and Practice of Infrared Technology for
    Nondestructive Testing*. Wiley.
    Shepard, S.M. et al. (2003). Reconstruction and enhancement of active
    thermographic image sequences. Opt. Eng. 42(5):1337-1342.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Uniform material background
    arr = np.full((H, W), 0.5, dtype=np.float32)

    yy, xx = np.ogrid[:H, :W]
    n_defects = rng.randint(3, 7)
    for _ in range(n_defects):
        cx = rng.randint(20, W - 20)
        cy = rng.randint(20, H - 20)
        r = rng.uniform(8, 30)
        # depth_fraction: 0=shallow (very dark), 1=deep (lighter)
        depth_fraction = rng.uniform(0.0, 1.0)
        defect_val = 0.10 + 0.10 * depth_fraction  # range [0.10, 0.20]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        arr[mask] = defect_val

    # Slight smoothing to simulate thermal diffusion blurring
    arr = gaussian_filter(arr, sigma=1.5)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated Adaptive Optics wavefront (Kolmogorov turbulence, Zernike modes)
# ---------------------------------------------------------------------------

def generate_ao_wavefront(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate turbulent wavefront phase map for Hartmann-Shack wavefront sensing.

    Constructs a wavefront phase map on a circular pupil by summing the first
    20 Zernike polynomials (Noll ordering, modes 2–21) with amplitudes drawn
    from the Kolmogorov turbulence power spectrum (amplitude ~ j^(-11/6)).
    Tip and tilt (modes 2, 3) dominate; higher modes decrease in variance.

    Physics basis
    -------------
    Kolmogorov atmospheric turbulence produces a power spectrum for the Zernike
    coefficients: Var(a_j) ∝ j^(-11/6) (Noll 1976).  The wavefront is:
        phi(rho, theta) = sum_{j=2}^{21} a_j * Z_j(rho, theta)
    where Z_j are the Noll-ordered Zernike polynomials on the unit disk.
    Zernike polynomials are computed analytically using radial/azimuthal indices.

    The resulting phase map represents the wavefront to be corrected by a
    deformable mirror in closed-loop AO.

    References
    ----------
    Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
    J. Opt. Soc. Am. 66(3):207-211.
    Fried, D.L. (1966). Optical resolution through a randomly inhomogeneous
    medium for very long and very short exposures. J. Opt. Soc. Am. 56(10):1372.
    """
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Build pupil grid (unit circle)
    y_coords = np.linspace(-1.0, 1.0, H)
    x_coords = np.linspace(-1.0, 1.0, W)
    xx, yy = np.meshgrid(x_coords, y_coords)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    pupil = (rho <= 1.0).astype(np.float32)

    # Noll ordering: map sequential index j (1-based) to (n, m)
    # j=1: piston, j=2: tip, j=3: tilt, j=4: defocus, ...
    def noll_to_nm(j):
        """Convert Noll index j to radial n and azimuthal m."""
        n = int(np.ceil((-3 + np.sqrt(9 + 8*(j-1))) / 2))
        j_start = n*(n+1)//2 + 1
        m_abs = (j - j_start) if (j - j_start) <= n else n - (j - j_start - n)
        m_abs = (j - j_start)
        # determine sign of m from Noll convention
        if (j - j_start) % 2 == 0:
            m = -m_abs // 2 * 2 if m_abs % 2 == 0 else 0
        else:
            m = (m_abs + 1) // 2 * 2 - 1 if m_abs % 2 != 0 else 0
        # Simplified: compute n and |m| directly
        n_val = 0
        j_count = 0
        while j_count + n_val + 1 < j:
            j_count += n_val + 1
            n_val += 1
        m_val = (j - j_count - 1)
        if n_val % 2 != m_val % 2:
            m_val -= 1
        return n_val, m_val

    def zernike_radial(n, m, rho):
        """Compute radial Zernike polynomial R_n^m(rho)."""
        import math as _math
        m_abs = abs(m)
        R = np.zeros_like(rho)
        for s in range((n - m_abs) // 2 + 1):
            coeff = ((-1) ** s * _math.factorial(n - s) /
                     (_math.factorial(s) *
                      _math.factorial((n + m_abs) // 2 - s) *
                      _math.factorial((n - m_abs) // 2 - s)))
            R += coeff * rho ** (n - 2 * s)
        return R

    # Compute wavefront as sum of Zernike modes j=2..21
    phi = np.zeros((H, W), dtype=np.float32)
    for j in range(2, 22):
        # Kolmogorov amplitude variance: ~ j^(-11/6)
        std_j = j ** (-11.0 / 12.0)  # amplitude std ~ sqrt(variance), variance ~ j^(-11/6)
        a_j = rng.normal(0.0, std_j)

        # Simplified Zernike evaluation using analytic radial/azimuthal structure
        n_val = int(np.ceil((-3 + np.sqrt(9 + 8*(j-1))) / 2))
        j_start = n_val * (n_val + 1) // 2 + 1
        idx = j - j_start  # 0-based index within this radial order

        # Azimuthal frequency m (simplified Noll ordering approximation)
        if n_val == 0:
            m_val = 0
        else:
            m_candidates = list(range(n_val % 2, n_val + 1, 2))
            if idx < len(m_candidates):
                m_val = m_candidates[idx]
            else:
                m_val = m_candidates[-1]

        # Radial polynomial
        R = zernike_radial(n_val, m_val, rho)

        # Azimuthal part — alternate sin/cos for ±m (Noll convention)
        if m_val == 0:
            Z = R
        elif idx % 2 == 0:
            Z = np.sqrt(2) * R * np.cos(m_val * theta)
        else:
            Z = np.sqrt(2) * R * np.sin(m_val * theta)

        phi += a_j * Z * pupil

    # Zero outside pupil
    phi *= pupil
    return normalize_array(phi)


# ---------------------------------------------------------------------------
# Generated AFM surface topography phantom
# ---------------------------------------------------------------------------

def generate_afm_surface(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate atomic force microscopy surface topography map.

    Produces one of three surface types, selected randomly:
    - Crystalline: periodic lattice rows (square lattice, sin+cos patterns)
    - Amorphous: layered rough surface using Gaussian blobs + random bumps
    - Biological: cell-like rounded features (3-6 bumps of varying size)

    All types have Gaussian measurement noise (sigma=0.02) added to simulate
    AFM detector noise and thermal drift fluctuations.

    Physics basis
    -------------
    In tapping-mode AFM the measured height image y ≈ s ⊕ t (morphological
    dilation of true surface s with tip shape t).  The ground truth x = s is
    the true surface before tip convolution.  Surface types span the range of
    real AFM specimens: crystalline samples (HOPG, protein 2D crystals),
    amorphous layers (polymer films, oxide glasses), and biological samples
    (cell membranes, DNA).

    References
    ----------
    Nečas, D. & Klapetek, P. (2012). Gwyddion: An open-source software for
    SPM data analysis. Open Physics 10(1):181-188.
    Jalili, N. & Laxminarayana, K. (2004). A review of atomic force microscopy
    imaging systems: application to molecular metrology and biological sciences.
    Mechatronics 14(8):907-945.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    surface_type = rng.choice(['crystalline', 'amorphous', 'biological'])
    arr = np.zeros((H, W), dtype=np.float32)

    yy_idx, xx_idx = np.mgrid[:H, :W]

    if surface_type == 'crystalline':
        # Regular square lattice bumps (molecular rows)
        lattice_const = rng.randint(8, 21)  # pixels
        amplitude = rng.uniform(0.3, 0.6)
        angle = rng.uniform(0, np.pi / 4)  # slight rotation
        x_rot = xx_idx * np.cos(angle) + yy_idx * np.sin(angle)
        y_rot = -xx_idx * np.sin(angle) + yy_idx * np.cos(angle)
        arr = amplitude * (
            np.sin(2 * np.pi * x_rot / lattice_const) *
            np.cos(2 * np.pi * y_rot / lattice_const) + 1.0
        ) / 2.0
        # Add slight surface roughness
        arr += rng.normal(0.0, 0.03, (H, W)).astype(np.float32)
        arr = gaussian_filter(arr, sigma=0.5)

    elif surface_type == 'amorphous':
        # Layered rough surface: multiple Gaussian blobs + random bumps
        n_blobs = rng.randint(8, 20)
        for _ in range(n_blobs):
            cx = rng.randint(0, W)
            cy = rng.randint(0, H)
            sigma_blob = rng.uniform(10, 40)
            amplitude = rng.uniform(0.2, 0.7)
            arr += amplitude * np.exp(
                -((xx_idx - cx) ** 2 + (yy_idx - cy) ** 2) / (2 * sigma_blob ** 2)
            ).astype(np.float32)
        # Add fine-scale roughness
        roughness = rng.normal(0.0, 0.05, (H, W)).astype(np.float32)
        roughness = gaussian_filter(roughness, sigma=2.0)
        arr += roughness

    else:  # biological
        # Cell-like features: 3-6 rounded bumps of varying size
        n_cells = rng.randint(3, 7)
        for _ in range(n_cells):
            cx = rng.randint(30, W - 30)
            cy = rng.randint(30, H - 30)
            r = rng.uniform(20, 60)
            height = rng.uniform(0.4, 0.9)
            # Smooth rounded bump (Gaussian approximation of cell body)
            sigma_cell = r / 2.5
            arr += height * np.exp(
                -((xx_idx - cx) ** 2 + (yy_idx - cy) ** 2) / (2 * sigma_cell ** 2)
            ).astype(np.float32)
        # Fine sub-cellular roughness
        fine = rng.normal(0.0, 0.04, (H, W)).astype(np.float32)
        fine = gaussian_filter(fine, sigma=1.5)
        arr += fine

    # AFM measurement noise (thermal + shot)
    arr += rng.normal(0.0, 0.02, (H, W)).astype(np.float32)

    return normalize_array(arr)


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
# Generated angiography vessel phantom (DSA / 3DRA iodine map)
# ---------------------------------------------------------------------------

def generate_angiography_vessel_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2-D vessel iodine attenuation map for X-ray angiography.

    Models the digital subtraction angiography (DSA) ground truth: the
    differential iodine concentration map of a vascular tree after contrast
    injection, as seen on a single detector plane (2D) or on the reference
    slice of a 3D rotational angiography (3DRA) volume.

    Physics basis
    -------------
    In DSA, the subtraction image approximates:
        y_DSA ≈ mu_iodine * t_vessel * (concentration / c_ref)
    where mu_iodine is the mass-attenuation coefficient of iodine at the
    X-ray energy (≈ 33 cm²/g at 33 keV K-edge, ≈ 5 cm²/g at 80 kVp), and
    t_vessel is the vessel lumen diameter integrated along the beam.

    The ground truth x is a vessel density map in [0, 1] where:
      - 0 = background (no iodine / soft tissue)
      - 1 = peak iodine concentration in main vessel

    Vascular tree structure
    -----------------------
    A fractal-like branching tree is generated by:
      1. Central trunk: near-vertical main vessel (aorta/ICA) with Gaussian
         cross-section (radius 8-15 px) and mild tortuosity.
      2. 2-4 first-order branches: smaller vessels (r=4-8 px) departing from
         random points along the trunk at angles 30-60 degrees, with gradual
         taper (Murray's law: r^3 = const).
      3. 2-4 second-order branches per first-order branch: fine vessels
         (r=2-5 px) continuing the bifurcation tree.
    Contrast decreases with vessel order to model physiological perfusion
    (iodine dilution through capillary wash-out).

    References
    ----------
    Feldkamp, L.A. et al. (1984). Practical cone-beam algorithm.
    J. Opt. Soc. Am. A 1(6):612-619.
    Shen, C. et al. (2024). Geometry-aware diffusion model for few-view
    angiography reconstruction. Med. Image Anal. 94:103102.
    Wang, Z. et al. (2024). Motion-compensated angiography reconstruction
    with implicit neural representation. IEEE Trans. Med. Imaging 43:1401.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 256
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    arr = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[:H, :W]

    def _draw_vessel(arr, cx0, cy0, cx1, cy1, radius, amplitude):
        """Draw a straight vessel segment as a Gaussian-profile tube."""
        # Parametric: segment from (cx0,cy0) to (cx1,cy1)
        dx, dy = cx1 - cx0, cy1 - cy0
        length = max(np.sqrt(dx**2 + dy**2), 1.0)
        # Unit tangent
        tx, ty = dx / length, dy / length
        # Normal distance from each pixel to the segment
        # Project (xx-cx0, yy-cy0) onto tangent
        t_param = np.clip(
            (xx - cx0) * tx + (yy - cy0) * ty, 0, length
        )
        # Closest point on segment
        cx_close = cx0 + t_param * tx
        cy_close = cy0 + t_param * ty
        dist2 = (xx - cx_close)**2 + (yy - cy_close)**2
        sigma = max(radius / 2.5, 0.5)
        arr += amplitude * np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)

    # ── Main trunk (aorta / ICA) ──────────────────────────────────────────
    trunk_r = rng.uniform(8, 15)
    # Mildly tortuous: 3-5 control points
    n_ctrl = rng.randint(3, 6)
    ctrl_x = np.linspace(W * 0.45, W * 0.55, n_ctrl) + rng.uniform(-8, 8, n_ctrl)
    ctrl_y = np.linspace(H * 0.05, H * 0.95, n_ctrl)
    for i in range(n_ctrl - 1):
        _draw_vessel(arr, ctrl_x[i], ctrl_y[i], ctrl_x[i+1], ctrl_y[i+1],
                     trunk_r, amplitude=1.0)

    # ── First-order branches ──────────────────────────────────────────────
    n_branches1 = rng.randint(2, 5)
    branch1_ends = []
    for _ in range(n_branches1):
        # Branch origin: random point along trunk
        t_frac = rng.uniform(0.15, 0.85)
        seg_idx = int(t_frac * (n_ctrl - 1))
        seg_idx = min(seg_idx, n_ctrl - 2)
        t_seg = t_frac * (n_ctrl - 1) - seg_idx
        bx0 = ctrl_x[seg_idx] + t_seg * (ctrl_x[seg_idx+1] - ctrl_x[seg_idx])
        by0 = ctrl_y[seg_idx] + t_seg * (ctrl_y[seg_idx+1] - ctrl_y[seg_idx])
        # Murray's law: branch radius r1 = trunk_r / 2^(1/3)
        b1_r = trunk_r / (2 ** (1/3))
        # Random direction (avoid going straight down along trunk)
        angle = rng.uniform(np.pi / 6, 5 * np.pi / 6)
        side = rng.choice([-1, 1])
        length1 = rng.uniform(40, 90)
        bx1 = bx0 + side * length1 * np.cos(angle)
        by1 = by0 + length1 * np.sin(angle) * rng.uniform(0.3, 0.8)
        bx1 = np.clip(bx1, 5, W - 5)
        by1 = np.clip(by1, 5, H - 5)
        amplitude1 = rng.uniform(0.55, 0.80)
        _draw_vessel(arr, bx0, by0, bx1, by1, b1_r, amplitude1)
        branch1_ends.append((bx1, by1, b1_r, amplitude1))

    # ── Second-order branches (capillary-level) ───────────────────────────
    for bx0, by0, b1_r, amp1 in branch1_ends:
        n_branches2 = rng.randint(1, 4)
        for _ in range(n_branches2):
            b2_r = b1_r / (2 ** (1/3))
            angle2 = rng.uniform(np.pi / 5, 4 * np.pi / 5)
            side2 = rng.choice([-1, 1])
            length2 = rng.uniform(20, 55)
            bx2 = bx0 + side2 * length2 * np.cos(angle2)
            by2 = by0 + length2 * np.sin(angle2) * rng.uniform(0.3, 0.9)
            bx2 = np.clip(bx2, 5, W - 5)
            by2 = np.clip(by2, 5, H - 5)
            amplitude2 = amp1 * rng.uniform(0.40, 0.65)
            _draw_vessel(arr, bx0, by0, bx2, by2, b2_r, amplitude2)

    # ── Slight Gaussian smoothing (X-ray focal spot blur) ─────────────────
    arr = gaussian_filter(arr, sigma=0.8)

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated ASL perfusion phantom (cerebral blood flow map)
# ---------------------------------------------------------------------------

def generate_asl_perfusion_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2-D cerebral blood flow (CBF) map for ASL MRI.

    Models the ground truth perfusion-weighted image after label-control
    subtraction in Arterial Spin Labeling MRI.  The signal is proportional
    to CBF and reflects the regional perfusion values of different brain
    tissue compartments.

    Physics basis
    -------------
    The ASL subtraction signal is:
        DeltaM(x) = 2 * M0 * f(x) / lambda * alpha * T1_blood
                    * exp(-t_d / T1_blood)
    where f(x) is the local CBF map (mL/100g/min).  The ground truth x is
    normalised so that:
      - 0.0  = background / CSF (no perfusion)
      - 0.35 = white matter cortical perfusion (~25 mL/100g/min)
      - 0.70 = grey matter cortical perfusion  (~55 mL/100g/min)
      - 1.0  = peak perfusion in deep grey matter structures (basal ganglia,
               ~70-80 mL/100g/min)

    Brain compartments
    ------------------
    The phantom reproduces the major tissue compartments visible in a
    continuous-ASL or pseudo-continuous-ASL (pCASL) perfusion scan:
      1. Scalp + skull ring (zero perfusion).
      2. CSF / ventricles (zero perfusion, dark interior cavities).
      3. White matter (low perfusion, ~0.30-0.40 normalised).
      4. Grey matter cortical ribbon (~0.55-0.70 normalised).
      5. Deep grey matter: basal ganglia, thalamus (~0.85-1.0 normalised).
      6. Large-vessel territories: subtle MCA/ACA/PCA perfusion zoning
         modelled by smooth spatial gradient overlays.
      7. Random vascular noise: physiological CBF heterogeneity with
         Gaussian spatial texture superimposed on each compartment.

    Calibration
    -----------
    Normalised values calibrated to published pCASL perfusion maps:
      - Wu, W.C. et al. (2007) "A theoretical and experimental investigation
        of the tagging efficiency of pseudocontinuous arterial spin labeling."
        MRM 58(5):1020-1027.
      - Alsop, D.C. et al. (2015) "Recommended implementation of arterial
        spin-labeled perfusion MRI for clinical applications."
        MRM 73(1):102-116.
      - Guo, J. & Wong, E.C. (2012) "Increased SNR efficiency in velocity
        selective arterial spin labeling." MRM 68(4):1046-1055.

    References
    ----------
    Alsop, D.C. et al. (2015). Recommended implementation of ASL perfusion
    MRI for clinical applications. MRM 73(1):102-116.
    Mutsaerts, H.J.M.M. et al. (2020). ExploreASL: An image processing
    toolbox for population-level ASL perfusion MRI studies.
    NeuroImage 219:116932.
    Tian, Y. et al. (2023). Deep learning for ASL MRI reconstruction.
    MRM 89(4):1616-1629.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 128
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)
    R2 = X**2 + Y**2

    arr = np.zeros((H, W), dtype=np.float32)

    # ── 1. Outer brain oval (cortical surface) ────────────────────────────
    brain_mask = (X / 0.88)**2 + (Y / 0.92)**2 < 1.0

    # ── 2. White matter (inner oval, low perfusion) ───────────────────────
    wm_mask = (X / 0.65)**2 + (Y / 0.70)**2 < 1.0
    arr[brain_mask] = 0.35 + rng.uniform(-0.03, 0.03)

    # ── 3. Grey matter cortex = brain_mask AND NOT wm_mask ────────────────
    gm_mask = brain_mask & ~wm_mask
    gm_val = 0.60 + rng.uniform(0.0, 0.08)
    arr[gm_mask] = gm_val

    # Slight within-WM heterogeneity
    wm_texture = 0.28 + 0.10 * rng.randn(H, W).astype(np.float32)
    arr[wm_mask] = wm_texture[wm_mask]

    # ── 4. Deep grey matter: basal ganglia (2 ellipses) ───────────────────
    # Left putamen / globus pallidus
    bg_l = ((X + 0.22) / 0.10)**2 + ((Y + 0.05) / 0.14)**2 < 1.0
    arr[bg_l] = 0.88 + rng.uniform(0.0, 0.10)
    # Right putamen / globus pallidus
    bg_r = ((X - 0.22) / 0.10)**2 + ((Y + 0.05) / 0.14)**2 < 1.0
    arr[bg_r] = 0.88 + rng.uniform(0.0, 0.10)

    # ── 5. Thalami (two smaller ellipses, high perfusion) ─────────────────
    thal_l = ((X + 0.10) / 0.08)**2 + ((Y + 0.20) / 0.10)**2 < 1.0
    arr[thal_l] = 0.92 + rng.uniform(0.0, 0.08)
    thal_r = ((X - 0.10) / 0.08)**2 + ((Y + 0.20) / 0.10)**2 < 1.0
    arr[thal_r] = 0.92 + rng.uniform(0.0, 0.08)

    # ── 6. CSF / lateral ventricles (zero perfusion) ──────────────────────
    vent_l = ((X + 0.18) / 0.07)**2 + ((Y - 0.02) / 0.18)**2 < 1.0
    vent_r = ((X - 0.18) / 0.07)**2 + ((Y - 0.02) / 0.18)**2 < 1.0
    third_vent = (X / 0.025)**2 + ((Y - 0.05) / 0.08)**2 < 1.0
    arr[vent_l | vent_r | third_vent] = 0.0

    # Zero outside brain
    arr[~brain_mask] = 0.0

    # ── 7. Smooth vascular territory gradients ────────────────────────────
    # MCA territory (lateral): slight gradient boost
    mca_weight = np.exp(-((np.abs(X) - 0.5)**2) / (2 * 0.18**2))
    arr[brain_mask] += (0.06 * mca_weight[brain_mask] *
                        rng.uniform(0.8, 1.2, brain_mask.sum())).astype(np.float32)

    # ── 8. Physiological CBF heterogeneity texture ────────────────────────
    texture = gaussian_filter(rng.randn(H, W).astype(np.float32), sigma=3.0)
    texture /= np.std(texture) + 1e-6
    arr[brain_mask] += (0.04 * texture[brain_mask]).astype(np.float32)
    arr[~brain_mask] = 0.0

    # ── 9. Mild Gaussian smoothing (partial volume effects) ──────────────
    arr = gaussian_filter(arr, sigma=0.7)
    arr[~brain_mask] = 0.0

    return normalize_array(arr)


# ---------------------------------------------------------------------------
# Generated Atom Probe Tomography (APT) composition map
# ---------------------------------------------------------------------------

def generate_apt_composition_map(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2-D elemental composition map for Atom Probe Tomography (APT).

    Models the projected elemental concentration map that forms the ground
    truth in the APT inverse problem: reconstructing 3-D atomic-scale
    composition from a (x_det, y_det, t_flight) hit sequence recorded by a
    position-sensitive time-of-flight detector.

    Physics basis
    -------------
    In APT, atoms are field-evaporated one-by-one from a needle-shaped
    specimen (~50 nm tip radius) under a pulsed high-voltage (~3-15 kV).
    Each atom's (X_det, Y_det) impact position and flight time t encode its
    original 3-D lattice position and mass-to-charge ratio (m/z).  The
    spatial reconstruction uses the Bas protocol:

        x_atom = xi * X_det / (R_tip * Omega_f)
        z_atom = sum_i d_z / N_evap    [depth from evaporation order]

    where xi is the image compression factor (~0.6), R_tip is the tip radius
    (evolving during analysis), and Omega_f is the field factor (~3.3 for W).

    The 2-D ground truth image represents a single elemental species'
    concentration map (e.g. Cr, Ni, or Al) on a cross-sectional plane through
    a metallic alloy or semiconductor specimen.  Key microstructural features:

    Microstructural features (based on LEAP 5000 measurements)
    ----------------------------------------------------------
    1. Matrix phase — homogeneous solid solution (~0.25 normalised Cr at.%).
    2. Precipitate particles — gamma-prime (Ni3Al) or carbide precipitates with
       high solute concentration (0.7-1.0 normalised), 2-20 nm diameter.
       Calibrated to Hellman et al., Microsc. Microanal. 2000 (steel carbides).
    3. Grain boundaries — thin planar solute segregation bands (0.6-0.8),
       width ~1-2 voxels; 2-4 boundaries per field of view.
       Calibrated to Blavette et al., Science 1999 (grain boundary Cr enrichment).
    4. Dislocation loops — curved line features with partial solute enrichment
       (0.5-0.7); simulates pipe diffusion segregation.
    5. Detector noise — Poisson-like local intensity fluctuations (~5% rms)
       reflecting the ~60% detection efficiency of MCP detectors.
    6. Trajectory aberrations — local magnification artefacts at precipitate
       interfaces (low-frequency smooth distortion field) modelled as a
       multiplicative Gaussian envelope.

    Calibration
    -----------
    Normalised values calibrated to published LEAP atom probe datasets:
      - Hellman, O.C. et al. (2000). Analysis of nanoscale precipitates in a
        nickel-based superalloy using the atom probe. Microsc. Microanal.
        6(5):437-444.
      - Blavette, D. et al. (1999). Atomic-scale observation of grain boundary
        segregation in tungsten. Science 286(5448):2317-2319.
      - Thompson, K. et al. (2007). In situ site-specific specimen preparation
        for atom probe tomography. Ultramicroscopy 107(2-3):131-139.
      - Larson, D.J. et al. (2013). Local Electrode Atom Probe Tomography:
        A User's Guide. Springer. [Detection efficiency ~0.37-0.80]

    References
    ----------
    Bas, P. et al. (1995). A general protocol for the reconstruction of 3D
    atom probe data. Appl. Surf. Sci. 87-88:298-304.
    Miller, M.K. & Forbes, R.G. (2014). Atom-Probe Tomography. Springer.
    Gault, B. et al. (2012). Atom Probe Microscopy. Springer.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 128
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # ── 1. Matrix background (solid solution baseline) ────────────────────
    # Uniform solute concentration with Poisson counting noise (~60% efficiency)
    base_concentration = 0.25
    poisson_noise_sigma = 0.05
    arr = np.full((H, W), base_concentration, dtype=np.float32)
    arr += (rng.randn(H, W) * poisson_noise_sigma).astype(np.float32)

    # ── 2. Precipitate particles (gamma-prime, carbides) ──────────────────
    # ~8-18 precipitates; log-normal size distribution (2-20 nm range)
    n_precipitates = rng.randint(8, 18)
    for _ in range(n_precipitates):
        # Position: uniformly distributed, avoid edges
        cx = rng.randint(10, W - 10)
        cy = rng.randint(10, H - 10)
        # Radius: log-normal (mean ~5 px, sigma_log ~0.5) — mimics Ostwald ripening
        r = max(2, int(rng.lognormal(mean=np.log(5), sigma=0.5)))
        # Concentration: high solute enrichment (segregation ratio ~3-5x)
        c_precipitate = rng.uniform(0.70, 1.0)
        yy, xx = np.ogrid[:H, :W]
        dist2 = ((xx - cx) / max(r, 1))**2 + ((yy - cy) / max(r, 1))**2
        # Soft interface (Gaussian edge to model trajectory aberrations)
        interface_width = max(0.5, r * 0.25)
        weight = np.exp(-0.5 * ((dist2 - 1.0) / (interface_width / r))**2)
        mask_core = dist2 < 1.0
        arr[mask_core] = c_precipitate
        # Solute-depleted zone (depletion shell) around precipitate
        depletion_shell = (dist2 >= 1.0) & (dist2 < 1.8)
        arr[depletion_shell] = np.minimum(arr[depletion_shell], 0.15)

    # ── 3. Grain boundaries (planar solute segregation) ───────────────────
    n_boundaries = rng.randint(2, 5)
    for _ in range(n_boundaries):
        # Random line through the image (parametrised by angle and offset)
        angle = rng.uniform(0, np.pi)
        offset = rng.uniform(0.1, 0.9)  # fractional offset from centre
        yy, xx = np.mgrid[:H, :W]
        # Line equation: cos(angle)*(x/W - 0.5) + sin(angle)*(y/H - 0.5) = offset - 0.5
        dist_to_line = np.abs(
            np.cos(angle) * (xx / max(W, 1) - 0.5)
            + np.sin(angle) * (yy / max(H, 1) - 0.5)
            - (offset - 0.5)
        )
        gb_mask = dist_to_line < (1.5 / max(H, W))  # ~1-2 pixel width
        gb_concentration = rng.uniform(0.55, 0.80)  # Cr/B/P enrichment at GB
        arr[gb_mask] = gb_concentration

    # ── 4. Dislocation loops (curved partial segregation) ─────────────────
    n_dislocations = rng.randint(1, 4)
    for _ in range(n_dislocations):
        cx = rng.randint(20, W - 20)
        cy = rng.randint(20, H - 20)
        r_loop = rng.randint(8, 20)
        yy, xx = np.ogrid[:H, :W]
        dist_to_circle = np.abs(
            np.sqrt(((xx - cx)**2 + (yy - cy)**2).astype(np.float32)) - r_loop
        )
        disloc_mask = dist_to_circle < 1.5
        arr[disloc_mask] = rng.uniform(0.50, 0.70)

    # ── 5. Trajectory aberration (local magnification artefact) ───────────
    # Low-frequency multiplicative distortion field from differential
    # evaporation at precipitate/matrix interfaces
    distortion = rng.randn(H // 8 + 1, W // 8 + 1).astype(np.float32)
    from PIL import Image
    distortion_full = np.array(
        Image.fromarray(distortion).resize((W, H), Image.BILINEAR),
        dtype=np.float32
    )
    distortion_full = gaussian_filter(distortion_full, sigma=max(H, W) // 16)
    distortion_full = distortion_full / (np.std(distortion_full) + 1e-8) * 0.04
    arr = arr * (1.0 + distortion_full)

    # ── 6. Final cleanup ─────────────────────────────────────────────────
    arr = np.clip(arr, 0.0, 1.0)
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
# Generated bioluminescence tomography (BLT) source phantom
# ---------------------------------------------------------------------------

def generate_blt_source_phantom(
    target_shape: Optional[Tuple[int, ...]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2-D bioluminescent source distribution for BLT benchmarks.

    Models the projected 2-D ground-truth bioluminescent source map that is to
    be recovered from photon-flux measurements on the body surface of a
    small-animal (mouse/rat) subject.  The forward model is the diffusion
    approximation to radiative transfer in tissue:

        -∇·[D(r)∇Φ(r)] + μ_a(r) Φ(r) = S(r)   (steady-state)

    where D = 1 / [3(μ_a + μ_s')] is the photon diffusion coefficient
    and S(r) is the volumetric bioluminescent source (W·cm⁻³).

    The 2-D image represents a single coronal-plane cross-section through
    the source distribution, as is standard in the BLT literature.

    Physics basis
    -------------
    BLT geometry: cylindrical mouse body (~2.5 cm diameter); depth range 0–12 mm.
    Tissue optical properties (from Jacques, Phys. Med. Biol. 2013):
      - Muscle:         μ_a ≈ 0.23 cm⁻¹, μ_s' ≈ 8.0 cm⁻¹
      - Fat/skin:       μ_a ≈ 0.10 cm⁻¹, μ_s' ≈ 10.0 cm⁻¹
      - Tumour/source:  typically 1–5 mm diameter, depth 3–10 mm
    Bioluminescent source intensity: 10⁷–10⁹ photons·s⁻¹ (Bhaumik & Bhaumik 2007)

    Microstructural features
    ------------------------
    1. Background tissue — uniform low-level autofluorescence (0.02–0.05)
       representing the ambient optical background in the mouse torso.
    2. Tumour foci (primary sources) — 2–5 elliptical high-intensity regions
       (0.70–1.0) modelling luciferase-expressing tumour cell populations;
       diameters 3–8 mm.  Calibrated to Lv et al., PMB 2006 (BLT phantom).
    3. Satellite sources — 1–3 smaller secondary foci (0.40–0.65) at varying
       depths, representing metastatic or satellite lesions.
    4. Depth-dependent attenuation gradient — linear decay with depth (r_depth)
       approximating diffusion attenuation exp(-μ_eff·d); μ_eff ~0.46 cm⁻¹
       for tissue, matched to Han et al., Opt. Express 2006.
    5. Poisson shot noise — photon-counting noise σ ≈ 0.03 representing CCD
       camera dark current + shot noise at typical BLT exposure times
       (Cong & Wang, Phys. Med. Biol. 2006).
    6. Smooth tissue heterogeneity — low-frequency Gaussian random field
       (σ_spatial ≈ 5 % of image size) modelling μ_a variability across
       tissue types.

    Calibration
    -----------
      - Lv, Y. et al. (2006). A three-dimensional BLT algorithm based on
        radiosity and optical diffusion theory. Phys. Med. Biol. 51:1479-1491.
      - Han, W. et al. (2006). Theoretical and computational analysis of BLT.
        Opt. Express 14(8):3673-3690.
      - Cong, W. & Wang, G. (2006). Boundary integral method for bioluminescence
        tomography. J. Biomed. Opt. 11(2):020503.
      - Jacques, S.L. (2013). Optical properties of biological tissues: a review.
        Phys. Med. Biol. 58(11):R37-R61.
      - Bhaumik, D.K. & Bhaumik, S. (2007). Optical vector analysis and
        multiplexed imaging with unconventional bioluminescent reporter proteins.
        Sci. Rep.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    H = target_shape[0] if target_shape else 128
    W = target_shape[1] if target_shape and len(target_shape) > 1 else H

    # Coordinate grids
    yy = np.linspace(0, 1, H, dtype=np.float32)
    xx = np.linspace(0, 1, W, dtype=np.float32)
    XX, YY = np.meshgrid(xx, yy)

    # ── 1. Tissue background (autofluorescence + ambient scatter) ─────────────
    # Uniform low-level background with gentle spatial heterogeneity
    background_level = rng.uniform(0.02, 0.05)
    arr = np.full((H, W), background_level, dtype=np.float32)

    # Add low-frequency tissue heterogeneity (optical property variation)
    heterogeneity_scale = int(max(H, W) * 0.15)
    tissue_noise = rng.randn(H, W).astype(np.float32) * 0.012
    tissue_noise = gaussian_filter(tissue_noise, sigma=heterogeneity_scale)
    arr += tissue_noise

    # ── 2. Depth-dependent attenuation (diffusion approximation) ─────────────
    # Effective attenuation coefficient μ_eff ≈ 0.46 cm⁻¹; over ~10 mm depth
    # produces ~40 % attenuation, consistent with BLT surface measurements.
    # Represented as a vertical gradient (depth = vertical axis of 2D projection).
    depth_attenuation = np.exp(-1.8 * YY).astype(np.float32)   # stronger near top
    depth_attenuation = (depth_attenuation - depth_attenuation.min()) / \
                        (depth_attenuation.max() - depth_attenuation.min() + 1e-8)
    depth_attenuation = 0.85 + 0.15 * depth_attenuation        # modest gradient

    # ── 3. Primary tumour foci (main bioluminescent sources) ─────────────────
    # 2–5 elliptical sources at varying depths calibrated to Lv et al. 2006
    n_primary = rng.randint(2, 6)
    for _ in range(n_primary):
        # Tumour foci tend to cluster in abdominal / thoracic region (y=0.2-0.8)
        cy = rng.uniform(0.20, 0.80)
        cx = rng.uniform(0.15, 0.85)
        # Semi-axes in [0.03, 0.10] (3-10 % of image) → 4-13 mm in 128 px
        ry = rng.uniform(0.030, 0.100)
        rx = rng.uniform(0.025, 0.090)
        angle = rng.uniform(0, np.pi)
        # Rotated ellipse
        dy = (YY - cy)
        dx = (XX - cx)
        dy_r = dy * np.cos(angle) + dx * np.sin(angle)
        dx_r = -dy * np.sin(angle) + dx * np.cos(angle)
        ellipse = (dy_r / ry) ** 2 + (dx_r / rx) ** 2
        mask = ellipse < 1.0
        # Soft Gaussian fall-off within ellipse for realistic source profile
        intensity = rng.uniform(0.70, 1.00)
        falloff = np.exp(-2.5 * ellipse)
        source_profile = np.where(mask, intensity * falloff, 0.0).astype(np.float32)
        # Apply depth attenuation: sources near surface are brighter
        depth_weight = float(1.0 - 0.5 * cy)  # shallower → brighter projection
        arr += depth_weight * source_profile

    # ── 4. Satellite / metastatic lesions ─────────────────────────────────────
    n_satellite = rng.randint(1, 4)
    for _ in range(n_satellite):
        cy = rng.uniform(0.10, 0.90)
        cx = rng.uniform(0.10, 0.90)
        # Smaller than primary: 2–5 mm diameter
        r = rng.uniform(0.012, 0.040)
        dist_sq = (YY - cy) ** 2 + (XX - cx) ** 2
        satellite_mask = dist_sq < r ** 2
        intensity = rng.uniform(0.35, 0.65)
        gaussian_profile = intensity * np.exp(-dist_sq / (2 * (r * 0.5) ** 2))
        arr += np.where(satellite_mask, gaussian_profile, 0.0).astype(np.float32)

    # ── 5. Poisson shot noise (CCD photon-counting noise) ─────────────────────
    # σ ≈ 0.03 in normalised units matching BLT phantom signal levels
    shot_noise = (rng.randn(H, W) * 0.030).astype(np.float32)
    arr += shot_noise

    # ── 6. Mild smoothing (finite CCD pixel + partial volume) ─────────────────
    arr = gaussian_filter(arr, sigma=0.6)

    return normalize_array(arr)


def generate_brachytherapy_seed_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """
    Brachytherapy post-implant seed phantom.

    Simulates I-125 prostate seed implants: ~80-120 seeds arranged in a
    template grid with ±2mm placement uncertainty, embedded in a soft-tissue
    ellipsoid with heterogeneous attenuation (urethra, rectum, pubic bone).
    Multi-view X-ray projections via Radon transform.

    Reference geometry: TG-43 prostate implant template (ABS, 2012).
    """
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- Anatomy: soft-tissue ellipsoid (prostate) ---
        mu_tissue = np.zeros((H, W), dtype=np.float32)
        cy, cx = H // 2, W // 2
        ry, rx = H // 3, W // 3
        Y, X = np.ogrid[:H, :W]
        ellipse = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
        mu_tissue[ellipse <= 1.0] = 0.20  # soft tissue ~0.20 /cm

        # Urethra: low-attenuation tube along centre
        ur = max(2, H // 40)
        mu_tissue[cy - ur:cy + ur, cx - ur:cx + ur] = 0.05

        # Pubic bone: high-attenuation arc (superior)
        bone_y = cy - int(ry * 0.85)
        for bx in range(cx - rx + 5, cx + rx - 5):
            if 0 <= bone_y < H:
                mu_tissue[bone_y, bx] = np.clip(
                    mu_tissue[bone_y, bx] + 0.80 + 0.2 * rng.random(), 0, 1.2
                )

        # --- Seeds: I-125 high-attenuation point sources ---
        n_seeds = int(rng.integers(70, 110))
        # Template grid (5×6 to 6×8) with random offsets
        grid_cols = int(rng.integers(5, 8))
        grid_rows = int(rng.integers(5, 8))
        step_y = max(1, int((ry * 1.4) / grid_rows))
        step_x = max(1, int((rx * 1.4) / grid_cols))
        start_y = cy - (grid_rows // 2) * step_y
        start_x = cx - (grid_cols // 2) * step_x

        seed_map = np.zeros((H, W), dtype=np.float32)
        placed = 0
        for gy in range(grid_rows):
            for gx in range(grid_cols):
                if placed >= n_seeds:
                    break
                sy = int(start_y + gy * step_y + rng.integers(-2, 3))
                sx = int(start_x + gx * step_x + rng.integers(-2, 3))
                sy = np.clip(sy, 2, H - 3)
                sx = np.clip(sx, 2, W - 3)
                # I-125 seed: titanium capsule, mu ~ 8.0 /cm (effective)
                seed_map[sy, sx] += 8.0
                placed += 1

        x_true = mu_tissue + seed_map  # combined attenuation map

        # --- Forward model: multi-view Radon projection ---
        try:
            from skimage.transform import radon
            n_angles = 18
            theta = np.linspace(0, 180, n_angles, endpoint=False)
            y_meas = radon(x_true, theta=theta, circle=True).astype(np.float32)
        except ImportError:
            y_meas = x_true.copy()

        # Add quantum noise (Poisson approximation)
        sigma_n = 0.02 * float(y_meas.max()) * (1 + 0.5 * rng.random())
        y_meas = y_meas + rng.normal(0, sigma_n, y_meas.shape).astype(np.float32)

        # Ideal operator (identity for FBP-based evaluation)
        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true.astype(np.float32),
            "y": y_meas.astype(np.float32),
            "H_ideal": H_ideal,
            "metadata": {
                "n_seeds": int(placed),
                "seed_activity_mCi": float(0.40 + 0.15 * rng.random()),
                "isotope": "I-125",
                "implant_template": "TG-43",
            },
        })

    # When called as a single-image generator (target_shape provided), return
    # the x_true array of the first sample normalised to [0, 1].
    if target_shape is not None:
        if samples:
            arr = samples[0]["x_true"]
            return normalize_array(arr)
        return np.zeros((H, W), dtype=np.float32)

    return samples


def generate_brillouin_vipa_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """
    Brillouin VIPA spectrometer phantom for viscoelastic mapping.

    Generates spatially-resolved Brillouin shift maps of biological samples
    (cell monolayers/tissue sections) with realistic VIPA spectral signatures.
    Forward model: Lorentzian peak at Ω_B(x,y) with elastic leakage background.

    Reference: Prevedel et al., Nat. Methods 2019; Antonacci & Braakman 2022.
    """
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    N_freq = 64  # spectral channels

    for i in range(n_samples):
        # --- Ground truth: Brillouin shift map (GHz) ---
        # Background: culture medium ~5.1 GHz
        shift_map = np.full((H, W), 5.1, dtype=np.float32)

        # Cell bodies: higher shift ~5.5-6.0 GHz (stiffer cytoplasm)
        n_cells = int(rng.integers(4, 10))
        for _ in range(n_cells):
            cy = int(rng.integers(H // 5, 4 * H // 5))
            cx = int(rng.integers(W // 5, 4 * W // 5))
            r = int(rng.integers(max(2, H // 10), max(3, H // 5)))
            Y, X = np.ogrid[:H, :W]
            mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= r ** 2
            shift_map[mask] = float(rng.uniform(5.5, 6.2))
            # Nucleus: even stiffer ~6.5-7.0 GHz
            r_nuc = max(2, r // 3)
            nuc_mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= r_nuc ** 2
            shift_map[nuc_mask] = float(rng.uniform(6.5, 7.2))

        # Smooth transitions
        shift_map = gaussian_filter(shift_map, sigma=1.5).astype(np.float32)
        x_true = shift_map  # GHz map

        # --- Forward model: VIPA spectra ---
        # Frequency axis: centred at 0, ±20 GHz range
        freq_axis = np.linspace(-20, 20, N_freq, dtype=np.float32)
        gamma_B = 0.8  # Brillouin linewidth (GHz, typical for biological samples)
        gamma_R = 0.1  # elastic peak width (instrument-limited)

        spectra = np.zeros((H, W, N_freq), dtype=np.float32)
        I_B = 0.05  # Brillouin peak intensity (relative to elastic)
        I_R = 1.0   # elastic peak intensity

        for hy in range(H):
            for hx in range(W):
                omega_b = shift_map[hy, hx]
                # Anti-Stokes and Stokes Brillouin peaks (Lorentzian)
                peak_as = I_B * (gamma_B / 2 / np.pi) / ((freq_axis - omega_b) ** 2 + (gamma_B / 2) ** 2)
                peak_s  = I_B * (gamma_B / 2 / np.pi) / ((freq_axis + omega_b) ** 2 + (gamma_B / 2) ** 2)
                # Elastic leakage
                elastic_leak = 0.02 * I_R * (gamma_R / 2 / np.pi) / (freq_axis ** 2 + (gamma_R / 2) ** 2)
                spectra[hy, hx] = peak_as + peak_s + elastic_leak

        # Shot noise
        max_signal = float(spectra.max())
        if max_signal > 0:
            noise_level = 0.005 * max_signal * (1 + 0.5 * rng.random())
            y_meas = (spectra + rng.normal(0, noise_level, spectra.shape)).astype(np.float32)
        else:
            y_meas = spectra.copy()

        # Ideal operator (identity — spectral fitting extracts shift)
        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        # Normalise x_true to [0,1] for compatibility with generic pipeline
        x_min = float(x_true.min())
        x_max = float(x_true.max())
        if x_max > x_min:
            x_norm = (x_true - x_min) / (x_max - x_min)
        else:
            x_norm = np.zeros_like(x_true)

        samples.append({
            "x_true": x_norm,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_cells": int(n_cells),
                "freq_range_GHz": [-20.0, 20.0],
                "laser_nm": 532,
                "gamma_B_GHz": float(gamma_B),
                "shift_min_GHz": x_min,
                "shift_max_GHz": x_max,
            },
        })

    return samples


def generate_cars_raman_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    CARS (Coherent Anti-Stokes Raman Scattering) microscopy phantom.

    Simulates hyperspectral CARS images of biological cells with lipid droplets
    (CH2 resonance ~2845 cm-1) and protein-rich regions (amide I ~1655 cm-1).
    Forward model: coherent superposition of resonant signal and non-resonant
    background (NRB). Reconstruction recovers Im[chi^(3)] (pure Raman spectrum).

    Reference: Cheng & Xie, J. Phys. Chem. B 2004; Camp & Cicerone, Nat. Photon. 2015.
    """
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    N_wn = 32  # wavenumber channels

    for i in range(n_samples):
        # --- Ground truth: Im[chi^3] chemical maps ---
        # Baseline: water/buffer ~0
        chi_im = np.zeros((H, W), dtype=np.float32)

        # Lipid droplets (CH2 at 2845 cm-1): high CARS signal
        n_lipid = int(rng.integers(3, 8))
        for _ in range(n_lipid):
            cy = int(rng.integers(H // 6, 5 * H // 6))
            cx = int(rng.integers(W // 6, 5 * W // 6))
            r = int(rng.integers(3, 10))
            Y, X = np.ogrid[:H, :W]
            mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= r ** 2
            chi_im[mask] = float(rng.uniform(0.7, 1.0))

        # Protein-rich cytoplasm
        n_cells = int(rng.integers(2, 5))
        for _ in range(n_cells):
            cy = int(rng.integers(H // 4, 3 * H // 4))
            cx = int(rng.integers(W // 4, 3 * W // 4))
            ry = int(rng.integers(H // 6, H // 3))
            rx = int(rng.integers(W // 6, W // 3))
            Y, X = np.ogrid[:H, :W]
            cell_mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
            chi_im[cell_mask] = np.maximum(chi_im[cell_mask], float(rng.uniform(0.2, 0.5)))

        chi_im = gaussian_filter(chi_im, sigma=0.8).astype(np.float32)
        x_true = chi_im

        # --- Forward model: CARS signal with NRB ---
        # NRB (non-resonant background) is spatially uniform
        A_NRB = float(rng.uniform(0.3, 0.8))  # NRB amplitude
        # CARS = |chi_r + chi_NRB|^2 = chi_r^2 + 2*chi_r*chi_NRB + chi_NRB^2
        y_cars = chi_im ** 2 + 2 * A_NRB * chi_im + A_NRB ** 2

        # Shot noise
        sigma_n = 0.02 * float(y_cars.max()) * (1 + rng.random() * 0.5)
        y_meas = (y_cars + rng.normal(0, sigma_n, y_cars.shape)).astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_lipid_droplets": int(n_lipid),
                "n_cells": int(n_cells),
                "A_NRB": float(A_NRB),
                "wavenumber_cm1": 2845,
            },
        })

    return samples


def generate_cacti_video_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """
    CACTI coded aperture compressive temporal imaging phantom.

    Simulates B=8 frame high-speed video encoded into a single coded snapshot.
    Dynamic scene: moving discs/bars simulating fast fluid flow or oscillating
    mechanical components. Binary coded aperture mask (50% fill factor).

    Reference: Llull et al., Optica 2015; Qiao et al., Nat. Photonics 2020.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    B = 8  # frames per shot

    for i in range(n_samples):
        # --- Binary coded aperture mask ---
        mask = rng.integers(0, 2, size=(H, W), dtype=np.float32)

        # --- Ground truth video: B frames of dynamic scene ---
        x_true = np.zeros((H, W, B), dtype=np.float32)

        # Static background
        background = rng.uniform(0.1, 0.4, (H, W)).astype(np.float32)

        # Moving objects (discs/bars)
        n_objects = int(rng.integers(2, 6))
        for _ in range(n_objects):
            oy0 = int(rng.integers(H // 5, 4 * H // 5))
            ox0 = int(rng.integers(W // 5, 4 * W // 5))
            r = int(rng.integers(5, 20))
            vy = float(rng.uniform(-3, 3))  # pixels/frame
            vx = float(rng.uniform(-3, 3))
            intensity = float(rng.uniform(0.6, 1.0))

            Y, X = np.ogrid[:H, :W]
            for t in range(B):
                cy = int(np.clip(oy0 + vy * t, r, H - r - 1))
                cx = int(np.clip(ox0 + vx * t, r, W - r - 1))
                disc = ((Y - cy) ** 2 + (X - cx) ** 2) <= r ** 2
                x_true[:, :, t] += disc * intensity

        x_true = np.clip(x_true + background[:, :, None], 0, 1).astype(np.float32)

        # --- Forward model: coded snapshot measurement ---
        y = np.zeros((H, W), dtype=np.float32)
        for t in range(B):
            y += mask * x_true[:, :, t]
        y /= B  # normalize

        # Add read noise
        sigma = 0.01 * (1 + 0.5 * rng.random())
        y = (y + rng.normal(0, sigma, y.shape)).astype(np.float32)

        # Ideal operator (mask-based linear measurement)
        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true[:, :, 0],  # store first frame as 2D for eval
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "n_frames": B,
                "fill_factor": float(mask.mean()),
                "n_objects": int(n_objects),
            },
        })

    return samples


def generate_cathodoluminescence_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Cathodoluminescence (CL) imaging phantom for SEM/TEM.

    Simulates CL intensity maps of semiconductor nanostructures with plasmonic
    nanoparticles, quantum dots, and grain boundary defects. Models the
    parabolic mirror collection PSF, spectral background, and PMT shot noise.

    Reference: Zagonel et al., Nano Lett. 2011; Tizei & Kociak, Phys. Rev. Lett. 2013.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- Ground truth: CL emission map ---
        # Background: substrate emission (low)
        cl_map = rng.uniform(0.02, 0.08, (H, W)).astype(np.float32)

        # Bright plasmonic nanoparticles / quantum dots
        n_particles = int(rng.integers(8, 20))
        for _ in range(n_particles):
            py = int(rng.integers(5, H - 5))
            px = int(rng.integers(5, W - 5))
            intensity = float(rng.uniform(0.6, 1.0))
            r = int(rng.integers(2, 6))
            Y, X = np.ogrid[:H, :W]
            mask = ((Y - py) ** 2 + (X - px) ** 2) <= r ** 2
            cl_map[mask] = np.maximum(cl_map[mask], intensity)

        # Grain boundaries: linear dark features (crystal defects reduce CL)
        n_grains = int(rng.integers(2, 5))
        for _ in range(n_grains):
            y0 = int(rng.integers(0, H))
            angle = float(rng.uniform(0, np.pi))
            for t in range(-max(H, W), max(H, W)):
                gy = int(y0 + t * np.sin(angle))
                gx = int(H // 2 + t * np.cos(angle))
                if 0 <= gy < H and 0 <= gx < W:
                    cl_map[gy, gx] *= 0.3  # dark grain boundary

        cl_map = np.clip(cl_map, 0, 1).astype(np.float32)
        x_true = cl_map

        # --- Forward model: CL measurement with PSF and noise ---
        # Parabolic mirror PSF broadening
        psf_sigma = float(rng.uniform(1.0, 2.5))
        y_blurred = gaussian_filter(cl_map, sigma=psf_sigma)

        # Shot noise (Poisson approximation)
        gain = float(rng.uniform(50, 200))  # PMT gain
        signal_counts = y_blurred * gain
        shot_noise = rng.normal(0, np.sqrt(np.maximum(signal_counts, 1))) / gain
        y_meas = (y_blurred + shot_noise).astype(np.float32)
        y_meas = np.clip(y_meas, 0, None)

        # Spectral background
        bg = float(rng.uniform(0.01, 0.05))
        y_meas = (y_meas + bg).astype(np.float32)

        # Ideal operator (PSF convolution → represented as identity for eval)
        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_particles": int(n_particles),
                "psf_sigma_px": float(psf_sigma),
                "pmt_gain": float(gain),
                "background": float(bg),
            },
        })

    return samples


def generate_cbct_head_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Cone-Beam CT (CBCT) head/dental phantom.

    Simulates CBCT measurements of dental/maxillofacial anatomy: teeth (high
    attenuation), mandible/maxilla bone, soft tissue, air cavities. Forward
    model: 2D fan-beam Radon projection (proxy for CBCT slice). Artefacts:
    metal implant streaks, beam hardening, scatter.

    Reference: Feldkamp et al., J. Opt. Soc. Am. A 1984; Miracle & Mukherji,
    AJNR 2009.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        mu = np.zeros((H, W), dtype=np.float32)

        # Soft tissue oval (head)
        cy, cx = H // 2, W // 2
        ry, rx = int(H * 0.42), int(W * 0.38)
        Y, X = np.ogrid[:H, :W]
        head = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
        mu[head] = 0.20

        # Skull bone ring
        skull_thick = max(3, H // 20)
        skull_outer = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
        skull_inner_ry = ry - skull_thick
        skull_inner_rx = rx - skull_thick
        skull_inner = ((Y - cy) / skull_inner_ry) ** 2 + ((X - cx) / skull_inner_rx) ** 2 <= 1.0
        mu[skull_outer & ~skull_inner] = 0.60

        # Air cavities (sinuses)
        n_cavities = int(rng.integers(2, 5))
        for _ in range(n_cavities):
            cvy = int(rng.integers(cy - ry // 2, cy + ry // 3))
            cvx = int(rng.integers(cx - rx // 2, cx + rx // 2))
            rv = int(rng.integers(4, 12))
            cavity = ((Y - cvy) ** 2 + (X - cvx) ** 2) <= rv ** 2
            mu[cavity & head] = 0.0  # air

        # Teeth (high attenuation)
        n_teeth = int(rng.integers(4, 8))
        teeth_y = cy + int(ry * 0.35)
        for t in range(n_teeth):
            tx = cx - int(rx * 0.5) + t * (rx // (n_teeth - 1))
            tooth_r = max(2, H // 30)
            tooth = ((Y - teeth_y) ** 2 + (X - tx) ** 2) <= tooth_r ** 2
            mu[tooth] = 1.80  # enamel

        # Metal implant (occasional)
        has_implant = rng.random() > 0.5
        if has_implant:
            impl_y = teeth_y + int(rng.integers(-3, 4))
            impl_x = cx + int(rng.integers(-rx // 3, rx // 3))
            impl_r = max(2, H // 40)
            impl = ((Y - impl_y) ** 2 + (X - impl_x) ** 2) <= impl_r ** 2
            mu[impl] = 4.50  # titanium implant

        mu = np.clip(mu, 0, None).astype(np.float32)
        x_true = mu

        # Forward: Radon projection
        try:
            from skimage.transform import radon
            n_angles = 36
            theta = np.linspace(0, 180, n_angles, endpoint=False)
            y_proj = radon(mu, theta=theta, circle=True).astype(np.float32)
        except ImportError:
            y_proj = mu.copy()

        sigma_n = 0.015 * float(y_proj.max()) * (1 + 0.3 * rng.random())
        y_meas = (y_proj + rng.normal(0, sigma_n, y_proj.shape)).astype(np.float32)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_teeth": int(n_teeth),
                "n_cavities": int(n_cavities),
                "has_implant": bool(has_implant),
            },
        })

    return samples


def generate_cest_mri_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    CEST MRI (Chemical Exchange Saturation Transfer) phantom.

    Simulates z-spectrum acquisitions across brain tissue with tumour regions.
    Models the asymmetric magnetisation transfer (MT) asymmetry and amide
    proton transfer (APT) effect at +3.5 ppm. Reconstruction goal: extract
    the APT map (proportional to mobile protein concentration / pH).

    Reference: Ward et al., J. Magn. Reson. 2000; Zhou et al., Nat. Med. 2003.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []
    N_offsets = 32  # z-spectrum offset points

    for i in range(n_samples):
        # --- Ground truth: APT map (% signal asymmetry) ---
        apt_map = np.zeros((H, W), dtype=np.float32)

        # Normal brain: APT ~1.5-2.0%
        brain_mask = np.zeros((H, W), bool)
        cy, cx = H // 2, W // 2
        ry, rx = H // 3, W // 3
        Y, X = np.ogrid[:H, :W]
        brain_mask = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0
        apt_map[brain_mask] = float(rng.uniform(1.5, 2.0))

        # Tumour: elevated APT ~3.0-4.5% (high protein content)
        n_tumours = int(rng.integers(1, 3))
        for _ in range(n_tumours):
            ty = int(rng.integers(cy - ry // 2, cy + ry // 2))
            tx = int(rng.integers(cx - rx // 2, cx + rx // 2))
            tr = int(rng.integers(5, 12))
            tumour = ((Y - ty) ** 2 + (X - tx) ** 2) <= tr ** 2
            apt_map[tumour & brain_mask] = float(rng.uniform(3.0, 4.5))

        # Stroke (low pH): reduced APT ~0.5-1.0%
        if rng.random() > 0.6:
            sy = int(rng.integers(cy - ry // 2, cy + ry // 2))
            sx = int(rng.integers(cx - rx // 2, cx + rx // 2))
            sr = int(rng.integers(4, 10))
            stroke = ((Y - sy) ** 2 + (X - sx) ** 2) <= sr ** 2
            apt_map[stroke & brain_mask] = float(rng.uniform(0.3, 0.9))

        apt_map = gaussian_filter(apt_map, sigma=1.0).astype(np.float32)
        x_true = apt_map  # percentage APT

        # --- Forward model: z-spectrum ---
        freq_offsets = np.linspace(-6, 6, N_offsets)  # ppm
        z_spec = np.ones((H, W, N_offsets), dtype=np.float32)

        for hi in range(H):
            for wi in range(W):
                if not brain_mask[hi, wi]:
                    continue
                apt_val = apt_map[hi, wi] / 100.0
                for j, offset in enumerate(freq_offsets):
                    # Direct water saturation (Lorentzian)
                    ds = 0.95 * np.exp(-offset ** 2 / (2 * 0.5 ** 2))
                    # MT asymmetry
                    mt = 0.03 * np.exp(-np.abs(offset) / 2.0)
                    # APT at +3.5 ppm
                    apt_effect = apt_val * np.exp(-(offset - 3.5) ** 2 / (2 * 0.4 ** 2))
                    z_spec[hi, wi, j] = float(np.clip(1 - ds - mt - apt_effect, 0, 1))

        # Add thermal noise
        sigma_n = 0.005 * (1 + 0.5 * rng.random())
        y_meas = (z_spec + rng.normal(0, sigma_n, z_spec.shape)).astype(np.float32)
        y_meas = np.clip(y_meas, 0, 1)

        # Measurement is the z-spectrum (H x W x N_offsets flattened to H x W for eval)
        # Store first z-spectrum slice as 2D
        y_2d = y_meas[:, :, N_offsets // 2]

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_2d,
            "H_ideal": H_ideal,
            "metadata": {
                "n_tumours": int(n_tumours),
                "n_offsets": int(N_offsets),
                "freq_range_ppm": [-6.0, 6.0],
                "noise_sigma": float(sigma_n),
            },
        })

    return samples


def generate_ceus_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Contrast-Enhanced Ultrasound (CEUS) microbubble phantom.

    Simulates B-mode + contrast mode ultrasound of liver vasculature with
    microbubble perfusion. Models the nonlinear harmonic response of
    microbubbles and speckle noise from tissue background. Reconstruction
    goal: super-resolved vessel map from multiple bubble frames (ULM).

    Reference: Errico et al., Nature 2015 (ULM); Lowerison et al., Nat. Commun. 2022.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- Ground truth: vessel perfusion map ---
        vessel_map = np.zeros((H, W), dtype=np.float32)

        # Main portal vein (large vessel)
        portal_y = H // 2 + int(rng.integers(-10, 10))
        vessel_w = int(rng.integers(4, 10))
        vessel_map[portal_y - vessel_w:portal_y + vessel_w, W // 4:3 * W // 4] = 1.0

        # Branching hepatic vessels
        n_branches = int(rng.integers(3, 7))
        for b in range(n_branches):
            bx_start = W // 4 + b * (W // (2 * n_branches))
            by = portal_y + int(rng.integers(-H // 4, H // 4))
            bw = max(1, vessel_w // 2)
            length = int(rng.integers(H // 6, H // 3))
            direction = rng.choice([-1, 1])
            for t in range(length):
                gy = by + direction * t
                gx = bx_start + int(rng.integers(-2, 3))
                if 0 <= gy < H and 0 <= gx < W:
                    vessel_map[max(0, gy - bw):min(H, gy + bw),
                               max(0, gx - bw):min(W, gx + bw)] = 0.8

        vessel_map = np.clip(vessel_map, 0, 1).astype(np.float32)
        x_true = vessel_map

        # --- Tissue background: ultrasound speckle ---
        tissue = rng.rayleigh(0.3, (H, W)).astype(np.float32)
        liver_mask = np.ones((H, W), bool)  # assume full FOV is liver
        tissue *= liver_mask

        # --- Microbubble contrast: nonlinear harmonic signal ---
        # Bubbles produce bright, sparse signals along vessels
        mb_signal = np.zeros((H, W), dtype=np.float32)
        n_bubbles = int(rng.integers(50, 150))
        for _ in range(n_bubbles):
            # Sample bubble position weighted by vessel probability
            flat_vessel = vessel_map.ravel()
            if flat_vessel.sum() > 0:
                probs = flat_vessel / flat_vessel.sum()
                idx = rng.choice(len(probs), p=probs)
            else:
                idx = rng.integers(H * W)
            by, bx = divmod(int(idx), W)
            mb_signal[by, bx] += float(rng.uniform(0.5, 1.0))

        mb_signal = gaussian_filter(mb_signal, sigma=1.5)

        # Combined CEUS measurement
        y_meas = (tissue + 2.0 * mb_signal + rng.normal(0, 0.05, (H, W))).astype(np.float32)
        y_meas = np.clip(y_meas, 0, None)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_branches": int(n_branches),
                "n_bubbles": int(n_bubbles),
                "contrast_agent": "SonoVue",
            },
        })

    return samples


def generate_clem_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    CLEM (Correlative Light and Electron Microscopy) phantom.

    Generates paired FM+EM image pairs of cellular structures. FM image:
    sparse fluorescent spots (proteins of interest) with diffraction-limited
    PSF. EM image: dense ultrastructural detail of the same region with
    organelles, membranes, and vesicles. The reconstruction challenge is
    multi-modal image fusion and super-resolution from FM guided by EM.

    Reference: Bharat et al., Nat. Methods 2018; Hurbain & Sachse, Biol. Cell 2011.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- EM image: ultrastructural detail (ground truth) ---
        em_img = rng.uniform(0.15, 0.35, (H, W)).astype(np.float32)

        # Cell membrane
        cy, cx = H // 2, W // 2
        ry, rx = H // 3, W // 3
        Y, X = np.ogrid[:H, :W]
        cell = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
        mem_thick = 2
        membrane = (cell <= 1.0) & (cell >= (1.0 - mem_thick / min(ry, rx)) ** 2)
        em_img[membrane] = 0.85

        # Organelles: mitochondria (elongated, dark matrix + bright cristae)
        n_mito = int(rng.integers(3, 7))
        for _ in range(n_mito):
            my = int(rng.integers(cy - ry // 2, cy + ry // 2))
            mx = int(rng.integers(cx - rx // 2, cx + rx // 2))
            ml = int(rng.integers(10, 20))
            mw = int(rng.integers(4, 8))
            angle = float(rng.uniform(0, np.pi))
            for t in range(-ml // 2, ml // 2):
                gy = int(my + t * np.sin(angle))
                gx = int(mx + t * np.cos(angle))
                for dy in range(-mw // 2, mw // 2):
                    for dx in range(-mw // 2, mw // 2):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < H and 0 <= nx < W:
                            em_img[ny, nx] = 0.25  # mito matrix

        # Vesicles (dense, round)
        n_vesicles = int(rng.integers(5, 15))
        for _ in range(n_vesicles):
            vy = int(rng.integers(cy - ry, cy + ry))
            vx = int(rng.integers(cx - rx, cx + rx))
            vr = int(rng.integers(2, 5))
            vesicle = ((Y - vy) ** 2 + (X - vx) ** 2) <= vr ** 2
            em_img[vesicle] = 0.85

        em_img = np.clip(em_img, 0, 1).astype(np.float32)
        x_true = em_img  # EM is the target (ground truth structure)

        # --- FM image: fluorescence of labelled proteins (y measurement) ---
        fm_img = np.zeros((H, W), dtype=np.float32)

        # Fluorescent labels co-localise with some vesicles
        n_labels = int(rng.integers(3, 8))
        for _ in range(n_labels):
            ly = int(rng.integers(cy - ry // 2, cy + ry // 2))
            lx = int(rng.integers(cx - rx // 2, cx + rx // 2))
            fm_img[ly, lx] = float(rng.uniform(0.7, 1.0))

        # Diffraction-limited PSF broadening (FM ~250nm / pixel ~10nm → sigma~25px at EM scale)
        fm_blur = gaussian_filter(fm_img, sigma=float(rng.uniform(3.0, 6.0)))
        # FM noise
        fm_noise = rng.normal(0, 0.03, (H, W)).astype(np.float32)
        y_meas = np.clip(fm_blur + fm_noise, 0, 1).astype(np.float32)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_mitochondria": int(n_mito),
                "n_vesicles": int(n_vesicles),
                "n_labels": int(n_labels),
                "fm_psf_sigma": float(rng.uniform(3.0, 6.0)),
            },
        })

    return samples


def generate_coded_exposure_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Coded exposure (flutter shutter) photography phantom.

    Simulates motion-blurred images captured with a Raskar-style binary
    flickering shutter code. A natural image is motion-blurred with the
    coded convolution kernel, then degraded with read noise. Reconstruction
    recovers the sharp frame by deconvolution with the known code.

    Reference: Raskar et al., SIGGRAPH 2006; Agrawal et al., CVPR 2009.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, convolve1d

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []
    CODE_LEN = 52  # Raskar flutter shutter code length

    # Optimal Raskar code (preserves frequencies)
    raskar_code = np.array([
        1,1,1,0,1,1,0,1,0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,0,
        0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,1,0,0,1
    ], dtype=np.float32)
    raskar_code = raskar_code[:CODE_LEN]
    raskar_code /= raskar_code.sum()  # normalise

    for i in range(n_samples):
        # Sharp ground truth: synthetic textured image
        x_true = np.zeros((H, W), dtype=np.float32)
        # Add geometric shapes
        n_shapes = int(rng.integers(3, 8))
        for _ in range(n_shapes):
            shape_type = rng.choice(['rect', 'disc', 'bar'])
            intensity = float(rng.uniform(0.4, 1.0))
            if shape_type == 'rect':
                y0, x0 = int(rng.integers(0, H - 20)), int(rng.integers(0, W - 20))
                h_, w_ = int(rng.integers(10, 40)), int(rng.integers(10, 40))
                x_true[y0:y0+h_, x0:x0+w_] = intensity
            elif shape_type == 'disc':
                cy, cx = int(rng.integers(10, H-10)), int(rng.integers(10, W-10))
                r = int(rng.integers(5, 20))
                Y, X = np.ogrid[:H, :W]
                x_true[((Y-cy)**2+(X-cx)**2) <= r**2] = intensity
            else:  # bar
                y0 = int(rng.integers(0, H-5))
                x_true[y0:y0+3, :] = intensity

        x_true = gaussian_filter(x_true, sigma=0.5).astype(np.float32)
        x_true = np.clip(x_true, 0, 1)

        # Motion direction: horizontal
        motion_len = int(rng.integers(CODE_LEN // 2, CODE_LEN))
        code = raskar_code[:motion_len] / raskar_code[:motion_len].sum()

        # Coded exposure blur: convolve along horizontal axis
        y_blur = convolve1d(x_true, code, axis=1, mode='reflect').astype(np.float32)

        # Read noise
        sigma_n = 0.01 * (1 + rng.random() * 0.5)
        y_meas = (y_blur + rng.normal(0, sigma_n, y_blur.shape)).astype(np.float32)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "code_length": int(motion_len),
                "motion_axis": "horizontal",
                "noise_sigma": float(sigma_n),
            },
        })

    return samples


def generate_confocal_endomicroscopy_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape=None,
) -> list[dict]:
    """
    Confocal laser endomicroscopy (CLE) phantom.

    Simulates CLE images of colonic mucosa with crypt architecture. Models
    fibre bundle honeycomb artefacts, speckle noise, and limited depth of
    field. Reconstruction target: clean cellular-resolution mucosal image.

    Reference: Kiesslich et al., Gastroenterology 2004; Andre et al., Med. Image Anal. 2011.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # Ground truth: colonic crypt architecture
        x_true = np.zeros((H, W), dtype=np.float32)

        # Regular crypt grid (hexagonal-ish arrangement)
        crypt_spacing = int(rng.integers(14, 22))
        crypt_radius = int(rng.integers(4, 7))
        lumen_radius = max(1, crypt_radius - 2)

        for cy in range(crypt_radius, H - crypt_radius, crypt_spacing):
            for cx_idx, cx in enumerate(range(crypt_radius, W - crypt_radius, crypt_spacing)):
                offset = (crypt_spacing // 2) if (cy // crypt_spacing) % 2 else 0
                cx_off = min(cx + offset, W - crypt_radius - 1)
                Y, X = np.ogrid[:H, :W]
                # Crypt wall (bright epithelium)
                crypt = ((Y - cy)**2 + (X - cx_off)**2) <= crypt_radius**2
                lumen = ((Y - cy)**2 + (X - cx_off)**2) <= lumen_radius**2
                x_true[crypt & ~lumen] = float(rng.uniform(0.6, 0.9))
                # Lumen: dark
                x_true[lumen] = float(rng.uniform(0.02, 0.08))

        # Background stroma
        stroma = x_true == 0
        x_true[stroma] = float(rng.uniform(0.15, 0.3))
        x_true = np.clip(x_true, 0, 1).astype(np.float32)

        # Forward model: fibre bundle honeycomb + speckle + blur
        # Honeycomb artefact: regular dark grid
        fibre_period = int(rng.integers(6, 10))
        y_meas = x_true.copy()
        for fy in range(0, H, fibre_period):
            y_meas[fy, :] *= float(rng.uniform(0.3, 0.6))
        for fx in range(0, W, fibre_period):
            y_meas[:, fx] *= float(rng.uniform(0.3, 0.6))

        # PSF blur
        y_meas = gaussian_filter(y_meas, sigma=float(rng.uniform(0.8, 1.5)))

        # Speckle noise
        speckle = rng.rayleigh(0.08, (H, W)).astype(np.float32)
        y_meas = np.clip(y_meas + speckle, 0, 1).astype(np.float32)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "crypt_spacing": int(crypt_spacing),
                "fibre_period": int(fibre_period),
            },
        })

    return samples


def generate_coronagraphy_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Coronagraphic direct imaging phantom for exoplanet detection.

    Simulates post-coronagraph focal plane images with residual stellar
    speckle halo and faint point-like planet companions. Reconstruction:
    detect/recover planet signal from the speckle background using
    ADI-based PSF subtraction.

    Reference: Marois et al., Science 2008; Soummer et al., ApJ 2012 (KLIP).
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        cy, cx = H // 2, W // 2

        # Stellar speckle halo (radially decaying, quasi-random speckles)
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(np.float32)
        r[cy, cx] = 0.01

        # Halo envelope: 1/r^2 decay from IWA (inner working angle)
        iwa = max(3, H // 12)
        halo = np.where(r > iwa, 1.0 / (r ** 2 + 1e-3), 0.0).astype(np.float32)
        halo_max = halo.max()
        if halo_max > 0:
            halo /= halo_max

        # Add speckle pattern (correlated noise)
        speckle_scale = float(rng.uniform(0.3, 0.8))
        speckle = rng.exponential(speckle_scale, (H, W)).astype(np.float32)
        speckle = gaussian_filter(speckle, sigma=float(rng.uniform(1.5, 3.0)))
        speckle *= halo

        # Planet signals (1-3 point sources at contrast ~1e-4 to 1e-3)
        n_planets = int(rng.integers(1, 4))
        planet_map = np.zeros((H, W), dtype=np.float32)
        planet_contrast = float(rng.uniform(1e-4, 1e-3))

        for _ in range(n_planets):
            sep = float(rng.uniform(iwa * 1.5, min(H, W) // 2 - 5))
            pa = float(rng.uniform(0, 2 * np.pi))
            py = int(cy + sep * np.sin(pa))
            px = int(cx + sep * np.cos(pa))
            if 0 <= py < H and 0 <= px < W:
                planet_map[py, px] = planet_contrast

        # Gaussian PSF for planet (diffraction-limited)
        planet_psf = gaussian_filter(planet_map, sigma=1.5)
        x_true = planet_psf / planet_contrast  # normalised planet position map

        # Measurement: speckles + planet
        y_meas = (speckle + planet_psf + rng.normal(0, 0.01, (H, W))).astype(np.float32)
        y_meas = np.clip(y_meas, 0, None)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true.astype(np.float32),
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_planets": int(n_planets),
                "planet_contrast": float(planet_contrast),
                "iwa_pixels": int(iwa),
                "speckle_scale": float(speckle_scale),
            },
        })

    return samples


def generate_confocal_livecell_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (128, 128),
    target_shape=None,
) -> list[dict]:
    """
    Confocal live-cell imaging phantom for denoising benchmarks.

    Simulates fluorescence images of living cells with dynamic organelles
    (mitochondria, endosomes) and cytoskeletal structures. Models low-dose
    acquisition noise (Poisson shot noise + Gaussian read noise) typical of
    live-cell imaging to minimise phototoxicity.

    Reference: Weigert et al., Nat. Methods 2018 (CARE); Krull et al., CVPR 2019 (Noise2Void).
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # Ground truth: clean fluorescence image
        x_true = np.zeros((H, W), dtype=np.float32)

        # Cell boundary
        cy, cx = H // 2 + int(rng.integers(-10, 10)), W // 2 + int(rng.integers(-10, 10))
        ry, rx = H // 3 + int(rng.integers(-5, 5)), W // 3 + int(rng.integers(-5, 5))
        Y, X = np.ogrid[:H, :W]
        cell = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0

        # Cytoplasm
        x_true[cell] = float(rng.uniform(0.1, 0.25))

        # Dynamic organelles: mitochondria (tubular, bright)
        n_mito = int(rng.integers(5, 12))
        for _ in range(n_mito):
            my = int(rng.integers(cy - ry, cy + ry))
            mx = int(rng.integers(cx - rx, cx + rx))
            mlen = int(rng.integers(5, 15))
            angle = float(rng.uniform(0, np.pi))
            mwidth = 1
            for t in range(-mlen // 2, mlen // 2):
                gy = int(my + t * np.cos(angle))
                gx = int(mx + t * np.sin(angle))
                if 0 <= gy < H and 0 <= gx < W and cell[gy, gx]:
                    x_true[max(0,gy-mwidth):min(H,gy+mwidth+1),
                           max(0,gx-mwidth):min(W,gx+mwidth+1)] = float(rng.uniform(0.6, 0.9))

        # Endosomes (small, bright spots)
        n_endo = int(rng.integers(8, 20))
        for _ in range(n_endo):
            ey = int(rng.integers(cy - ry, cy + ry))
            ex = int(rng.integers(cx - rx, cx + rx))
            if 0 <= ey < H and 0 <= ex < W and cell[ey, ex]:
                x_true[ey, ex] = float(rng.uniform(0.7, 1.0))

        # Nucleus
        nuc = ((Y - cy) / (ry // 2)) ** 2 + ((X - cx) / (rx // 2)) ** 2 <= 1.0
        x_true[nuc] = float(rng.uniform(0.3, 0.5))

        x_true = gaussian_filter(x_true, sigma=0.5).astype(np.float32)
        x_true = np.clip(x_true, 0, 1)

        # Forward model: low-dose acquisition noise
        photon_count = float(rng.uniform(10, 50))  # very low for live-cell
        signal_counts = x_true * photon_count
        poisson_noisy = rng.poisson(np.maximum(signal_counts, 0)).astype(np.float32) / photon_count
        read_noise = float(rng.uniform(0.02, 0.08))
        y_meas = np.clip(poisson_noisy + rng.normal(0, read_noise, (H, W)), 0, None).astype(np.float32)

        H_size = min(H * W, 4096)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "photon_count": float(photon_count),
                "read_noise": float(read_noise),
                "n_mitochondria": int(n_mito),
                "n_endosomes": int(n_endo),
            },
        })

    return samples


def generate_cryo_em_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """Generate synthetic single-particle cryo-EM phantom.

    Simulates 2D projections of protein structures with CTF corruption and
    low-dose Poisson noise. Ground truth is a 2D projection of an ellipsoidal
    protein model with internal density variations.

    Forward model:
      - Apply contrast transfer function (CTF) in Fourier space
      - Add Poisson noise at ~10 electrons/Angstrom^2

    Reference: Frank, Three-Dimensional Electron Microscopy of Macromolecular
    Assemblies, Oxford University Press, 2006.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- Ground truth: 2D projection of protein ellipsoid ---
        x_true = np.zeros((H, W), dtype=np.float32)
        cy, cx = H // 2, W // 2

        # Outer ellipsoid (protein envelope)
        ry, rx = H // 3, W // 4
        Y, X = np.ogrid[:H, :W]
        ellipse = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
        x_true[ellipse <= 1.0] = float(rng.uniform(0.4, 0.6))

        # Inner domain (dense core, secondary structure)
        ry_inner, rx_inner = H // 6, W // 7
        inner = ((Y - cy) / ry_inner) ** 2 + ((X - cx) / rx_inner) ** 2
        x_true[inner <= 1.0] = float(rng.uniform(0.7, 0.9))

        # Alpha-helix-like density blobs
        n_helices = int(rng.integers(3, 7))
        for _ in range(n_helices):
            angle = float(rng.uniform(0, 2 * np.pi))
            dist = float(rng.uniform(ry // 4, int(ry * 0.75)))
            hy = int(cy + dist * np.sin(angle))
            hx = int(cx + dist * np.cos(angle))
            blob_r = float(rng.uniform(2, 5))
            blob_dist = np.sqrt((Y - hy) ** 2 + (X - hx) ** 2)
            x_true += (float(rng.uniform(0.15, 0.35)) * np.exp(
                -blob_dist ** 2 / (2 * blob_r ** 2)
            )).astype(np.float32)

        # Normalise ground truth to [0, 1]
        x_min, x_max = float(x_true.min()), float(x_true.max())
        if x_max > x_min:
            x_true = (x_true - x_min) / (x_max - x_min)
        x_true = x_true.astype(np.float32)

        # --- CTF model parameters ---
        defocus_um = float(rng.uniform(1.0, 3.0))   # 1–3 µm defocus
        Cs_mm = 2.7                                   # spherical aberration (mm)
        V_kV = 300.0                                  # accelerating voltage (kV)
        pixel_size_A = 1.06                           # pixel size in Angstroms

        # Electron wavelength (relativistic)
        m_e = 9.10938e-31
        e_charge = 1.60218e-19
        c = 2.99792e8
        h = 6.62607e-34
        V = V_kV * 1e3
        lam_m = h / np.sqrt(2 * m_e * e_charge * V * (1 + e_charge * V / (2 * m_e * c ** 2)))
        lam_A = lam_m * 1e10   # wavelength in Angstroms

        # Spatial frequency grid (1/Angstrom)
        fy = np.fft.fftfreq(H, d=pixel_size_A).astype(np.float32)
        fx = np.fft.fftfreq(W, d=pixel_size_A).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)
        s2 = FX ** 2 + FY ** 2   # |f|^2

        # CTF phase contrast transfer
        df_A = defocus_um * 1e4   # defocus in Angstroms
        Cs_A = Cs_mm * 1e7        # Cs in Angstroms
        chi = np.pi * lam_A * df_A * s2 - 0.5 * np.pi * Cs_A * lam_A ** 3 * s2 ** 2
        ctf = -np.sin(chi).astype(np.float32)

        # Apply CTF in Fourier domain
        X_fft = np.fft.fft2(x_true)
        y_ctf = np.real(np.fft.ifft2(X_fft * ctf)).astype(np.float32)

        # Shift to non-negative for Poisson sampling
        y_ctf -= float(y_ctf.min())

        # --- Low-dose Poisson noise (~10 e-/Å²) ---
        dose = float(rng.uniform(8.0, 12.0))   # electrons per Angstrom^2
        scale = dose * pixel_size_A ** 2        # electrons per pixel
        y_counts = rng.poisson(np.maximum(y_ctf * scale, 0)).astype(np.float32)
        y_meas = y_counts / (scale + 1e-8)

        # Normalise measurement
        y_min, y_max = float(y_meas.min()), float(y_meas.max())
        if y_max > y_min:
            y_meas = (y_meas - y_min) / (y_max - y_min)
        y_meas = y_meas.astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "cryo_em",
                "defocus_um": float(defocus_um),
                "dose_e_per_A2": float(dose),
                "pixel_size_A": float(pixel_size_A),
            },
        })

    return samples


def generate_cryo_et_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """Generate synthetic cryo-electron tomography (cellular tomography) phantom.

    Simulates a 2D slice of a cellular tomogram with missing-wedge corruption.
    Ground truth contains membranes (ellipsoidal shells), ribosomes (small discs),
    and mitochondria (larger ellipsoids).

    Forward model:
      - Take Fourier transform of x_true
      - Zero out angular wedge ±60° from vertical (missing wedge)
      - Add Gaussian noise (sigma ~0.05)

    Reference: Bharat & Bharat, Resolving macromolecular structures from
    electron cryo-tomography data using subtomogram averaging in RELION,
    Nat. Methods 2015.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # --- Ground truth: 2D slice of a cellular tomogram ---
        x_true = np.zeros((H, W), dtype=np.float32)
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]

        # Membranes: ellipsoidal shells (thin ring regions)
        n_membranes = int(rng.integers(2, 5))
        for _ in range(n_membranes):
            m_cy = int(rng.integers(H // 4, 3 * H // 4))
            m_cx = int(rng.integers(W // 4, 3 * W // 4))
            ry = float(rng.uniform(H // 8, H // 4))
            rx = float(rng.uniform(W // 8, W // 4))
            dist = ((Y - m_cy) / ry) ** 2 + ((X - m_cx) / rx) ** 2
            shell = (dist >= 0.75) & (dist <= 1.0)
            x_true[shell] = float(rng.uniform(0.6, 0.9))

        # Ribosomes: small dense discs
        n_ribosomes = int(rng.integers(5, 12))
        for _ in range(n_ribosomes):
            r_cy = int(rng.integers(0, H))
            r_cx = int(rng.integers(0, W))
            r_rad = float(rng.uniform(1.5, 3.5))
            disc = np.sqrt((Y - r_cy) ** 2 + (X - r_cx) ** 2)
            x_true[disc <= r_rad] = float(rng.uniform(0.7, 1.0))

        # Mitochondria: larger ellipsoids
        n_mito = int(rng.integers(1, 3))
        for _ in range(n_mito):
            m_cy = int(rng.integers(H // 5, 4 * H // 5))
            m_cx = int(rng.integers(W // 5, 4 * W // 5))
            ry = float(rng.uniform(H // 6, H // 3))
            rx = float(rng.uniform(W // 10, W // 6))
            ellipse = ((Y - m_cy) / ry) ** 2 + ((X - m_cx) / rx) ** 2
            x_true[ellipse <= 1.0] = float(rng.uniform(0.3, 0.6))

        # Normalise ground truth to [0, 1]
        x_min, x_max = float(x_true.min()), float(x_true.max())
        if x_max > x_min:
            x_true = (x_true - x_min) / (x_max - x_min)
        x_true = x_true.astype(np.float32)

        # --- Missing-wedge corruption in Fourier space ---
        X_fft = np.fft.fftshift(np.fft.fft2(x_true))

        # Build missing-wedge mask: zero out ±60° from vertical axis
        # Vertical axis: ky direction; missing wedge spans |kx/ky| > tan(60°)
        fy = np.fft.fftshift(np.fft.fftfreq(H)).astype(np.float32)
        fx = np.fft.fftshift(np.fft.fftfreq(W)).astype(np.float32)
        FX, FY = np.meshgrid(fx, fy)
        # Missing wedge: angles within 60° of horizontal (i.e., |FY| < |FX| * tan(30°))
        missing_wedge_angle_deg = 60.0
        threshold = np.tan(np.deg2rad(90.0 - missing_wedge_angle_deg))
        missing_mask = np.abs(FY) < np.abs(FX) * threshold
        X_fft[missing_mask] = 0.0

        y_meas = np.real(np.fft.ifft2(np.fft.ifftshift(X_fft))).astype(np.float32)

        # Add Gaussian noise
        noise_sigma = 0.05
        y_meas += rng.normal(0.0, noise_sigma, size=(H, W)).astype(np.float32)

        # Normalise measurement to [0, 1]
        y_min, y_max = float(y_meas.min()), float(y_meas.max())
        if y_max > y_min:
            y_meas = (y_meas - y_min) / (y_max - y_min)
        y_meas = y_meas.astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "cryo_et",
                "tilt_range_deg": 60.0,
                "missing_wedge_deg": missing_wedge_angle_deg,
                "pixel_size_A": 10.0,
            },
        })

    return samples


def generate_ct_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list:
    """Generate synthetic X-ray CT Shepp-Logan-style phantom with Poisson sinogram noise.

    Simulates fan-beam / parallel-beam CT acquisition:
      - x_true: 64×64 float32 Shepp-Logan-style phantom.  A large oval body outline
        (ellipse) plus inner ellipses representing organs (liver, lung regions, bone
        cortex), normalised to [0, 1] representing linear attenuation coefficients.
      - y: Sinogram via Radon transform (128 angles from 0 to π).  Beer-Lambert law
        applied with I₀=1e5 photons, Poisson noise added, then log-normalised to [0, 1].
      - H_ideal: identity (the Radon operator is implicit in the acquisition model).
      - metadata: dict with keys "modality", "n_angles", "detector_pixels",
        "source_to_detector_mm".

    Uses scipy.ndimage's Radon transform when available; falls back to a simple
    column-sum forward projection loop for environments without skimage/scipy.special.

    Reference: Shepp & Logan, "The Fourier reconstruction of a head section",
    IEEE Trans. Nucl. Sci. 1974.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    n_angles = 128
    I0 = 1e5  # photon count for Beer-Lambert noise

    samples = []

    for i in range(n_samples):
        # ── Build Shepp-Logan-style phantom ───────────────────────────────
        x_true = np.zeros((H, W), dtype=np.float32)
        Y, X = np.ogrid[:H, :W]
        cy, cx = H / 2.0, W / 2.0

        # Large body ellipse (outer boundary)
        ry_body = H * 0.45
        rx_body = W * 0.37
        body = ((Y - cy) / ry_body) ** 2 + ((X - cx) / rx_body) ** 2 <= 1.0
        x_true[body] = 0.20 + float(rng.uniform(-0.02, 0.02))

        # Skull / cortical bone shell (thin ring just inside body)
        ry_sk = ry_body * 0.92
        rx_sk = rx_body * 0.92
        skull_out = ((Y - cy) / ry_body) ** 2 + ((X - cx) / rx_body) ** 2 <= 1.0
        skull_in  = ((Y - cy) / ry_sk)   ** 2 + ((X - cx) / rx_sk)   ** 2 <= 1.0
        shell = skull_out & ~skull_in
        x_true[shell] = 0.50 + float(rng.uniform(-0.03, 0.03))

        # Liver-like region (large, upper-right)
        liver_cy = cy - H * 0.05 + float(rng.uniform(-H * 0.03, H * 0.03))
        liver_cx = cx + W * 0.12 + float(rng.uniform(-W * 0.03, W * 0.03))
        ry_l = H * 0.22 + float(rng.uniform(-H * 0.02, H * 0.02))
        rx_l = W * 0.18 + float(rng.uniform(-W * 0.02, W * 0.02))
        liver = ((Y - liver_cy) / ry_l) ** 2 + ((X - liver_cx) / rx_l) ** 2 <= 1.0
        liver &= skull_in
        x_true[liver] = 0.35 + float(rng.uniform(-0.02, 0.02))

        # Lung-left (low attenuation — air-filled)
        lung_l_cy = cy + float(rng.uniform(-H * 0.02, H * 0.02))
        lung_l_cx = cx - W * 0.20 + float(rng.uniform(-W * 0.02, W * 0.02))
        ry_ll = H * 0.18 + float(rng.uniform(-H * 0.02, H * 0.02))
        rx_ll = W * 0.12 + float(rng.uniform(-W * 0.01, W * 0.01))
        lung_l = ((Y - lung_l_cy) / ry_ll) ** 2 + ((X - lung_l_cx) / rx_ll) ** 2 <= 1.0
        lung_l &= skull_in
        x_true[lung_l] = 0.05 + float(rng.uniform(-0.01, 0.01))

        # Lung-right
        lung_r_cy = cy + float(rng.uniform(-H * 0.02, H * 0.02))
        lung_r_cx = cx + W * 0.18 + float(rng.uniform(-W * 0.02, W * 0.02))
        ry_lr = H * 0.17 + float(rng.uniform(-H * 0.02, H * 0.02))
        rx_lr = W * 0.11 + float(rng.uniform(-W * 0.01, W * 0.01))
        lung_r = ((Y - lung_r_cy) / ry_lr) ** 2 + ((X - lung_r_cx) / rx_lr) ** 2 <= 1.0
        lung_r &= skull_in
        x_true[lung_r] = 0.05 + float(rng.uniform(-0.01, 0.01))

        # Spine / vertebral bone (small, high attenuation, centre-posterior)
        spine_cy = cy + H * 0.28 + float(rng.uniform(-H * 0.02, H * 0.02))
        spine_cx = cx + float(rng.uniform(-W * 0.01, W * 0.01))
        ry_sp = H * 0.06
        rx_sp = W * 0.05
        spine = ((Y - spine_cy) / ry_sp) ** 2 + ((X - spine_cx) / rx_sp) ** 2 <= 1.0
        spine &= skull_in
        x_true[spine] = 0.65 + float(rng.uniform(-0.03, 0.03))

        # Normalise to [0, 1]
        x_min, x_max = float(x_true.min()), float(x_true.max())
        if x_max > x_min:
            x_true = (x_true - x_min) / (x_max - x_min)
        x_true = x_true.astype(np.float32)

        # ── Radon / sinogram forward model ───────────────────────────────
        angles_deg = np.linspace(0.0, 180.0, n_angles, endpoint=False)
        detector_pixels = max(H, W)

        try:
            from skimage.transform import radon
            sinogram = radon(x_true.astype(np.float64), theta=angles_deg, circle=True)
            # sinogram shape: (detector_pixels_skimage, n_angles)
        except ImportError:
            # Fallback: simple parallel-beam forward projection via rotation
            from scipy.ndimage import rotate as ndrotate
            sinogram = np.zeros((detector_pixels, n_angles), dtype=np.float64)
            pad = (detector_pixels - H) // 2
            for j, ang in enumerate(angles_deg):
                rot = ndrotate(x_true.astype(np.float64), -ang, reshape=False, order=1)
                proj = rot.sum(axis=0)
                if len(proj) < detector_pixels:
                    sinogram[:len(proj), j] = proj
                else:
                    sinogram[:, j] = proj[:detector_pixels]

        # Beer-Lambert: I = I0 * exp(-sinogram); add Poisson noise
        sinogram_float = sinogram.astype(np.float64)
        intensity = I0 * np.exp(-sinogram_float)
        noisy = rng.poisson(np.maximum(intensity, 1)).astype(np.float64)

        # Log back to line-integral domain
        log_sino = -np.log(np.maximum(noisy, 1) / I0)

        # Normalise measurement to [0, 1]
        y_min, y_max = float(log_sino.min()), float(log_sino.max())
        if y_max > y_min:
            log_sino = (log_sino - y_min) / (y_max - y_min)
        y_meas = log_sino.astype(np.float32)

        # H_ideal: identity (Radon operator is implicit)
        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "ct",
                "n_angles": n_angles,
                "detector_pixels": detector_pixels,
                "source_to_detector_mm": 1000.0,
            },
        })

    return samples


def generate_ct_fluorescence_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list:
    """Generate synthetic X-ray Fluorescence CT (XRF-CT) phantom.

    Simulates X-ray fluorescence computed tomography:
      - x_true: 64×64 float32 map of fluorescent marker distribution
        (e.g., gold nanoparticles or iodine K-edge) — ellipsoidal clusters of
        high-fluorescence regions on a low background, normalised to [0, 1].
      - y: Corrupted measurement — Poisson noise (lambda=50 counts) plus
        Compton scatter background (uniform ~5 counts), normalised to [0, 1].
      - H_ideal: identity (the fluorescence emission / detection operator is
        implicit in the acquisition model).
      - metadata: dict with keys "modality", "fluorescent_element",
        "excitation_keV", "pixel_size_um".

    Reference: Larsson et al., "A framework for quantitative X-ray fluorescence
    CT reconstruction", Phys. Med. Biol. 65 (2020).
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)

    # XRF-CT element and energy parameters (varied per sample for diversity)
    _elements = ["Au", "I", "Gd", "Ba"]
    _excitation_keV = {"Au": 80.7, "I": 33.2, "Gd": 50.2, "Ba": 37.4}
    _pixel_size_um = 50.0

    samples = []

    for i in range(n_samples):
        element = _elements[i % len(_elements)]

        # ── Build fluorescent marker distribution ────────────────────────
        x_true = np.zeros((H, W), dtype=np.float32)
        Y, X = np.ogrid[:H, :W]
        cy, cx = H / 2.0, W / 2.0

        # Low uniform tissue background
        background_level = float(rng.uniform(0.02, 0.06))
        x_true[:, :] = background_level

        # 2–4 ellipsoidal clusters of high fluorescence
        n_clusters = int(rng.integers(2, 5))
        for _ in range(n_clusters):
            # Random cluster centre (within inner 60% of FOV)
            c_y = cy + float(rng.uniform(-H * 0.25, H * 0.25))
            c_x = cx + float(rng.uniform(-W * 0.25, W * 0.25))
            ry = float(rng.uniform(H * 0.06, H * 0.18))
            rx = float(rng.uniform(W * 0.06, W * 0.18))
            amplitude = float(rng.uniform(0.6, 1.0))
            mask = ((Y - c_y) / ry) ** 2 + ((X - c_x) / rx) ** 2 <= 1.0
            x_true[mask] = np.maximum(x_true[mask], amplitude)

        # Normalise to [0, 1]
        x_min, x_max = float(x_true.min()), float(x_true.max())
        if x_max > x_min:
            x_true = (x_true - x_min) / (x_max - x_min)
        x_true = x_true.astype(np.float32)

        # ── Forward model: fluorescence emission with noise ───────────────
        # Scale to expected photon counts (lambda ~ 50 at peak)
        expected_counts = x_true * 50.0

        # Poisson noise on signal
        signal_noisy = rng.poisson(np.maximum(expected_counts, 1e-6)).astype(np.float32)

        # Compton scatter background: uniform ~5 counts
        compton_bg = float(rng.uniform(4.0, 6.0))
        scatter = rng.poisson(
            np.full((H, W), compton_bg, dtype=np.float64)
        ).astype(np.float32)

        y_raw = signal_noisy + scatter

        # Normalise measurement to [0, 1]
        y_min, y_max = float(y_raw.min()), float(y_raw.max())
        if y_max > y_min:
            y_raw = (y_raw - y_min) / (y_max - y_min)
        y_meas = y_raw.astype(np.float32)

        # H_ideal: identity
        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "ct_fluorescence",
                "fluorescent_element": element,
                "excitation_keV": _excitation_keV[element],
                "pixel_size_um": _pixel_size_um,
            },
        })

    return samples


def generate_confocal_3d_phantom(
    n_samples: int = 10,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Confocal 3D microscopy phantom for optical sectioning deconvolution.

    Generates fluorescent cell phantoms with organelles (nucleus, mitochondria,
    actin filaments) in 3D. Forward model: convolution with asymmetric 3D PSF
    (lateral FWHM ~200 nm, axial FWHM ~600 nm at 63×/1.4NA, 488nm).
    Returns 2D max-projection for benchmark evaluation.

    Reference: Born & Wolf, Principles of Optics; Conchello & Lichtman, Nat. Methods 2005.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []
    Z = 16  # z-slices

    for i in range(n_samples):
        # 3D fluorescence volume
        vol = np.zeros((Z, H, W), dtype=np.float32)

        cy, cx, cz = H // 2, W // 2, Z // 2

        # Nucleus (bright, spherical)
        ry, rx, rz = H // 5, W // 5, Z // 3
        for z in range(Z):
            Y, X = np.ogrid[:H, :W]
            nuc = ((Y-cy)/ry)**2 + ((X-cx)/rx)**2 + ((z-cz)/rz)**2 <= 1.0
            vol[z][nuc] = float(rng.uniform(0.6, 0.9))

        # Mitochondria network (tubular)
        n_mito = int(rng.integers(4, 8))
        for _ in range(n_mito):
            mz = int(rng.integers(Z // 4, 3 * Z // 4))
            my = int(rng.integers(cy - ry, cy + ry))
            mx = int(rng.integers(cx - rx, cx + rx))
            length = int(rng.integers(8, 20))
            angle = float(rng.uniform(0, np.pi))
            for t in range(length):
                gy = int(my + t * np.cos(angle))
                gx = int(mx + t * np.sin(angle))
                if 0 <= gy < H and 0 <= gx < W:
                    vol[mz, gy, gx] = float(rng.uniform(0.4, 0.7))

        # Actin filaments (thin, bright)
        n_actin = int(rng.integers(3, 6))
        for _ in range(n_actin):
            az = int(rng.integers(0, Z))
            ay0 = int(rng.integers(0, H))
            ax0 = int(rng.integers(0, W))
            angle = float(rng.uniform(0, np.pi))
            for t in range(W // 2):
                gy = int(ay0 + t * np.cos(angle))
                gx = int(ax0 + t * np.sin(angle))
                if 0 <= gy < H and 0 <= gx < W:
                    vol[az, gy, gx] = float(rng.uniform(0.7, 1.0))

        # Ground truth: max projection
        x_true = vol.max(axis=0).astype(np.float32)

        # Forward model: 3D PSF convolution (asymmetric)
        sigma_lat = float(rng.uniform(1.0, 2.0))  # lateral
        sigma_ax = float(rng.uniform(3.0, 5.0))   # axial (worse)
        vol_blurred = gaussian_filter(vol, sigma=[sigma_ax, sigma_lat, sigma_lat])

        # Shot noise
        photon_count = float(rng.uniform(50, 200))
        vol_counts = vol_blurred * photon_count
        vol_noisy = rng.poisson(np.maximum(vol_counts, 0)).astype(np.float32) / photon_count

        # Max projection of blurred+noisy volume
        y_meas = vol_noisy.max(axis=0).astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y_meas,
            "H_ideal": H_ideal,
            "metadata": {
                "n_z_slices": Z,
                "sigma_lateral": float(sigma_lat),
                "sigma_axial": float(sigma_ax),
                "photon_count": float(photon_count),
            },
        })

    return samples


def generate_cup_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Compressed Ultrafast Photography (CUP) phantom for single-shot femtosecond imaging.

    Simulates a dynamic scene of a light pulse propagating through a scene.
    x_true: 64x64 float32 image with a Gaussian intensity profile moving across the frame.
    y: Compressed measurement via random binary mask (50% compression) summed along
       temporal dimension, with added Gaussian noise (sigma=0.05).
    H_ideal: identity matrix.
    metadata: dict with modality, temporal_frames, compression_ratio, frame_rate_fps.

    Reference: Gao et al., Nature 2014 (single-shot compressed ultrafast photography).
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []
    T = 10  # temporal frames
    compression_ratio = 0.5

    for i in range(n_samples):
        # Dynamic scene: Gaussian light pulse propagating across the frame
        x_true = np.zeros((H, W), dtype=np.float32)

        # Pulse position moves from left to right across the frame
        pulse_center_x = int(rng.integers(W // 4, 3 * W // 4))
        pulse_center_y = int(rng.integers(H // 4, 3 * H // 4))
        sigma_pulse = float(rng.uniform(H / 12, H / 6))

        Y, X = np.ogrid[:H, :W]
        # Final frame: Gaussian intensity profile at the pulse center
        x_true = np.exp(
            -((Y - pulse_center_y) ** 2 + (X - pulse_center_x) ** 2) / (2 * sigma_pulse ** 2)
        ).astype(np.float32)

        # Simulate T temporal frames of the pulse propagating left to right
        speed_x = float(rng.uniform(W / (2 * T), W / T))
        speed_y = float(rng.uniform(-H / (4 * T), H / (4 * T)))

        frames = []
        for t in range(T):
            cx = pulse_center_x - speed_x * (T - 1 - t)
            cy = pulse_center_y - speed_y * (T - 1 - t)
            frame = np.exp(
                -((Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma_pulse ** 2)
            ).astype(np.float32)
            frames.append(frame)

        # Compressed measurement: random binary mask (50% compression) summed over time
        mask = (rng.random((T, H, W)) < compression_ratio).astype(np.float32)
        video = np.stack(frames, axis=0)  # (T, H, W)
        compressed = (video * mask).sum(axis=0).astype(np.float32)

        # Add Gaussian noise
        noise = rng.normal(0, 0.05, size=(H, W)).astype(np.float32)
        y = (compressed + noise).astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "cup",
                "temporal_frames": T,
                "compression_ratio": compression_ratio,
                "frame_rate_fps": 1.0e13,  # 10 trillion fps (femtosecond scale)
            },
        })

    return samples


def generate_dark_field_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Dark-field microscopy phantom for DIC/dark-field optical microscopy.

    Simulates scattered light from sub-wavelength particles: sparse bright spots
    on a dark background, as seen in dark-field optical microscopy.
    x_true: 64x64 float32 image with sparse Gaussian spots (intensity 0.8-1.0)
            on a dark background (~0.02).
    y: Noisy dark-field measurement: Poisson noise (scale 100) + Gaussian noise (sigma=0.02).
    H_ideal: identity matrix.
    metadata: dict with modality, particle_size_nm, wavelength_nm, NA.

    Reference: Siedentopf & Zsigmondy, Ann. Physik 1902 (dark-field illumination).
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # Dark background (~0.02 intensity)
        x_true = np.full((H, W), 0.02, dtype=np.float32)

        # Sparse bright spots: 5-15 sub-wavelength particles
        n_particles = int(rng.integers(5, 16))
        Y_grid, X_grid = np.ogrid[:H, :W]

        for _ in range(n_particles):
            # Random particle position
            cy = float(rng.uniform(0, H))
            cx = float(rng.uniform(0, W))
            # Small Gaussian spot (~1-2 pixel sigma to simulate sub-wavelength)
            sigma_spot = float(rng.uniform(0.8, 2.0))
            # Intensity between 0.8 and 1.0
            intensity = float(rng.uniform(0.8, 1.0))
            spot = intensity * np.exp(
                -((Y_grid - cy) ** 2 + (X_grid - cx) ** 2) / (2 * sigma_spot ** 2)
            ).astype(np.float32)
            x_true = np.maximum(x_true, spot)

        x_true = x_true.astype(np.float32)

        # Noisy dark-field measurement: Poisson + Gaussian noise
        # Scale by 100, apply Poisson, divide by 100
        scaled = x_true * 100.0
        poisson_noisy = rng.poisson(scaled).astype(np.float32) / 100.0
        gaussian_noise = rng.normal(0, 0.02, size=(H, W)).astype(np.float32)
        y = (poisson_noisy + gaussian_noise).astype(np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        # Realistic metadata for dark-field microscopy
        particle_size_nm = float(rng.uniform(50, 200))
        wavelength_nm = float(rng.choice([488, 532, 561, 647]))
        NA = float(rng.uniform(0.8, 1.4))

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "dark_field",
                "particle_size_nm": particle_size_nm,
                "wavelength_nm": wavelength_nm,
                "NA": NA,
            },
        })

    return samples


def generate_dexa_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    DEXA (Dual-Energy X-ray Absorptiometry) bone density imaging phantom.

    Simulates bone mineral density (BMD) mapping from dual-energy X-ray
    measurements using the Beer-Lambert two-component attenuation model.
    x_true: 64x64 float32 bone mineral density map — central bone region
            (oval, BMD ~0.8-1.0), surrounding soft tissue (~0.3-0.5),
            background ~0.05.
    y: Noisy DEXA measurement: apply Beer-Lambert (2 energy channels combined
       linearly), add Poisson noise (scale factor 1e4), normalize to [0, 1].
    H_ideal: identity matrix.
    metadata: dict with keys modality, low_energy_keV, high_energy_keV,
              detector_spacing_mm.

    Reference: Blake & Fogelman, J. Clin. Densitom. 1997.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    # Mass attenuation coefficients (cm^2/g) at low (~40 keV) and high (~70 keV) energy
    mu_bone_low = 0.40
    mu_bone_high = 0.22
    mu_soft_low = 0.25
    mu_soft_high = 0.18

    for i in range(n_samples):
        # Background: very low BMD (~0.05)
        x_true = np.full((H, W), 0.05, dtype=np.float32)

        Y_grid, X_grid = np.mgrid[:H, :W]
        cy = H / 2.0 + float(rng.uniform(-H * 0.1, H * 0.1))
        cx = W / 2.0 + float(rng.uniform(-W * 0.1, W * 0.1))

        # Central bone region (oval, BMD ~0.8-1.0)
        ry_bone = float(rng.uniform(H * 0.12, H * 0.22))
        rx_bone = float(rng.uniform(W * 0.10, W * 0.18))
        bone_bmd = float(rng.uniform(0.8, 1.0))
        bone_ellipse = ((Y_grid - cy) / ry_bone) ** 2 + ((X_grid - cx) / rx_bone) ** 2
        bone_mask = bone_ellipse <= 1.0
        x_true[bone_mask] = bone_bmd

        # Surrounding soft tissue ring (~0.3-0.5)
        ry_soft = ry_bone * float(rng.uniform(1.5, 2.2))
        rx_soft = rx_bone * float(rng.uniform(1.5, 2.2))
        soft_bmd = float(rng.uniform(0.3, 0.5))
        soft_ellipse = ((Y_grid - cy) / ry_soft) ** 2 + ((X_grid - cx) / rx_soft) ** 2
        soft_mask = (soft_ellipse <= 1.0) & (~bone_mask)
        x_true[soft_mask] = soft_bmd

        x_true = np.clip(x_true, 0.0, 1.0).astype(np.float32)

        # Compute soft tissue thickness map (complement of BMD)
        soft_tissue = np.where(bone_mask, 0.2, np.where(soft_mask, 0.6, 0.1)).astype(np.float32)

        # Beer-Lambert forward model: two energy channels combined linearly
        # p = mu_bone * x_bone + mu_soft * x_soft (projected attenuation)
        attn_low = (mu_bone_low * x_true + mu_soft_low * soft_tissue).astype(np.float32)
        attn_high = (mu_bone_high * x_true + mu_soft_high * soft_tissue).astype(np.float32)

        # Poisson noise with scale factor 1e4
        scale = 1e4
        counts_low = rng.poisson(scale * np.exp(-attn_low).astype(np.float64)).astype(np.float32)
        counts_high = rng.poisson(scale * np.exp(-attn_high).astype(np.float64)).astype(np.float32)

        # Linear combination of two energy channels
        y_combined = 0.6 * counts_low + 0.4 * counts_high

        # Normalize to [0, 1]
        y_min, y_max = float(y_combined.min()), float(y_combined.max())
        if y_max - y_min > 1e-8:
            y = ((y_combined - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        low_energy_keV = float(rng.uniform(38.0, 42.0))
        high_energy_keV = float(rng.uniform(68.0, 75.0))
        detector_spacing_mm = float(rng.uniform(0.5, 1.5))

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "dexa",
                "low_energy_keV": low_energy_keV,
                "high_energy_keV": high_energy_keV,
                "detector_spacing_mm": detector_spacing_mm,
            },
        })

    return samples


def generate_desi_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    DESI (Desorption Electrospray Ionization) mass spectrometry imaging phantom.

    Simulates lipid/metabolite spatial distribution in tissue sections.
    x_true: 64x64 float32 image of lipid/metabolite spatial distribution —
            ellipsoidal tissue regions with distinct chemical compositions
            (background ~0.1, regions ~0.6-1.0).
    y: Noisy MSI measurement: multiplicative lognormal noise (sigma=0.15)
       + Gaussian noise (sigma=0.05), clipped to [0, 1].
    H_ideal: identity matrix.
    metadata: dict with keys modality, mass_range_da, spatial_resolution_um, ion_mode.

    Reference: Takats et al., Science 2004 (DESI ionization).
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        # Background with low lipid/metabolite signal (~0.1)
        x_true = np.full((H, W), 0.1, dtype=np.float32)

        Y_grid, X_grid = np.mgrid[:H, :W]
        cy_center = H / 2.0
        cx_center = W / 2.0

        # 2-4 ellipsoidal tissue regions with distinct chemical compositions
        n_regions = int(rng.integers(2, 5))
        for _ in range(n_regions):
            # Random center offset from image center
            cy = cy_center + float(rng.uniform(-H * 0.25, H * 0.25))
            cx = cx_center + float(rng.uniform(-W * 0.25, W * 0.25))
            # Ellipse semi-axes
            ry = float(rng.uniform(H * 0.08, H * 0.25))
            rx = float(rng.uniform(W * 0.08, W * 0.25))
            # Intensity in range 0.6-1.0
            intensity = float(rng.uniform(0.6, 1.0))
            # Ellipsoidal mask
            ellipse = ((Y_grid - cy) / ry) ** 2 + ((X_grid - cx) / rx) ** 2
            mask = ellipse <= 1.0
            x_true[mask] = np.maximum(x_true[mask], intensity)

        x_true = x_true.astype(np.float32)

        # Noisy MSI measurement: multiplicative lognormal + additive Gaussian
        lognormal_noise = rng.lognormal(mean=0.0, sigma=0.15, size=(H, W)).astype(np.float32)
        gaussian_noise = rng.normal(0.0, 0.05, size=(H, W)).astype(np.float32)
        y = (x_true * lognormal_noise + gaussian_noise).astype(np.float32)
        y = np.clip(y, 0.0, 1.0)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        # Realistic metadata for DESI-MSI
        mass_range_da = [float(rng.uniform(50, 200)), float(rng.uniform(800, 1200))]
        spatial_resolution_um = float(rng.uniform(50, 200))
        ion_mode = rng.choice(["positive", "negative"])

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "desi",
                "mass_range_da": mass_range_da,
                "spatial_resolution_um": spatial_resolution_um,
                "ion_mode": str(ion_mode),
            },
        })

    return samples


def generate_dic_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    DIC (Differential Interference Contrast) microscopy phase phantom.

    Simulates the optical path difference (OPD) map of a biological cell and
    the corresponding DIC intensity measurement.

    x_true: 64x64 float32 phase image (optical path difference map) —
            nucleus (central ellipse, OPD ~0.8), cytoplasm (outer ellipse,
            OPD ~0.3-0.5), background ~0.0; normalized to [0, 1].
    y: DIC measurement: DIC converts phase gradients to intensity —
       shear gradient in x-direction [1, -1] kernel convolution, offset 0.5,
       Gaussian noise (sigma=0.05), clipped to [0, 1].
    H_ideal: identity matrix.
    metadata: dict with keys modality, shear_direction_deg, shear_amount_px,
              bias_retardance_rad.

    Reference: Mehta & Sheppard, Nat. Photonics 2009.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        Y_grid, X_grid = np.mgrid[:H, :W]
        cy = H / 2.0
        cx = W / 2.0

        # Nucleus: central ellipse with high OPD (~0.8)
        nuc_ry = float(rng.uniform(H * 0.10, H * 0.18))
        nuc_rx = float(rng.uniform(W * 0.10, W * 0.18))
        nuc_cy = cy + float(rng.uniform(-H * 0.05, H * 0.05))
        nuc_cx = cx + float(rng.uniform(-W * 0.05, W * 0.05))
        nuc_opd = float(rng.uniform(0.75, 0.85))
        nucleus_mask = ((Y_grid - nuc_cy) / nuc_ry) ** 2 + ((X_grid - nuc_cx) / nuc_rx) ** 2 <= 1.0

        # Cytoplasm: outer ellipse with moderate OPD (~0.3-0.5)
        cyt_ry = float(rng.uniform(H * 0.22, H * 0.35))
        cyt_rx = float(rng.uniform(W * 0.22, W * 0.35))
        cyt_opd = float(rng.uniform(0.3, 0.5))
        cytoplasm_mask = ((Y_grid - cy) / cyt_ry) ** 2 + ((X_grid - cx) / cyt_rx) ** 2 <= 1.0

        # Build OPD map: background 0, cytoplasm, nucleus
        x_true = np.zeros((H, W), dtype=np.float32)
        x_true[cytoplasm_mask] = cyt_opd
        x_true[nucleus_mask] = nuc_opd

        # Normalize to [0, 1]
        xmax = x_true.max()
        if xmax > 0:
            x_true = x_true / xmax
        x_true = x_true.astype(np.float32)

        # DIC forward model: apply shear gradient in x-direction
        # DIC shear kernel [1, -1] along x-axis (finite difference)
        dic_grad = np.zeros((H, W), dtype=np.float32)
        dic_grad[:, :-1] = x_true[:, 1:] - x_true[:, :-1]
        # Shift to intensity: offset 0.5 so gradient=0 → intensity=0.5
        y = dic_grad + 0.5
        # Add Gaussian noise
        noise = rng.normal(0.0, 0.05, size=(H, W)).astype(np.float32)
        y = (y + noise).astype(np.float32)
        y = np.clip(y, 0.0, 1.0)

        H_size = min(H * W, 2048)
        H_ideal = np.eye(H_size, dtype=np.float32)

        shear_direction_deg = 0.0  # x-direction shear
        shear_amount_px = 1.0
        bias_retardance_rad = float(rng.uniform(0.4, 0.6))  # ~pi/2 ± small

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "dic",
                "shear_direction_deg": shear_direction_deg,
                "shear_amount_px": shear_amount_px,
                "bias_retardance_rad": bias_retardance_rad,
            },
        })

    return samples


def generate_diffusion_mri_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Diffusion MRI DTI fractional anisotropy (FA) map phantom.

    Simulates a DTI FA map with white matter fiber tracts (FA ~0.7-0.9)
    and gray matter regions (FA ~0.1-0.3), with k-space undersampling
    forward model.

    x_true: 64x64 float32 FA map normalized to [0, 1] —
            white matter tracts (high FA ~0.7-0.9), gray matter (low FA ~0.1-0.3),
            background ~0.0.
    y: undersampled k-space reconstruction — every 4th k-space line sampled,
       complex Gaussian noise added in k-space, inverse FFT taken.
    H_ideal: 64x64 float32 identity matrix.
    metadata: dict with keys modality, acceleration_factor, acquisition_scheme.

    Reference: Behrens et al., MRM 2003; Merlet & Deriche, MRM 2013.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    for i in range(n_samples):
        Y_grid, X_grid = np.mgrid[:H, :W]
        cy = H / 2.0
        cx = W / 2.0

        # Build FA map: background ~0 (CSF / outside brain)
        fa_map = np.zeros((H, W), dtype=np.float32)

        # Gray matter: outer ellipse (FA ~0.1-0.3)
        gm_ry = float(rng.uniform(H * 0.30, H * 0.42))
        gm_rx = float(rng.uniform(W * 0.30, W * 0.42))
        gm_fa = float(rng.uniform(0.10, 0.30))
        gm_mask = ((Y_grid - cy) / gm_ry) ** 2 + ((X_grid - cx) / gm_rx) ** 2 <= 1.0
        fa_map[gm_mask] = gm_fa

        # White matter corpus callosum: horizontal band (FA ~0.7-0.9)
        cc_height = float(rng.uniform(H * 0.08, H * 0.14))
        cc_width = float(rng.uniform(W * 0.45, W * 0.65))
        cc_cy = cy + float(rng.uniform(-H * 0.05, H * 0.05))
        cc_fa = float(rng.uniform(0.70, 0.90))
        cc_mask = (
            (np.abs(Y_grid - cc_cy) < cc_height / 2.0) &
            (np.abs(X_grid - cx) < cc_width / 2.0)
        )
        fa_map[cc_mask] = cc_fa

        # White matter corticospinal tract: vertical band (FA ~0.65-0.85)
        cst_width = float(rng.uniform(W * 0.06, W * 0.10))
        cst_cx = cx + float(rng.uniform(-W * 0.10, W * 0.10))
        cst_height = float(rng.uniform(H * 0.40, H * 0.55))
        cst_fa = float(rng.uniform(0.65, 0.85))
        cst_mask = (
            (np.abs(X_grid - cst_cx) < cst_width / 2.0) &
            (np.abs(Y_grid - cy) < cst_height / 2.0)
        )
        fa_map[cst_mask] = cst_fa

        # Normalize to [0, 1]
        fa_max = fa_map.max()
        if fa_max > 0:
            x_true = fa_map / fa_max
        else:
            x_true = fa_map.copy()
        x_true = x_true.astype(np.float32)

        # k-space undersampling forward model
        # Sample every 4th line in k-space (acceleration factor = 4)
        acceleration_factor = 4
        kspace = np.fft.fft2(x_true.astype(np.complex64))
        kspace_undersampled = np.zeros_like(kspace)
        kspace_undersampled[::acceleration_factor, :] = kspace[::acceleration_factor, :]

        # Add complex Gaussian noise in k-space
        noise_std = float(rng.uniform(0.01, 0.03)) * np.abs(kspace).max()
        noise_real = rng.normal(0.0, noise_std, size=kspace.shape).astype(np.float32)
        noise_imag = rng.normal(0.0, noise_std, size=kspace.shape).astype(np.float32)
        kspace_undersampled += (noise_real + 1j * noise_imag)

        # Inverse FFT to get undersampled reconstruction
        y_complex = np.fft.ifft2(kspace_undersampled)
        y = np.abs(y_complex).astype(np.float32)
        y_max = y.max()
        if y_max > 0:
            y = y / y_max
        y = np.clip(y, 0.0, 1.0)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "diffusion_mri",
                "acceleration_factor": acceleration_factor,
                "acquisition_scheme": "DTI",
            },
        })

    return samples


def generate_digital_breast_tomo_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    Digital Breast Tomosynthesis (DBT) phantom with adipose tissue, glandular
    regions, and a small lesion/mass.

    Simulates limited-angle tomosynthesis: projects through 11 angles (-25° to
    +25°), adds Poisson noise, then back-projects (FBP) to produce y.

    x_true: 64×64 float32 breast phantom normalized to [0, 1] —
            adipose tissue background (~0.2), glandular regions (~0.6-0.8),
            small lesion/mass (~1.0).
    y: 64×64 float32 — FBP reconstruction from limited-angle projections with
       Poisson noise, exhibiting limited-angle artifacts.
    H_ideal: 64×64 float32 identity matrix.
    metadata: dict with keys modality, n_angles, angle_range_deg,
              dose_reduction_factor.

    Reference: Sechopoulos, Med. Phys. 2013; Sidky et al., Med. Phys. 2014.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    n_angles = 11
    angle_range_deg = 25.0
    angles_deg = np.linspace(-angle_range_deg, angle_range_deg, n_angles)
    angles_rad = np.deg2rad(angles_deg)

    for i in range(n_samples):
        Y_grid, X_grid = np.mgrid[:H, :W]
        cy = H / 2.0
        cx = W / 2.0

        # Adipose tissue background (fat, low attenuation ~0.15-0.25)
        phantom = np.full((H, W), float(rng.uniform(0.15, 0.25)), dtype=np.float32)

        # Breast boundary: semi-ellipse mask
        breast_ry = float(rng.uniform(H * 0.38, H * 0.46))
        breast_rx = float(rng.uniform(W * 0.38, W * 0.46))
        breast_mask = ((Y_grid - cy) / breast_ry) ** 2 + ((X_grid - cx) / breast_rx) ** 2 <= 1.0
        # Outside breast: air (0.0)
        phantom[~breast_mask] = 0.0

        # Glandular tissue regions (denser, higher attenuation ~0.55-0.80)
        n_glands = int(rng.integers(2, 5))
        for _ in range(n_glands):
            gy = float(rng.uniform(cy - breast_ry * 0.5, cy + breast_ry * 0.5))
            gx = float(rng.uniform(cx - breast_rx * 0.5, cx + breast_rx * 0.5))
            g_ry = float(rng.uniform(H * 0.06, H * 0.14))
            g_rx = float(rng.uniform(W * 0.06, W * 0.14))
            g_val = float(rng.uniform(0.55, 0.80))
            gland_mask = (
                (((Y_grid - gy) / g_ry) ** 2 + ((X_grid - gx) / g_rx) ** 2 <= 1.0) &
                breast_mask
            )
            phantom[gland_mask] = g_val

        # Small lesion/mass (high attenuation ~0.85-1.0)
        lesion_y = float(rng.uniform(cy - breast_ry * 0.35, cy + breast_ry * 0.35))
        lesion_x = float(rng.uniform(cx - breast_rx * 0.35, cx + breast_rx * 0.35))
        lesion_r = float(rng.uniform(H * 0.025, H * 0.05))
        lesion_val = float(rng.uniform(0.85, 1.0))
        lesion_mask = (
            (Y_grid - lesion_y) ** 2 + (X_grid - lesion_x) ** 2 <= lesion_r ** 2
        ) & breast_mask
        phantom[lesion_mask] = lesion_val

        # Normalize to [0, 1]
        ph_min, ph_max = phantom.min(), phantom.max()
        if ph_max - ph_min > 0:
            x_true = (phantom - ph_min) / (ph_max - ph_min)
        else:
            x_true = phantom.copy()
        x_true = x_true.astype(np.float32)

        # Limited-angle tomosynthesis forward model
        # Simple projection along each angle using shift-and-sum
        projections = []
        for theta in angles_rad:
            # Project: for each row, shift by tan(theta) * (row - cy)
            proj_image = np.zeros((H, W), dtype=np.float32)
            for row in range(H):
                shift = int(round(np.tan(theta) * (row - cy)))
                shifted_row = np.roll(x_true[row, :], shift)
                proj_image[row, :] = shifted_row
            # Sum along rows to get 1D projection, then tile back
            line_integral = proj_image.sum(axis=0)  # shape (W,)
            projections.append(line_integral)

        projections = np.array(projections, dtype=np.float32)  # (n_angles, W)

        # Add Poisson noise (dose reduction)
        dose_factor = float(rng.uniform(0.3, 0.7))
        I0 = 1e4 * dose_factor
        projections_noisy = rng.poisson(
            np.maximum(I0 * (1.0 - projections), 1.0)
        ).astype(np.float32)
        projections_noisy = (I0 - projections_noisy) / (I0 + 1e-6)
        projections_noisy = np.clip(projections_noisy, 0.0, 1.0)

        # Back-projection (FBP approximation): shift back and average
        backproj = np.zeros((H, W), dtype=np.float32)
        for k, theta in enumerate(angles_rad):
            line = projections_noisy[k, :]  # shape (W,)
            bp_image = np.tile(line, (H, 1))  # broadcast to (H, W)
            for row in range(H):
                shift = int(round(np.tan(theta) * (row - cy)))
                bp_image[row, :] = np.roll(line, -shift)
            backproj += bp_image
        backproj /= n_angles

        # Normalize y to [0, 1]
        y_min, y_max = backproj.min(), backproj.max()
        if y_max - y_min > 0:
            y = (backproj - y_min) / (y_max - y_min)
        else:
            y = backproj.copy()
        y = np.clip(y, 0.0, 1.0).astype(np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "digital_breast_tomo",
                "n_angles": n_angles,
                "angle_range_deg": angle_range_deg,
                "dose_reduction_factor": dose_factor,
            },
        })

    return samples


def generate_dna_paint_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape: Optional[Tuple[int, ...]] = None,
) -> list[dict]:
    """
    DNA-PAINT super-resolution phantom with stochastic blinking and PSF model.

    Creates a 64×64 float32 DNA-PAINT super-resolution target: sparse emitter
    positions arranged in DNA nanostructure patterns (DNA origami grid with
    5-20 nm spacing, represented as point spread functions at sub-pixel emitter
    locations).

    Applies stochastic blinking forward model: Poisson-sampled photon counts per
    blinking event, Gaussian PSF (sigma ~1.5 pixels at diffraction limit),
    accumulated over multiple frames with random on/off blinking states.

    x_true: 64×64 float32, normalized [0,1] — ground truth emitter density map.
    y: 64×64 float32 — widefield diffraction-limited accumulation image.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with keys modality, n_frames, photons_per_blinking, psf_sigma_px.

    Reference: Jungmann et al., Nat. Methods 2014; Schnitzbauer et al., Nat. Protocols 2017.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    n_frames = 200
    psf_sigma_px = 1.5
    photons_per_blinking = 300
    # DNA-PAINT docking strand density: fraction of pixels with emitters
    emitter_density = 0.015

    for i in range(n_samples):
        # --- Ground truth emitter density map (DNA origami grid pattern) ---
        x_true = np.zeros((H, W), dtype=np.float32)

        # DNA origami grid: place clusters of emitters in a regular grid
        # Grid spacing ~8-12 pixels (representing 5-20 nm at typical SR scales)
        grid_spacing = int(rng.integers(8, 13))
        cluster_radius = float(rng.uniform(1.0, 2.5))

        Y_grid, X_grid = np.mgrid[:H, :W]

        # Number of emitters drawn from a Poisson process at each grid node
        n_emitters_total = max(1, int(rng.poisson(emitter_density * H * W)))

        for _ in range(n_emitters_total):
            # Sub-pixel emitter location within the grid
            ey = float(rng.uniform(cluster_radius, H - cluster_radius))
            ex = float(rng.uniform(cluster_radius, W - cluster_radius))
            # Snap to nearest grid node with small random offset (~DNA origami jitter)
            ey_node = round(ey / grid_spacing) * grid_spacing
            ex_node = round(ex / grid_spacing) * grid_spacing
            jitter_y = float(rng.normal(0, 0.5))
            jitter_x = float(rng.normal(0, 0.5))
            ey_final = float(np.clip(ey_node + jitter_y, 0, H - 1))
            ex_final = float(np.clip(ex_node + jitter_x, 0, W - 1))

            # Add emitter as a delta at integer pixel (sub-pixel accuracy encoded in density)
            iy = int(round(ey_final))
            ix = int(round(ex_final))
            if 0 <= iy < H and 0 <= ix < W:
                x_true[iy, ix] += 1.0

        # Normalize emitter density map to [0, 1]
        if x_true.max() > 0:
            x_true = x_true / x_true.max()

        # --- Stochastic blinking forward model ---
        # Accumulate photon counts over n_frames with random on/off blinking
        accumulated = np.zeros((H, W), dtype=np.float64)
        emitter_positions = np.argwhere(x_true > 0.01)

        # Per-emitter blinking probability (imager strand binding kinetics)
        k_on = float(rng.uniform(0.05, 0.15))   # binding rate per frame
        k_off = float(rng.uniform(0.7, 0.9))    # unbinding rate per frame

        for _frame in range(n_frames):
            frame_image = np.zeros((H, W), dtype=np.float64)
            for pos in emitter_positions:
                iy, ix = int(pos[0]), int(pos[1])
                # Stochastic blinking: on-state determined by binding kinetics
                if rng.random() < k_on:
                    # Poisson-sampled photon count for this blinking event
                    n_photons = int(rng.poisson(photons_per_blinking * float(x_true[iy, ix])))
                    if n_photons > 0:
                        # Place photons as a delta, PSF applied after accumulation
                        frame_image[iy, ix] += n_photons
            # Add Poisson background noise (camera dark counts + autofluorescence)
            bg_level = float(rng.uniform(2.0, 8.0))
            frame_image += rng.poisson(bg_level, size=(H, W)).astype(np.float64)
            accumulated += frame_image

        # Apply Gaussian PSF (diffraction-limited widefield accumulation)
        y_raw = gaussian_filter(accumulated, sigma=psf_sigma_px)

        # Normalize y to [0, 1]
        y_min = y_raw.min()
        y_max = y_raw.max()
        if y_max > y_min:
            y = ((y_raw - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "dna_paint",
                "n_frames": n_frames,
                "photons_per_blinking": photons_per_blinking,
                "psf_sigma_px": psf_sigma_px,
            },
        })

    return samples


def generate_doppler_ultrasound_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Doppler ultrasound flow phantom with parabolic Poiseuille flow and speckle noise.

    Creates a 64x64 float32 blood flow velocity map: background tissue (velocity ~0),
    vessel lumen with parabolic Poiseuille flow profile (v_max ~0.5-1.0 m/s), vessel
    wall boundaries.

    Applies Doppler ultrasound forward model: Doppler frequency shift proportional to
    flow velocity, multiplicative Rayleigh speckle noise, simulates aliasing at the
    Nyquist limit.

    x_true: 64x64 float32, normalized [0,1] — blood flow velocity map.
    y: 64x64 float32 — noisy Doppler measurement with speckle.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, prf_hz, beam_angle_deg, vessel_diameter_mm.

    Reference: Evans & McDicken, Doppler Ultrasound, 2nd ed., 2000.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    # Doppler acquisition parameters
    prf_hz = 5000.0           # Pulse repetition frequency (Hz)
    beam_angle_deg = 60.0     # Beam-to-flow angle (degrees)
    cos_theta = float(np.cos(np.deg2rad(beam_angle_deg)))

    for i in range(n_samples):
        # --- Ground truth velocity map (Poiseuille parabolic flow in vessel) ---
        x_true = np.zeros((H, W), dtype=np.float32)

        # Vessel parameters: random centre, orientation, and radius
        cx = float(rng.uniform(0.25 * W, 0.75 * W))
        cy = float(rng.uniform(0.25 * H, 0.75 * H))
        vessel_radius_px = float(rng.uniform(0.08 * min(H, W), 0.18 * min(H, W)))
        vessel_diameter_mm = vessel_radius_px * 0.3  # ~0.3 mm/pixel typical
        v_max = float(rng.uniform(0.5, 1.0))          # peak velocity m/s

        # Vessel orientation angle
        angle_deg = float(rng.uniform(0.0, 360.0))
        angle_rad = np.deg2rad(angle_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))

        Y_grid, X_grid = np.mgrid[:H, :W]
        # Distance from vessel centreline (rotated coordinate)
        dx = X_grid.astype(np.float32) - cx
        dy = Y_grid.astype(np.float32) - cy
        r_perp = np.abs(-dx * sin_a + dy * cos_a)  # perpendicular distance to axis

        # Parabolic Poiseuille profile: v(r) = v_max * (1 - (r/R)^2) inside vessel
        inside = r_perp < vessel_radius_px
        v_profile = np.where(
            inside,
            v_max * (1.0 - (r_perp / vessel_radius_px) ** 2),
            0.0,
        ).astype(np.float32)

        # Vessel wall: thin ring with slightly elevated velocity (~0.05 m/s)
        wall = (r_perp >= vessel_radius_px) & (r_perp < vessel_radius_px + 1.5)
        v_profile[wall] = float(rng.uniform(0.02, 0.06))

        # Normalize velocity map to [0, 1]
        if v_profile.max() > 0:
            x_true = (v_profile / v_profile.max()).astype(np.float32)
        else:
            x_true = v_profile

        # --- Doppler forward model ---
        # Doppler frequency shift: f_d = 2 * v * cos(theta) * f0 / c
        # Represent as normalized Doppler signal proportional to velocity
        doppler_signal = x_true * cos_theta  # Doppler-weighted velocity projection

        # Nyquist aliasing: wrap aliased velocities (v > v_nyquist maps to negative)
        v_nyquist = 0.7  # normalized Nyquist limit
        doppler_signal = np.where(
            doppler_signal > v_nyquist,
            doppler_signal - 2.0 * v_nyquist,
            doppler_signal,
        ).astype(np.float32)

        # Multiplicative Rayleigh speckle noise (tissue echoes)
        sigma_rayleigh = float(rng.uniform(0.05, 0.15))
        speckle = rng.rayleigh(sigma_rayleigh, size=(H, W)).astype(np.float32)
        # Speckle is higher in tissue (background) regions
        background_mask = (~inside).astype(np.float32)
        y_raw = doppler_signal + background_mask * speckle * 0.3 + speckle * 0.1

        # Add small additive Gaussian noise (thermal/electronic)
        noise_std = float(rng.uniform(0.01, 0.04))
        y_raw = y_raw + rng.standard_normal((H, W)).astype(np.float32) * noise_std

        # Normalize y to [0, 1]
        y_min = float(y_raw.min())
        y_max = float(y_raw.max())
        if y_max > y_min:
            y = ((y_raw - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "doppler_ultrasound",
                "prf_hz": prf_hz,
                "beam_angle_deg": beam_angle_deg,
                "vessel_diameter_mm": round(vessel_diameter_mm, 2),
            },
        })

    return samples


def generate_dot_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Diffuse Optical Tomography phantom with absorption coefficient inclusions.

    Creates a 64x64 float32 optical property map (absorption coefficient mu_a):
    heterogeneous tissue background (mu_a ~0.01-0.02 mm^-1) and one or two
    tumor inclusions (mu_a ~0.05-0.10 mm^-1).

    Applies diffuse optical forward model: Born approximation with 4
    source-detector pairs around the boundary, adds Gaussian measurement
    noise (~3% relative).

    x_true: 64x64 float32, normalized [0,1] — absorption coefficient map.
    y: 64x64 float32 — DOT reconstruction from boundary measurements.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, n_sources, n_detectors, wavelength_nm.

    Reference: Arridge, S.R., Inverse Problems 1999; Schweiger et al.,
    J. Biomed. Opt. 2005.
    """
    import numpy as np

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    n_sources = 4
    n_detectors = 4
    wavelength_nm = 785.0

    # Reduced scattering coefficient (background tissue, mm^-1)
    mu_s_prime = 1.0

    # Diffusion coefficient D = 1 / (3 * mu_s_prime)
    D = 1.0 / (3.0 * mu_s_prime)

    for i in range(n_samples):
        # --- Ground truth absorption map (mu_a in mm^-1) ---
        mu_a_bg = float(rng.uniform(0.01, 0.02))
        mu_a_map = np.full((H, W), mu_a_bg, dtype=np.float32)

        n_inclusions = rng.integers(1, 3)  # 1 or 2 tumor inclusions
        for _ in range(n_inclusions):
            # Tumor center (avoid edges)
            cx = float(rng.uniform(0.2 * W, 0.8 * W))
            cy = float(rng.uniform(0.2 * H, 0.8 * H))
            radius_px = float(rng.uniform(0.06 * min(H, W), 0.15 * min(H, W)))
            mu_a_tumor = float(rng.uniform(0.05, 0.10))

            Y_grid, X_grid = np.mgrid[:H, :W]
            dist = np.sqrt((X_grid - cx) ** 2 + (Y_grid - cy) ** 2).astype(np.float32)
            inclusion_mask = dist < radius_px
            mu_a_map[inclusion_mask] = mu_a_tumor

        # --- Born approximation forward model ---
        # Place n_sources equidistant source positions along left/top boundary
        # and n_detectors equidistant detector positions along right/bottom boundary.
        # Approximate Green's function: G(r, r_s) ~ exp(-mu_eff * |r - r_s|) / (4*pi*D*|r-r_s|)
        # Measurement: delta_phi = integral over volume of G(r_s, r) * delta_mu_a(r) * G(r, r_d) dV

        # Source positions: evenly spaced along left edge (x=0)
        src_positions = [(0.0, (j + 0.5) * H / n_sources) for j in range(n_sources)]
        # Detector positions: evenly spaced along right edge (x=W-1)
        det_positions = [(float(W - 1), (j + 0.5) * H / n_detectors) for j in range(n_detectors)]

        Y_grid, X_grid = np.mgrid[:H, :W]
        X_grid = X_grid.astype(np.float32)
        Y_grid = Y_grid.astype(np.float32)

        delta_mu_a = (mu_a_map - mu_a_bg).astype(np.float32)

        # Effective attenuation coefficient for background
        mu_eff_bg = float(np.sqrt(3.0 * mu_a_bg * mu_s_prime))

        # Accumulate Born-approximation boundary measurements into a sensitivity map
        sensitivity_map = np.zeros((H, W), dtype=np.float32)
        for sx, sy in src_positions:
            r_src = np.sqrt((X_grid - sx) ** 2 + (Y_grid - sy) ** 2) + 1e-6
            G_src = np.exp(-mu_eff_bg * r_src) / (4.0 * np.pi * D * r_src)
            for dx, dy in det_positions:
                r_det = np.sqrt((X_grid - dx) ** 2 + (Y_grid - dy) ** 2) + 1e-6
                G_det = np.exp(-mu_eff_bg * r_det) / (4.0 * np.pi * D * r_det)
                # Sensitivity (weight) for this source-detector pair at each voxel
                W_sd = G_src * G_det
                sensitivity_map += W_sd

        # Born approximation reconstruction: weight delta_mu_a by sensitivity
        recon = delta_mu_a * sensitivity_map

        # Add ~3% relative Gaussian noise
        noise_scale = 0.03 * (float(np.max(np.abs(recon))) + 1e-9)
        recon = recon + rng.standard_normal((H, W)).astype(np.float32) * noise_scale

        # Back-project to get DOT image estimate (add background component)
        y_raw = recon + mu_a_bg

        # Normalize x_true and y to [0, 1]
        mu_a_min = float(mu_a_map.min())
        mu_a_max = float(mu_a_map.max())
        if mu_a_max > mu_a_min:
            x_true = ((mu_a_map - mu_a_min) / (mu_a_max - mu_a_min)).astype(np.float32)
        else:
            x_true = np.zeros((H, W), dtype=np.float32)

        y_min = float(y_raw.min())
        y_max = float(y_raw.max())
        if y_max > y_min:
            y = ((y_raw - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "dot",
                "n_sources": n_sources,
                "n_detectors": n_detectors,
                "wavelength_nm": wavelength_nm,
            },
        })

    return samples


def generate_ebsd_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Electron Backscatter Diffraction (EBSD) phantom with polycrystalline microstructure.

    Creates a 64x64 float32 grain orientation map (Euler angle map, single channel)
    using Voronoi tessellation: ~10-20 grains each assigned a random crystal
    orientation in [0, 2*pi].

    Applies EBSD forward model: simulates Kikuchi pattern degradation by adding
    orientation-dependent Gaussian blur (sigma ~1-2 px) at grain boundaries and
    uniform Poisson-like shot noise (~5% relative).

    x_true: 64x64 float32, normalized [0,1] — grain orientation map.
    y: 64x64 float32 — noisy/blurred EBSD orientation map.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, n_grains, step_size_um, accelerating_voltage_kv.

    Reference: Krieger Lassen, N.C., J. Microsc. 1994; Chen et al.,
    Ultramicroscopy 2015.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    step_size_um = 0.5
    accelerating_voltage_kv = 20.0

    for i in range(n_samples):
        n_grains = int(rng.integers(10, 21))

        # --- Voronoi tessellation for grain structure ---
        # Place grain seeds randomly in the image domain
        seed_x = rng.uniform(0, W, size=n_grains).astype(np.float32)
        seed_y = rng.uniform(0, H, size=n_grains).astype(np.float32)

        # Assign random orientation (0 to 2*pi) to each grain
        orientations = rng.uniform(0.0, 2.0 * np.pi, size=n_grains).astype(np.float32)

        # Build grain label map via nearest-seed (Voronoi)
        Y_grid, X_grid = np.mgrid[:H, :W]
        X_grid = X_grid.astype(np.float32)
        Y_grid = Y_grid.astype(np.float32)

        # Compute distance from each pixel to each grain seed; assign to nearest
        grain_map = np.zeros((H, W), dtype=np.int32)
        min_dist = np.full((H, W), np.inf, dtype=np.float32)
        for g in range(n_grains):
            dist = np.sqrt((X_grid - seed_x[g]) ** 2 + (Y_grid - seed_y[g]) ** 2)
            closer = dist < min_dist
            grain_map[closer] = g
            min_dist[closer] = dist[closer]

        # Build orientation map: assign each pixel the orientation of its grain
        orientation_map = orientations[grain_map]  # shape (H, W), float32

        # --- Grain boundary detection (for orientation-dependent blur) ---
        # Detect boundary pixels: where any 4-connected neighbor has a different grain
        boundary_mask = np.zeros((H, W), dtype=np.float32)
        boundary_mask[:-1, :] += (grain_map[:-1, :] != grain_map[1:, :]).astype(np.float32)
        boundary_mask[1:, :] += (grain_map[:-1, :] != grain_map[1:, :]).astype(np.float32)
        boundary_mask[:, :-1] += (grain_map[:, :-1] != grain_map[:, 1:]).astype(np.float32)
        boundary_mask[:, 1:] += (grain_map[:, :-1] != grain_map[:, 1:]).astype(np.float32)
        boundary_mask = np.clip(boundary_mask, 0.0, 1.0)

        # --- EBSD forward model ---
        # 1. Orientation-dependent Gaussian blur (sigma ~1-2 px) stronger at boundaries
        sigma_interior = float(rng.uniform(0.5, 1.0))
        sigma_boundary = float(rng.uniform(1.0, 2.0))

        # Blend blurred versions: interior blur + extra boundary blur
        blurred_interior = gaussian_filter(orientation_map, sigma=sigma_interior)
        blurred_boundary = gaussian_filter(orientation_map, sigma=sigma_boundary)
        y_raw = (1.0 - boundary_mask) * blurred_interior + boundary_mask * blurred_boundary

        # 2. Poisson-like shot noise (~5% relative)
        noise_scale = 0.05 * float(np.max(np.abs(y_raw)) + 1e-9)
        y_raw = y_raw + rng.standard_normal((H, W)).astype(np.float32) * noise_scale

        # --- Normalize x_true and y to [0, 1] ---
        # x_true: normalize orientation map [0, 2*pi] -> [0, 1]
        x_true = (orientation_map / (2.0 * np.pi)).astype(np.float32)

        y_min = float(y_raw.min())
        y_max = float(y_raw.max())
        if y_max > y_min:
            y = ((y_raw - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "ebsd",
                "n_grains": n_grains,
                "step_size_um": step_size_um,
                "accelerating_voltage_kv": accelerating_voltage_kv,
            },
        })

    return samples


def generate_eddy_current_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Eddy Current Non-Destructive Testing (NDT) phantom with conductivity defects.

    Creates a 64x64 float32 conductivity map: conductive metal plate background
    (sigma ~10^6 S/m), surface defects/cracks (sigma ~0, voids), and corrosion
    regions (sigma reduced by 50-80%).

    Applies eddy current forward model: electromagnetic induction produces
    impedance change signals proportional to defect geometry; simulated as a
    blurred derivative of the conductivity map plus Gaussian noise.

    x_true: 64x64 float32, normalized [0,1] — defect map.
    y: 64x64 float32 — eddy current signal map.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, frequency_khz, lift_off_mm, material.

    Reference: Bowler, J. Appl. Phys. 1994; Sabbagh et al., IEEE Trans. Magn. 2010.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    frequency_khz = 100.0
    lift_off_mm = 1.0
    material = "aluminum"

    for i in range(n_samples):
        # --- Conductivity map (sigma normalized, 1.0 = bulk metal) ---
        conductivity = np.ones((H, W), dtype=np.float32)

        # Surface cracks / voids: sigma ~ 0
        n_cracks = int(rng.integers(2, 6))
        for _ in range(n_cracks):
            cx = int(rng.integers(5, W - 5))
            cy = int(rng.integers(5, H - 5))
            length = int(rng.integers(4, 12))
            width = int(rng.integers(1, 3))
            # Randomly oriented crack (horizontal or vertical)
            if rng.random() > 0.5:
                x0 = max(0, cx - length // 2)
                x1 = min(W, cx + length // 2)
                y0 = max(0, cy - width // 2)
                y1 = min(H, cy + width // 2)
            else:
                x0 = max(0, cx - width // 2)
                x1 = min(W, cx + width // 2)
                y0 = max(0, cy - length // 2)
                y1 = min(H, cy + length // 2)
            conductivity[y0:y1, x0:x1] = 0.0

        # Corrosion regions: sigma reduced by 50-80%
        n_corrosion = int(rng.integers(1, 4))
        for _ in range(n_corrosion):
            cx = int(rng.integers(8, W - 8))
            cy = int(rng.integers(8, H - 8))
            r = int(rng.integers(4, 10))
            reduction = float(rng.uniform(0.5, 0.8))
            yy, xx = np.ogrid[:H, :W]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
            conductivity[mask] = np.minimum(conductivity[mask], 1.0 - reduction)

        # --- Eddy current forward model ---
        # Impedance change signal ~ blurred spatial derivative of conductivity
        # (defect edges produce the largest eddy current perturbation)
        grad_y = np.gradient(conductivity, axis=0).astype(np.float32)
        grad_x = np.gradient(conductivity, axis=1).astype(np.float32)
        gradient_magnitude = np.sqrt(grad_y ** 2 + grad_x ** 2)

        # Blur to simulate sensor spatial resolution (lift-off effect)
        sigma_blur = float(rng.uniform(1.0, 2.5))
        y_raw = gaussian_filter(gradient_magnitude, sigma=sigma_blur)

        # Add Gaussian noise (~3% relative)
        noise_scale = 0.03 * float(np.max(np.abs(y_raw)) + 1e-9)
        y_raw = y_raw + rng.standard_normal((H, W)).astype(np.float32) * noise_scale

        # --- Normalize x_true (defect map): invert conductivity so defects = 1 ---
        x_true = (1.0 - conductivity).astype(np.float32)  # defects are high
        x_min = float(x_true.min())
        x_max = float(x_true.max())
        if x_max > x_min:
            x_true = ((x_true - x_min) / (x_max - x_min)).astype(np.float32)

        # Normalize y to [0, 1]
        y_min = float(y_raw.min())
        y_max = float(y_raw.max())
        if y_max > y_min:
            y = ((y_raw - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "eddy_current",
                "frequency_khz": frequency_khz,
                "lift_off_mm": lift_off_mm,
                "material": material,
            },
        })

    return samples


def generate_edx_mapping_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Energy-Dispersive X-ray (EDX/EDS) elemental mapping phantom.

    Creates a 64x64 float32 elemental distribution map simulating a
    multi-phase material with distinct compositional regions:
      - Fe-rich phase: occupies ~40% area, intensity 0.8-1.0
      - Si-rich inclusions: scattered circular regions, intensity 0.3-0.5
      - Al matrix background: intensity 0.1-0.2

    Applies EDX forward model:
      - Poisson counting statistics (low-count regime, ~100-500 counts/pixel)
      - X-ray background (Bremsstrahlung): smooth low-level ~10-30 counts
      - Peak overlap blurring: Gaussian blur simulating detector energy resolution

    x_true: 64x64 float32, normalized [0,1] — ground truth elemental map.
    y: 64x64 float32 — noisy EDX count map (normalized).
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, element, beam_energy_kv, acquisition_time_s,
              n_counts_per_pixel.

    References: Statham, J. Anal. At. Spectrom. 1995;
                Nicoletti et al., Nature 2013.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    beam_energy_kv = 15.0
    acquisition_time_s = 60.0
    element = "Fe"

    for i in range(n_samples):
        # --- Ground truth elemental distribution map ---
        # Al matrix background: 0.1-0.2
        al_level = float(rng.uniform(0.1, 0.2))
        x_true = np.full((H, W), al_level, dtype=np.float32)

        # Fe-rich phase: ~40% area as contiguous region(s)
        fe_intensity = float(rng.uniform(0.8, 1.0))
        # Use a threshold on smooth noise to create organic-looking Fe phase
        fe_noise = rng.standard_normal((H, W)).astype(np.float32)
        fe_smooth = gaussian_filter(fe_noise, sigma=float(rng.uniform(6.0, 10.0)))
        fe_threshold = float(np.percentile(fe_smooth, 60))  # top 40% area
        fe_mask = fe_smooth >= fe_threshold
        x_true[fe_mask] = fe_intensity

        # Si-rich inclusions: scattered circular regions, 0.3-0.5
        n_inclusions = int(rng.integers(3, 8))
        for _ in range(n_inclusions):
            cx = int(rng.integers(4, W - 4))
            cy = int(rng.integers(4, H - 4))
            r = int(rng.integers(2, 6))
            si_intensity = float(rng.uniform(0.3, 0.5))
            yy, xx = np.ogrid[:H, :W]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
            x_true[mask] = si_intensity

        # Clip to [0, 1]
        x_true = np.clip(x_true, 0.0, 1.0)

        # --- EDX forward model ---
        # Scale to count space: 100-500 counts/pixel for the maximum intensity
        peak_counts = float(rng.uniform(100.0, 500.0))
        count_map = x_true * peak_counts

        # X-ray background (Bremsstrahlung): smooth low-level 10-30 counts
        bkg_level = float(rng.uniform(10.0, 30.0))
        bkg_noise = rng.standard_normal((H, W)).astype(np.float32)
        bkg_smooth = gaussian_filter(bkg_noise, sigma=float(rng.uniform(8.0, 14.0)))
        # Normalize Bremsstrahlung background to [0.5, 1.5] * bkg_level
        bkg_min = float(bkg_smooth.min())
        bkg_max = float(bkg_smooth.max())
        if bkg_max > bkg_min:
            bkg_map = bkg_level * (0.5 + (bkg_smooth - bkg_min) / (bkg_max - bkg_min))
        else:
            bkg_map = np.full((H, W), bkg_level, dtype=np.float32)

        count_map_with_bkg = count_map + bkg_map.astype(np.float32)

        # Peak overlap blurring: Gaussian blur simulating detector energy resolution
        sigma_overlap = float(rng.uniform(0.5, 1.5))
        blurred = gaussian_filter(count_map_with_bkg, sigma=sigma_overlap)

        # Poisson counting statistics (low-count regime)
        lambda_map = np.maximum(blurred, 0.0)
        y_counts = rng.poisson(lambda_map).astype(np.float32)

        # Normalize y to [0, 1]
        y_min = float(y_counts.min())
        y_max = float(y_counts.max())
        if y_max > y_min:
            y = ((y_counts - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        n_counts_per_pixel = int(round(float(np.mean(y_counts))))

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "edx_mapping",
                "element": element,
                "beam_energy_kv": beam_energy_kv,
                "acquisition_time_s": acquisition_time_s,
                "n_counts_per_pixel": n_counts_per_pixel,
            },
        })

    return samples


def generate_eels_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Electron Energy Loss Spectroscopy (EELS) chemical phase map phantom.

    Creates a 64x64 float32 elemental/oxidation-state map simulating a TEM
    sample with distinct chemical phases:
      - MnO2 regions: characteristic EELS edge at ~640 eV, normalized intensity 0.9
      - MnO regions: intermediate oxidation state, intensity 0.6
      - Metallic Mn background: intensity 0.1-0.2

    Applies EELS forward model:
      - Poisson-dominated shot noise at low beam current (~200-500 counts/pixel)
      - Multiple scattering convolution: Gaussian blur sigma~0.5 px
      - Plural scattering background subtraction artifacts: smooth polynomial baseline

    x_true: 64x64 float32, normalized [0,1] — chemical phase map.
    y: 64x64 float32 — noisy EELS elemental map.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, energy_loss_ev, beam_current_pa,
              energy_resolution_ev.

    References: Egerton, EELS in the Electron Microscope, Springer 2011;
                Verbeeck & Van Aert, Ultramicroscopy 2004.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    energy_loss_ev = 640.0      # Mn L2,3 edge
    energy_resolution_ev = 0.8  # typical cold-FEG resolution

    for i in range(n_samples):
        # --- Ground truth chemical phase map ---
        # Metallic Mn background: 0.1-0.2
        mn_bg_level = float(rng.uniform(0.1, 0.2))
        x_true = np.full((H, W), mn_bg_level, dtype=np.float32)

        # MnO2 regions (~640 eV edge): organic-shaped regions, intensity 0.9
        mno2_noise = rng.standard_normal((H, W)).astype(np.float32)
        mno2_smooth = gaussian_filter(mno2_noise, sigma=float(rng.uniform(5.0, 9.0)))
        mno2_threshold = float(np.percentile(mno2_smooth, 75))  # top 25% area
        mno2_mask = mno2_smooth >= mno2_threshold
        x_true[mno2_mask] = 0.9

        # MnO regions (intermediate oxidation): patchy regions, intensity 0.6
        mno_noise = rng.standard_normal((H, W)).astype(np.float32)
        mno_smooth = gaussian_filter(mno_noise, sigma=float(rng.uniform(4.0, 7.0)))
        mno_threshold = float(np.percentile(mno_smooth, 65))  # top 35% area
        mno_mask = (mno_smooth >= mno_threshold) & (~mno2_mask)
        x_true[mno_mask] = 0.6

        # Clip to [0, 1]
        x_true = np.clip(x_true, 0.0, 1.0)

        # --- EELS forward model ---
        # Scale to count space: 200-500 counts/pixel for max intensity
        beam_current_pa = float(rng.uniform(200.0, 500.0))
        count_map = x_true * beam_current_pa

        # Plural scattering background: smooth polynomial baseline (artifact)
        yy, xx = np.mgrid[0:H, 0:W]
        # Normalize coordinates to [0, 1]
        yn = yy / max(H - 1, 1)
        xn = xx / max(W - 1, 1)
        # Low-order polynomial baseline (simulates plural scattering background)
        a0 = float(rng.uniform(5.0, 15.0))
        a1 = float(rng.uniform(-5.0, 5.0))
        a2 = float(rng.uniform(-5.0, 5.0))
        baseline = (a0 + a1 * xn + a2 * yn).astype(np.float32)
        count_map_with_bkg = count_map + baseline

        # Multiple scattering convolution: Gaussian blur sigma~0.5 px
        sigma_ms = float(rng.uniform(0.4, 0.7))
        blurred = gaussian_filter(count_map_with_bkg, sigma=sigma_ms)

        # Poisson-dominated shot noise (low beam current regime)
        lambda_map = np.maximum(blurred, 0.0)
        y_counts = rng.poisson(lambda_map).astype(np.float32)

        # Normalize y to [0, 1]
        y_min = float(y_counts.min())
        y_max = float(y_counts.max())
        if y_max > y_min:
            y = ((y_counts - y_min) / (y_max - y_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "eels",
                "energy_loss_ev": energy_loss_ev,
                "beam_current_pa": beam_current_pa,
                "energy_resolution_ev": energy_resolution_ev,
            },
        })

    return samples


def generate_eht_imaging_phantom(
    n_samples: int = 3,
    seed: int = 42,
    shape: tuple = (64, 64),
    target_shape=None,
) -> list[dict]:
    """
    Event Horizon Telescope (EHT) / VLBI black hole imaging phantom.

    Creates a 64x64 float32 radio brightness map simulating accretion disk
    emission around a black hole:
      - Bright ring structure at radius ~15-25 pixels from center
      - Dark photon ring depression in the center (Gaussian shadow)
      - Bright hot spot on one side (Doppler boosting from relativistic motion)

    Applies EHT/VLBI forward model:
      - Sparse Fourier sampling with ~10 baseline pairs (u-v plane mask
        covering ~20% of spatial frequencies)
      - Thermal noise: complex Gaussian on visibilities
      - Dirty image reconstruction via back-projection (inverse FFT of
        sparsely sampled visibilities)

    x_true: 64x64 float32, normalized [0,1] — ground truth brightness.
    y: 64x64 float32 — dirty image / back-projected reconstruction.
    H_ideal: np.eye(64, dtype=np.float32).
    metadata: dict with modality, n_baselines, uv_coverage_fraction,
              wavelength_mm, target.

    References: Hogbom, A&AS 1974; Event Horizon Telescope Collaboration,
                ApJL 2019; Chael et al., ApJ 2018.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if target_shape is not None:
        H = target_shape[0]
        W = target_shape[1] if len(target_shape) > 1 else H
    else:
        H, W = shape

    rng = np.random.default_rng(seed)
    samples = []

    wavelength_mm = 1.3       # EHT observing wavelength (230 GHz)
    n_baselines = 10          # ~10 baseline pairs for EHT array
    uv_coverage_fraction = 0.20  # sparse u-v coverage (~20%)

    for i in range(n_samples):
        # --- Ground truth brightness distribution ---
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0

        # Distance from center
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

        # Accretion disk ring: Gaussian annulus centered at radius ~20 px
        ring_radius = float(rng.uniform(15.0, 25.0))
        ring_width = float(rng.uniform(3.0, 6.0))
        ring = np.exp(-0.5 * ((r - ring_radius) / ring_width) ** 2).astype(np.float32)

        # Central shadow (photon ring depression): Gaussian hole at center
        shadow_sigma = float(rng.uniform(5.0, 9.0))
        shadow = np.exp(-0.5 * (r / shadow_sigma) ** 2).astype(np.float32)
        # Suppress ring inside shadow radius
        x_true = ring * (1.0 - 0.85 * shadow)

        # Doppler-boosted hot spot: bright blob on one side of the ring
        hotspot_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        hotspot_r = ring_radius
        hotspot_cx = cx + hotspot_r * np.cos(hotspot_angle)
        hotspot_cy = cy + hotspot_r * np.sin(hotspot_angle)
        hotspot_sigma = float(rng.uniform(2.0, 5.0))
        hotspot_dist = np.sqrt((yy - hotspot_cy) ** 2 + (xx - hotspot_cx) ** 2)
        hotspot = float(rng.uniform(0.4, 0.8)) * np.exp(
            -0.5 * (hotspot_dist / hotspot_sigma) ** 2
        ).astype(np.float32)
        x_true = x_true + hotspot

        # Normalize to [0, 1]
        x_max = float(x_true.max())
        if x_max > 0:
            x_true = (x_true / x_max).astype(np.float32)
        else:
            x_true = x_true.astype(np.float32)

        # --- EHT/VLBI forward model ---
        # 1. Compute full 2D FFT of the brightness distribution
        vis_full = np.fft.fft2(x_true)

        # 2. Build sparse u-v mask: ~20% coverage via n_baselines random baselines
        uv_mask = np.zeros((H, W), dtype=bool)
        for _ in range(n_baselines):
            # Random baseline orientation and length (in Fourier pixels)
            angle = float(rng.uniform(0, np.pi))
            max_len = min(H, W) // 2
            length = int(rng.integers(3, max_len))
            # Rasterize baseline as a line through Fourier space (and conjugate)
            for t in range(-length, length + 1):
                u = int(round(cx + t * np.cos(angle))) % W
                v = int(round(cy + t * np.sin(angle))) % H
                uv_mask[v, u] = True
                # Hermitian conjugate
                uv_mask[(H - v) % H, (W - u) % W] = True

        # 3. Apply mask + thermal noise on visibilities
        thermal_noise_std = float(rng.uniform(0.02, 0.08))
        noise_real = rng.standard_normal((H, W)).astype(np.float32) * thermal_noise_std
        noise_imag = rng.standard_normal((H, W)).astype(np.float32) * thermal_noise_std
        noise_complex = noise_real + 1j * noise_imag

        vis_sparse = np.zeros((H, W), dtype=np.complex64)
        vis_sparse[uv_mask] = (vis_full[uv_mask] + noise_complex[uv_mask]).astype(np.complex64)

        # 4. Dirty image: back-projection via inverse FFT of sparse visibilities
        dirty = np.real(np.fft.ifft2(vis_sparse)).astype(np.float32)

        # 5. Normalize dirty image to [0, 1]
        d_min = float(dirty.min())
        d_max = float(dirty.max())
        if d_max > d_min:
            y = ((dirty - d_min) / (d_max - d_min)).astype(np.float32)
        else:
            y = np.zeros((H, W), dtype=np.float32)

        actual_coverage = float(uv_mask.sum()) / float(H * W)
        H_ideal = np.eye(64, dtype=np.float32)

        samples.append({
            "x_true": x_true,
            "y": y,
            "H_ideal": H_ideal,
            "metadata": {
                "modality": "eht_imaging",
                "n_baselines": n_baselines,
                "uv_coverage_fraction": round(actual_coverage, 3),
                "wavelength_mm": wavelength_mm,
                "target": "M87*" if i % 2 == 0 else "SgrA*",
            },
        })

    return samples


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
        "generate_sam_phantom": lambda: generate_sam_phantom(target_shape=target_shape),
        "generate_thermography_phantom": lambda: generate_thermography_phantom(target_shape=target_shape),
        "generate_ao_wavefront": lambda: generate_ao_wavefront(target_shape=target_shape),
        "generate_afm_surface": lambda: generate_afm_surface(target_shape=target_shape),
        "generate_angiography_vessel_phantom": lambda: generate_angiography_vessel_phantom(target_shape=target_shape),
        "generate_asl_perfusion_phantom": lambda: generate_asl_perfusion_phantom(target_shape=target_shape),
        "generate_apt_composition_map": lambda: generate_apt_composition_map(target_shape=target_shape),
        "generate_blt_source_phantom": lambda: generate_blt_source_phantom(target_shape=target_shape),
        "generate_brachytherapy_seed_phantom": lambda: generate_brachytherapy_seed_phantom(target_shape=target_shape),
        "generate_brillouin_vipa_phantom": lambda: generate_brillouin_vipa_phantom(target_shape=target_shape),
        "generate_cars_raman_phantom": lambda: generate_cars_raman_phantom(target_shape=target_shape),
        "generate_cacti_video_phantom": lambda: generate_cacti_video_phantom(target_shape=target_shape),
        "generate_cathodoluminescence_phantom": lambda: generate_cathodoluminescence_phantom(target_shape=target_shape),
        "generate_cbct_head_phantom": lambda: generate_cbct_head_phantom(target_shape=target_shape),
        "generate_cest_mri_phantom": lambda: generate_cest_mri_phantom(target_shape=target_shape),
        "generate_ceus_phantom": lambda: generate_ceus_phantom(target_shape=target_shape),
        "generate_clem_phantom": lambda: generate_clem_phantom(target_shape=target_shape),
        "generate_coded_exposure_phantom": lambda: generate_coded_exposure_phantom(target_shape=target_shape),
        "generate_confocal_3d_phantom": lambda: generate_confocal_3d_phantom(target_shape=target_shape),
        "generate_confocal_endomicroscopy_phantom": lambda: generate_confocal_endomicroscopy_phantom(target_shape=target_shape),
        "generate_confocal_livecell_phantom": lambda: generate_confocal_livecell_phantom(target_shape=target_shape),
        "generate_coronagraphy_phantom": lambda: generate_coronagraphy_phantom(target_shape=target_shape),
        "generate_cryo_em_phantom": generate_cryo_em_phantom,
        "generate_cryo_et_phantom": generate_cryo_et_phantom,
        "generate_ct_phantom": generate_ct_phantom,
        "generate_ct_fluorescence_phantom": generate_ct_fluorescence_phantom,
        "generate_cup_phantom": generate_cup_phantom,
        "generate_dark_field_phantom": lambda: generate_dark_field_phantom(target_shape=target_shape),
        "generate_dexa_phantom": lambda: generate_dexa_phantom(target_shape=target_shape),
        "generate_desi_phantom": lambda: generate_desi_phantom(target_shape=target_shape),
        "generate_dic_phantom": lambda: generate_dic_phantom(target_shape=target_shape),
        "generate_diffusion_mri_phantom": generate_diffusion_mri_phantom,
        "generate_digital_breast_tomo_phantom": generate_digital_breast_tomo_phantom,
        "generate_dna_paint_phantom": generate_dna_paint_phantom,
        "generate_doppler_ultrasound_phantom": generate_doppler_ultrasound_phantom,
        "generate_dot_phantom": generate_dot_phantom,
        "generate_ebsd_phantom": generate_ebsd_phantom,
        "generate_eddy_current_phantom": generate_eddy_current_phantom,
        "generate_edx_mapping_phantom": generate_edx_mapping_phantom,
        "generate_eels_phantom": generate_eels_phantom,
        "generate_eht_imaging_phantom": generate_eht_imaging_phantom,
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
        "generate_angiography_vessel_phantom": lambda: generate_angiography_vessel_phantom(target_shape=target_shape),
        "generate_asl_perfusion_phantom": lambda: generate_asl_perfusion_phantom(target_shape=target_shape),
        "generate_apt_composition_map": lambda: generate_apt_composition_map(target_shape=target_shape),
        "generate_blt_source_phantom": lambda: generate_blt_source_phantom(target_shape=target_shape),
        "generate_brachytherapy_seed_phantom": lambda: generate_brachytherapy_seed_phantom(target_shape=target_shape),
        "generate_brillouin_vipa_phantom": lambda: generate_brillouin_vipa_phantom(target_shape=target_shape),
        "generate_cars_raman_phantom": lambda: generate_cars_raman_phantom(target_shape=target_shape),
        "generate_cacti_video_phantom": lambda: generate_cacti_video_phantom(target_shape=target_shape),
        "generate_cathodoluminescence_phantom": lambda: generate_cathodoluminescence_phantom(target_shape=target_shape),
        "generate_cbct_head_phantom": lambda: generate_cbct_head_phantom(target_shape=target_shape),
        "generate_cest_mri_phantom": lambda: generate_cest_mri_phantom(target_shape=target_shape),
        "generate_ceus_phantom": lambda: generate_ceus_phantom(target_shape=target_shape),
        "generate_clem_phantom": lambda: generate_clem_phantom(target_shape=target_shape),
        "generate_coded_exposure_phantom": lambda: generate_coded_exposure_phantom(target_shape=target_shape),
        "generate_confocal_3d_phantom": lambda: generate_confocal_3d_phantom(target_shape=target_shape),
        "generate_confocal_endomicroscopy_phantom": lambda: generate_confocal_endomicroscopy_phantom(target_shape=target_shape),
        "generate_confocal_livecell_phantom": lambda: generate_confocal_livecell_phantom(target_shape=target_shape),
        "generate_coronagraphy_phantom": lambda: generate_coronagraphy_phantom(target_shape=target_shape),
        "generate_cryo_em_phantom": lambda: generate_cryo_em_phantom(),
        "generate_cryo_et_phantom": lambda: generate_cryo_et_phantom(),
        "generate_ct_phantom": lambda: generate_ct_phantom(),
        "generate_ct_fluorescence_phantom": lambda: generate_ct_fluorescence_phantom(),
        "generate_cup_phantom": lambda: generate_cup_phantom(),
        "generate_dark_field_phantom": lambda: generate_dark_field_phantom(target_shape=target_shape),
        "generate_dexa_phantom": lambda: generate_dexa_phantom(target_shape=target_shape),
        "generate_desi_phantom": lambda: generate_desi_phantom(target_shape=target_shape),
        "generate_dic_phantom": lambda: generate_dic_phantom(target_shape=target_shape),
        "generate_diffusion_mri_phantom": lambda: generate_diffusion_mri_phantom(),
        "generate_digital_breast_tomo_phantom": lambda: generate_digital_breast_tomo_phantom(),
        "generate_dna_paint_phantom": lambda: generate_dna_paint_phantom(),
        "generate_doppler_ultrasound_phantom": lambda: generate_doppler_ultrasound_phantom(),
        "generate_dot_phantom": lambda: generate_dot_phantom(),
        "generate_ebsd_phantom": lambda: generate_ebsd_phantom(),
        "generate_eddy_current_phantom": lambda: generate_eddy_current_phantom(),
        "generate_edx_mapping_phantom": lambda: generate_edx_mapping_phantom(),
        "generate_eels_phantom": lambda: generate_eels_phantom(),
        "generate_eht_imaging_phantom": lambda: generate_eht_imaging_phantom(),
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
