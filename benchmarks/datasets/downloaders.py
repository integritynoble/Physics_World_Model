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
        "generate_sam_phantom": lambda: generate_sam_phantom(target_shape=target_shape),
        "generate_thermography_phantom": lambda: generate_thermography_phantom(target_shape=target_shape),
        "generate_ao_wavefront": lambda: generate_ao_wavefront(target_shape=target_shape),
        "generate_afm_surface": lambda: generate_afm_surface(target_shape=target_shape),
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
