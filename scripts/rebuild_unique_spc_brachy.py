"""Rebuild SPC and brachytherapy_img with unique phantoms."""
import numpy as np, h5py, os, json
from pathlib import Path

BASE = Path("D:/onedrive/startup/program/physics_world_model/PWM5/Physics_World_Model/datasets/benchmark")
N_SAMPLES = 10


def to_uint8(x):
    x = np.nan_to_num(x, 0)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def save_img(arr, path):
    """Save 2D array as PGM (no PIL needed)."""
    u = to_uint8(arr)
    if u.ndim == 2:
        h, w = u.shape
        with open(path.replace('.png', '.pgm'), 'wb') as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(u.tobytes())


# ─── SPC: USAF-1951 resolution target ───────────────────────────────
# Single-pixel cameras are tested with resolution targets (bar patterns)
# Reference: Duarte et al., "Single-Pixel Imaging via Compressive Sampling",
# IEEE Signal Processing Magazine, 2008. DOI:10.1109/MSP.2007.914730
def phantom_spc_resolution_target(seed):
    """USAF-1951 style resolution target — standard SPC test object."""
    rng = np.random.RandomState(seed)
    n = 64  # SPC x_shape is 64x64
    img = np.ones((n, n), dtype=np.float32) * 0.05  # dark background

    # Horizontal bar groups at different scales
    for group_idx in range(4):
        scale = 2 ** (group_idx + 1)  # 2, 4, 8, 16 pixels
        y_start = 4 + group_idx * 14
        for bar in range(3):
            y = y_start + bar * (scale // 2 + 1)
            x_start = 4 + rng.randint(0, 4)
            if y + scale // 2 <= n:
                img[y:y + max(1, scale // 4), x_start:x_start + scale] = 0.9

    # Vertical bar groups
    for group_idx in range(4):
        scale = 2 ** (group_idx + 1)
        x_start = 34 + group_idx * 7
        for bar in range(3):
            x = x_start + bar * max(1, scale // 4 + 1)
            y_start = 4 + rng.randint(0, 4)
            if x + max(1, scale // 4) <= n:
                img[y_start:y_start + scale, x:x + max(1, scale // 4)] = 0.9

    # Add a few point targets (stars)
    for _ in range(3 + seed % 3):
        cy, cx = rng.randint(4, n - 4, size=2)
        img[cy, cx] = 1.0
        if cy + 1 < n: img[cy + 1, cx] = 0.7
        if cy - 1 >= 0: img[cy - 1, cx] = 0.7
        if cx + 1 < n: img[cy, cx + 1] = 0.7
        if cx - 1 >= 0: img[cy, cx - 1] = 0.7

    # Random slight rotation/shift for variety
    shift = rng.randint(-2, 3, size=2)
    img = np.roll(img, shift[0], axis=0)
    img = np.roll(img, shift[1], axis=1)

    return img.astype(np.float32)


def fwd_spc(x, seed):
    """SPC forward model: random binary measurement matrix."""
    rng = np.random.RandomState(seed + 9999)
    n = x.size
    m = 614  # y_shape from YAML
    # Random Bernoulli sensing matrix (standard for SPC)
    Phi = (rng.rand(m, n) > 0.5).astype(np.float32)
    y = Phi @ x.ravel()
    return y


# ─── Brachytherapy: seed/catheter dose distribution ──────────────────
# Brachytherapy imaging reconstructs dose distributions from seed positions
# Reference: Rivard et al., "Update of AAPM TG-43 Report",
# Medical Physics, 2004. DOI:10.1118/1.1646040
def phantom_brachytherapy_seeds(seed):
    """Brachytherapy seed arrangement — dose rate distribution phantom."""
    rng = np.random.RandomState(seed)
    n = 64  # x_shape is 64x64
    dose = np.zeros((n, n), dtype=np.float32)

    # Anatomical background: prostate-like elliptical organ
    cy, cx = n // 2 + rng.randint(-3, 4), n // 2 + rng.randint(-2, 3)
    ry, rx = 18 + rng.randint(-3, 4), 22 + rng.randint(-3, 4)
    yy, xx = np.ogrid[:n, :n]
    organ_mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    dose[organ_mask] = 0.05  # low background tissue

    # Place 8-15 brachytherapy seeds (Ir-192 or I-125 like)
    n_seeds = 8 + rng.randint(0, 8)
    for _ in range(n_seeds):
        sy = cy + rng.randint(-ry + 3, ry - 2)
        sx = cx + rng.randint(-rx + 3, rx - 2)
        # TG-43 dose falloff: ~1/r^2 with anisotropy
        for dy in range(-15, 16):
            for dx in range(-15, 16):
                py, px = sy + dy, sx + dx
                if 0 <= py < n and 0 <= px < n:
                    r2 = dy * dy + dx * dx + 0.5
                    # Anisotropy: slightly elongated along seed axis
                    angle_factor = 1.0 + 0.3 * abs(dy) / (abs(dx) + 1)
                    dose[py, px] += (1.0 / (r2 * angle_factor)) * (1 + 0.1 * rng.rand())

    # Urethra avoidance zone (dose dip)
    uy, ux = cy, cx + rng.randint(-2, 3)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            py, px = uy + dy, ux + dx
            if 0 <= py < n and 0 <= px < n:
                dose[py, px] *= 0.3

    # Normalize
    dose = dose / (dose.max() + 1e-12)
    return dose.astype(np.float32)


def fwd_brachytherapy(x, seed):
    """Brachytherapy forward: gamma camera projection with attenuation."""
    rng = np.random.RandomState(seed + 7777)
    n = x.shape[0]
    # Simple attenuation + scatter
    mu = 0.02 + 0.01 * rng.rand()  # attenuation coefficient
    y = np.zeros_like(x)
    for i in range(n):
        atten = np.exp(-mu * np.arange(n))
        y[i, :] = x[i, :] * atten
    # Add scatter kernel
    from scipy.ndimage import gaussian_filter
    y = gaussian_filter(y, sigma=1.5 + 0.5 * rng.rand())
    return y.astype(np.float32)


def save_modality(mod, phantom_fn, fwd_fn, x_shape, y_shape_hint=None):
    out_dir = BASE / mod / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    for i in range(N_SAMPLES):
        x = phantom_fn(42 + i)
        y = fwd_fn(x, 42 + i)
        h5_path = out_dir / f"standard_{mod}_{i:02d}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            f.attrs["modality"] = mod
            f.attrs["sample_index"] = i
            f.attrs["phantom_seed"] = 42 + i

        # Save visualization
        save_img(x, str(img_dir / f"x_true_{i:02d}.png"))
        if y.ndim == 2:
            save_img(y, str(img_dir / f"y_ideal_{i:02d}.png"))

    # Metadata
    meta = {
        "modality": mod,
        "n_samples": N_SAMPLES,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "phantom": phantom_fn.__doc__.strip() if phantom_fn.__doc__ else "",
        "format": "per-sample HDF5, gzip compressed"
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {mod}: {N_SAMPLES} samples, x={list(x.shape)}, y={list(y.shape)}")


if __name__ == "__main__":
    print("Rebuilding SPC and brachytherapy_img with unique phantoms...")
    save_modality("spc", phantom_spc_resolution_target, fwd_spc, (64, 64))
    save_modality("brachytherapy_img", phantom_brachytherapy_seeds, fwd_brachytherapy, (64, 64))
    print("Done!")
