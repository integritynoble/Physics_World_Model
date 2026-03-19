"""
build_standard_panorama.py
Build standard panorama / equirectangular imaging dataset for PWM benchmark.

x_true : (256,512,3) float32 [0,1] — equirectangular RGB panorama
y_ideal : (128,128,3) float32 [0,1] — perspective crop at azimuth=0, elevation=0

Source / citation:
  SUN360, Xiao et al. CVPR 2012, doi:10.1109/CVPR.2012.6247971
  Since SUN360 requires licence, synthetic panoramic textures are used.
"""

import sys
import json
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "datasets" / "benchmark" / "panorama" / "standard"
MODALITY = "panorama"
N_SAMPLES = 10
PANO_H = 256
PANO_W = 512
CROP_H = 128
CROP_W = 128


# ---------------------------------------------------------------------------
# Synthetic equirectangular panorama generator
# ---------------------------------------------------------------------------

def _make_panorama(rng):
    """Return (PANO_H, PANO_W, 3) float32 equirectangular image in [0,1].

    UV sphere texture built from geometric patterns on longitude/latitude grid.
    """
    # Longitude: [0, 2pi], Latitude: [-pi/2, pi/2]
    lat = np.linspace(-np.pi / 2, np.pi / 2, PANO_H)
    lon = np.linspace(-np.pi, np.pi, PANO_W)
    LON, LAT = np.meshgrid(lon, lat)

    # Base colour gradient from sky (top) to ground (bottom)
    sky_col = rng.uniform(0.4, 1.0, 3).astype(np.float32)
    ground_col = rng.uniform(0.0, 0.5, 3).astype(np.float32)
    t = np.clip((LAT + np.pi / 2) / np.pi, 0, 1)[:, :, None].astype(np.float32)
    pano = (1 - t) * sky_col + t * ground_col   # (H, W, 3)

    # Add structured geometric bands (simulating horizon, horizon stripe, clouds)
    n_bands = rng.integers(2, 6)
    for _ in range(n_bands):
        lat_centre = rng.uniform(-0.8, 0.8)
        lat_width = rng.uniform(0.05, 0.25)
        band_col = rng.uniform(0.0, 1.0, 3).astype(np.float32)
        weight = np.exp(-0.5 * ((LAT - lat_centre) / lat_width) ** 2)[:, :, None]
        pano = pano * (1 - weight) + band_col * weight

    # Add vertical stripes (pillars, trees, buildings)
    n_stripes = rng.integers(3, 12)
    for _ in range(n_stripes):
        lon_centre = rng.uniform(-np.pi, np.pi)
        lon_width = rng.uniform(0.05, 0.3)
        stripe_col = rng.uniform(0.0, 1.0, 3).astype(np.float32)
        weight_h = np.exp(-0.5 * ((LON - lon_centre) / lon_width) ** 2)
        # Only in lower half (ground / structure region)
        height_mask = np.clip((LAT + np.pi / 2) / (np.pi / 2), 0, 1)
        weight = (weight_h * height_mask)[:, :, None]
        pano = pano * (1 - weight * 0.6) + stripe_col * weight * 0.6

    # Add some circular sun/moon disc
    if rng.random() > 0.3:
        sun_lat = rng.uniform(0.1, 0.8)
        sun_lon = rng.uniform(-2.5, 2.5)
        sun_r = rng.uniform(0.05, 0.15)
        sun_col = rng.uniform(0.7, 1.0, 3).astype(np.float32)
        dist2 = (LAT - sun_lat) ** 2 + (LON - sun_lon) ** 2
        mask = (dist2 < sun_r ** 2).astype(np.float32)[:, :, None]
        pano = pano * (1 - mask) + sun_col * mask

    pano = np.clip(pano, 0, 1).astype(np.float32)
    return pano


# ---------------------------------------------------------------------------
# Perspective crop from equirectangular
# ---------------------------------------------------------------------------

def _equirect_to_perspective(pano, fov_deg=90.0, azimuth_deg=0.0, elevation_deg=0.0,
                              out_h=128, out_w=128):
    """Extract a perspective crop from an equirectangular panorama.

    Uses central-projection sampling on the sphere surface.
    Returns (out_h, out_w, 3) float32.
    """
    H, W, C = pano.shape
    fov = np.deg2rad(fov_deg)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    # Output pixel grid in normalised device coords [-1,1]
    xs = np.linspace(-1, 1, out_w)
    ys = np.linspace(-1, 1, out_h)
    XS, YS = np.meshgrid(xs, ys)

    # Focal length for given FOV
    f = 1.0 / np.tan(fov / 2)

    # 3-D ray directions in camera frame
    Xc = XS
    Yc = YS
    Zc = np.full_like(XS, f)
    # Normalise
    norm = np.sqrt(Xc ** 2 + Yc ** 2 + Zc ** 2)
    Xc /= norm; Yc /= norm; Zc /= norm

    # Rotate by azimuth (yaw around Y) and elevation (pitch around X)
    # Rotation matrix: Ry(az) * Rx(-el)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float64)
    R = Ry @ Rx

    dirs = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=0)  # (3, N)
    dirs_world = R @ dirs   # (3, N)

    Xw, Yw, Zw = dirs_world[0], dirs_world[1], dirs_world[2]

    # Convert world direction to equirectangular coordinates
    lat = np.arcsin(np.clip(Yw, -1, 1))               # [-pi/2, pi/2]
    lon = np.arctan2(Xw, Zw)                           # [-pi, pi]

    # Map to pixel indices
    u = (lon / (2 * np.pi) + 0.5) * W   # [0, W]
    v = (0.5 - lat / np.pi) * H          # [0, H]

    u = np.clip(u, 0, W - 1).reshape(out_h, out_w)
    v = np.clip(v, 0, H - 1).reshape(out_h, out_w)

    # Bilinear sample
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    wu = (u - u0).astype(np.float32)[:, :, None]
    wv = (v - v0).astype(np.float32)[:, :, None]

    crop = ((1 - wv) * (1 - wu) * pano[v0, u0, :]
            + (1 - wv) * wu * pano[v0, u1, :]
            + wv * (1 - wu) * pano[v1, u0, :]
            + wv * wu * pano[v1, u1, :])

    return crop.astype(np.float32)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[panorama] Output directory: {OUT_DIR}")

    metadata_list = []

    for idx in range(N_SAMPLES):
        rng = np.random.default_rng(seed=idx + 42)
        sample_id = f"sample_{idx:04d}"
        h5_path = OUT_DIR / f"{sample_id}.h5"

        print(f"  [{idx+1}/{N_SAMPLES}] Building {sample_id} ...")

        x_true = _make_panorama(rng)            # (256,512,3)
        y_ideal = _equirect_to_perspective(     # (128,128,3)
            x_true,
            fov_deg=90.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
            out_h=CROP_H,
            out_w=CROP_W,
        )

        assert x_true.shape == (PANO_H, PANO_W, 3)
        assert y_ideal.shape == (CROP_H, CROP_W, 3)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_true", data=x_true, compression="gzip")
            hf.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = hf.create_group("H_params")
            hp.attrs["forward_model"] = "Perspective projection from equirectangular (lat/lon crop)"
            hp.attrs["fov_deg"] = 90.0
            hp.attrs["azimuth_deg"] = 0.0
            hp.attrs["elevation_deg"] = 0.0
            hp.attrs["crop_height"] = CROP_H
            hp.attrs["crop_width"] = CROP_W
            hp.attrs["pano_height"] = PANO_H
            hp.attrs["pano_width"] = PANO_W

            meta = hf.create_group("metadata")
            meta.attrs["modality"] = MODALITY
            meta.attrs["sample_id"] = sample_id
            meta.attrs["x_shape"] = str(list(x_true.shape))
            meta.attrs["y_shape"] = str(list(y_ideal.shape))
            meta.attrs["seed"] = idx + 42
            meta.attrs["created"] = datetime.utcnow().isoformat()
            meta.attrs["source"] = "Synthetic equirectangular panorama (UV sphere texture)"
            meta.attrs["reference"] = "Xiao et al. CVPR 2012 doi:10.1109/CVPR.2012.6247971"

        metadata_list.append({
            "sample_id": sample_id,
            "h5_file": f"{sample_id}.h5",
            "x_shape": list(x_true.shape),
            "y_shape": list(y_ideal.shape),
            "seed": idx + 42,
        })

    meta_doc = {
        "modality": MODALITY,
        "n_samples": N_SAMPLES,
        "split": "standard",
        "created": datetime.utcnow().isoformat(),
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
        "source": "Synthetic panorama: UV sphere texture maps from geometric patterns",
        "reference": "Xiao et al. CVPR 2012, doi:10.1109/CVPR.2012.6247971",
        "samples": metadata_list,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta_doc, f, indent=2)

    spec_doc = {
        "modality": MODALITY,
        "x_description": "Equirectangular RGB panorama, shape (256,512,3), float32 [0,1]",
        "y_description": "Perspective crop at azimuth=0 elevation=0, shape (128,128,3), float32 [0,1]",
        "forward_model": "Perspective projection: lat/lon bilinear sampling at FOV=90 deg",
        "fov_deg": 90.0,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "x_shape": [PANO_H, PANO_W, 3],
        "y_shape": [CROP_H, CROP_W, 3],
        "dtype": "float32",
        "normalisation": "[0,1] RGB for both x_true and y_ideal",
        "reconstruction_algorithms": [
            "equirectangular inpainting", "360-degree completion CNN",
            "OmniDreamer", "PanoSwin"
        ],
        "reference": "Xiao et al. CVPR 2012, doi:10.1109/CVPR.2012.6247971",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{MODALITY}/standard/",
    }
    with open(OUT_DIR / "spec.json", "w") as f:
        json.dump(spec_doc, f, indent=2)

    print(f"[panorama] Done. {N_SAMPLES} samples written to {OUT_DIR}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
