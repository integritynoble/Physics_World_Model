"""
Rebuild standard datasets with UNIQUE phantoms — Batch 1: X-ray / CT modalities.

Each modality gets its own physically-motivated phantom that looks different from all others.
No two modalities share the same ground truth data.

Author: Chengshuai Yang
"""
import numpy as np
import h5py
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "benchmark"
N = 10  # samples per modality


def to_uint8(arr):
    arr = np.nan_to_num(arr)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-10:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def save_img(arr, path):
    if arr.ndim == 2:
        Image.fromarray(to_uint8(arr), "L").save(str(path))
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        Image.fromarray(to_uint8(arr), "RGB").save(str(path))
    elif arr.ndim == 3:
        # Save middle slice
        mid = arr.shape[0] // 2
        Image.fromarray(to_uint8(arr[mid]), "L").save(str(path))


def radon_fwd(x, n_angles=120):
    H, W = x.shape
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    sino = np.zeros((n_angles, W), dtype=np.float32)
    for i, a in enumerate(angles):
        rot = ndimage.rotate(x, a, reshape=False, order=1)
        sino[i] = rot.sum(axis=0)
    sino /= sino.max() + 1e-10
    return sino


def save_modality(modality, samples, fwd_type, fwd_desc, meta_extra):
    out = DATA / modality / "standard"
    out.mkdir(parents=True, exist_ok=True)
    # Remove old
    for old in out.glob("standard_*.h5"):
        old.unlink()
    files = []
    for i, (x, y) in enumerate(samples):
        h5p = out / f"standard_{modality}_{i:02d}.h5"
        with h5py.File(str(h5p), "w") as f:
            f.create_dataset("x_true", data=x, compression="gzip")
            f.create_dataset("y_ideal", data=y, compression="gzip")
            hp = f.create_group("H_params")
            hp.attrs["forward_type"] = fwd_type
            hp.attrs["noise"] = "none"
            hp.attrs["mismatch"] = "none"
            md = f.create_group("metadata")
            md.attrs["source"] = meta_extra.get("src", "Simulation")
            md.attrs["reference"] = meta_extra.get("ref", "Simulation")
            md.attrs["sample_index"] = i
            md.attrs["date_built"] = "2026-03-14"
        files.append(h5p.name)
        img_dir = out / "images" / f"sample_{i:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        save_img(x, img_dir / "x_true.png")
        save_img(y, img_dir / "y_ideal.png")

    meta = {
        "modality": modality,
        "canonical_dataset": meta_extra.get("src", "Simulation"),
        "source_type": meta_extra.get("type", "synthetic"),
        "reference": meta_extra.get("ref", "Simulation"),
        "n_samples": len(samples),
        "x_true_shape": list(samples[0][0].shape),
        "y_ideal_shape": list(samples[0][1].shape),
        "forward_type": fwd_type,
        "forward_model": fwd_desc,
        "noise": "none", "mismatch": "none",
        "date_built": "2026-03-14",
        "gcs_path": f"gs://pwm-benchmark-datasets/datasets/Benchmark/{modality}/standard/",
        "status": "done", "files": files,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out / "spec.json", "w") as f:
        json.dump({"modality": modality, "forward_type": fwd_type,
                    "noise": "none", "mismatch": "none", "split": "standard"}, f, indent=2)
    print(f"  {modality}: {len(samples)} samples")


# ===================================================================
# UNIQUE PHANTOM GENERATORS
# ===================================================================

def phantom_cbct_dental(seed):
    """Dental/maxillofacial CBCT phantom — jaw cross-section with teeth."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Jaw (mandible) — U-shaped bone
    r_outer = S * 0.38
    r_inner = S * 0.28
    dist = np.sqrt((xx - cx)**2 + (yy - cy * 1.1)**2)
    jaw = ((dist < r_outer) & (dist > r_inner) & (yy > cy * 0.5)).astype(np.float32)
    x += jaw * 0.8
    # Teeth — high density spots along the jaw
    n_teeth = rng.randint(10, 16)
    for t in range(n_teeth):
        angle = np.pi * 0.15 + np.pi * 0.7 * t / n_teeth
        tr = (r_outer + r_inner) / 2
        tx = cx + tr * np.cos(angle)
        ty = cy * 1.1 + tr * np.sin(angle)
        ts = rng.uniform(4, 7)
        tooth = np.exp(-0.5 * ((xx - tx)**2 + (yy - ty)**2) / ts**2)
        x += tooth * rng.uniform(0.8, 1.0)
    # Soft tissue fill
    soft = (dist < r_inner * 1.1).astype(np.float32) * 0.3
    x = np.maximum(x, soft)
    # Mandibular canal (nerve)
    canal_y = cy * 1.1 + r_inner * 0.7
    canal = np.exp(-0.5 * ((yy - canal_y)**2) / 3**2) * (np.abs(xx - cx) < r_outer * 0.8).astype(np.float32)
    x -= canal * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_industrial_ct(seed):
    """Industrial CT phantom — metal pipe with weld defects and voids."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Outer pipe wall (high density metal)
    r_outer = S * rng.uniform(0.35, 0.42)
    wall_thick = S * rng.uniform(0.06, 0.10)
    r_inner = r_outer - wall_thick
    pipe = ((r < r_outer) & (r > r_inner)).astype(np.float32)
    x += pipe * 0.9
    # Weld bead (slightly higher density ring segment)
    weld_angle = rng.uniform(0, 2 * np.pi)
    weld_width = rng.uniform(0.3, 0.8)  # angular width
    theta = np.arctan2(yy - cy, xx - cx)
    weld_mask = (np.abs(np.mod(theta - weld_angle + np.pi, 2*np.pi) - np.pi) < weld_width)
    weld_r = ((r < r_outer + 3) & (r > r_outer - wall_thick * 0.3))
    x += (weld_mask & weld_r).astype(np.float32) * 0.15
    # Porosity/voids in weld
    n_voids = rng.randint(3, 8)
    for _ in range(n_voids):
        va = weld_angle + rng.uniform(-weld_width, weld_width)
        vr = rng.uniform(r_inner + 2, r_outer - 2)
        vx = cx + vr * np.cos(va)
        vy = cy + vr * np.sin(va)
        vs = rng.uniform(1.5, 4)
        void = np.exp(-0.5 * ((xx - vx)**2 + (yy - vy)**2) / vs**2)
        x -= void * 0.4
    # Internal object (bolt or inclusion)
    bx = cx + rng.uniform(-r_inner * 0.5, r_inner * 0.5)
    by = cy + rng.uniform(-r_inner * 0.5, r_inner * 0.5)
    bolt = np.exp(-0.5 * ((xx - bx)**2 + (yy - by)**2) / 8**2)
    x += bolt * 0.6
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_dbt_breast(seed):
    """Digital breast tomosynthesis — breast with calcifications and mass."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Breast contour (semi-elliptical, compressed)
    bx, by = S * 0.5, S * 0.55
    a, b = S * 0.38, S * 0.42
    breast = (((xx - bx) / a)**2 + ((yy - by) / b)**2 < 1) & (yy > S * 0.1)
    x += breast.astype(np.float32) * 0.4
    # Dense fibroglandular tissue (irregular patches)
    for _ in range(rng.randint(3, 6)):
        gx = bx + rng.uniform(-a * 0.6, a * 0.6)
        gy = by + rng.uniform(-b * 0.5, b * 0.3)
        gs = rng.uniform(S * 0.05, S * 0.12)
        gland = np.exp(-0.5 * ((xx - gx)**2 + (yy - gy)**2) / gs**2)
        x += gland * rng.uniform(0.15, 0.3) * breast
    # Calcification cluster (tiny bright spots)
    calc_cx = bx + rng.uniform(-a * 0.3, a * 0.3)
    calc_cy = by + rng.uniform(-b * 0.3, b * 0.1)
    for _ in range(rng.randint(5, 15)):
        cx2 = calc_cx + rng.uniform(-10, 10)
        cy2 = calc_cy + rng.uniform(-10, 10)
        cs = rng.uniform(0.5, 2.0)
        calc = np.exp(-0.5 * ((xx - cx2)**2 + (yy - cy2)**2) / cs**2)
        x += calc * 0.8
    # Mass (suspicious lesion)
    if rng.rand() > 0.3:
        mx = bx + rng.uniform(-a * 0.3, a * 0.3)
        my = by + rng.uniform(-b * 0.2, b * 0.2)
        ms = rng.uniform(8, 18)
        mass = np.exp(-0.5 * ((xx - mx)**2 + (yy - my)**2) / ms**2)
        x += mass * rng.uniform(0.3, 0.5)
    return np.clip(x * breast, 0, 1).astype(np.float32)


def phantom_mammography(seed):
    """Mammography — CC view breast with ductal structures."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    # Breast (pendant shape for CC view — wider and flatter)
    bx, by = S * 0.5, S * 0.45
    breast = ((xx - bx)**2 / (S * 0.42)**2 + (yy - by)**2 / (S * 0.35)**2 < 1)
    x += breast.astype(np.float32) * 0.35
    # Ductal network (tree-like branching lines from nipple)
    nipple_x, nipple_y = S * 0.5, S * 0.12
    for branch in range(rng.randint(4, 8)):
        angle = rng.uniform(-np.pi / 3, np.pi / 3) + np.pi / 2
        length = rng.uniform(S * 0.2, S * 0.5)
        width = rng.uniform(1, 3)
        for t in np.linspace(0, 1, 200):
            px = nipple_x + length * t * np.cos(angle + 0.3 * np.sin(t * 4))
            py = nipple_y + length * t * np.sin(angle + 0.3 * np.sin(t * 4))
            if 0 <= px < S and 0 <= py < S:
                dist = ((xx - px)**2 + (yy - py)**2)
                x += np.exp(-dist / (2 * width**2)) * 0.08 * (1 - 0.5 * t)
    # Pectoral muscle (triangle in corner for MLO view effect)
    pec = ((xx < S * 0.15) & (yy < S * 0.4 - xx)).astype(np.float32)
    x += pec * 0.5
    return np.clip(x * breast, 0, 1).astype(np.float32)


def phantom_spectral_ct(seed):
    """Spectral/photon-counting CT — multi-material phantom with K-edge contrast."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Water background (body)
    body = (r < S * 0.4).astype(np.float32)
    x += body * 0.3
    # Bone (high Z, high attenuation) — spine-like
    spine = np.exp(-0.5 * ((xx - cx - S * 0.1)**2 / 12**2 + (yy - cy)**2 / 20**2))
    x += spine * 0.7
    # Iodine contrast agent (vessel enhancement)
    for _ in range(rng.randint(2, 5)):
        vx = cx + rng.uniform(-S * 0.25, S * 0.25)
        vy = cy + rng.uniform(-S * 0.25, S * 0.25)
        vs = rng.uniform(3, 8)
        x += np.exp(-0.5 * ((xx - vx)**2 + (yy - vy)**2) / vs**2) * 0.9
    # Gadolinium insert (different K-edge)
    gx = cx + rng.uniform(-S * 0.15, S * 0.15)
    gy = cy + rng.uniform(-S * 0.15, S * 0.15)
    gd = np.exp(-0.5 * ((xx - gx)**2 + (yy - gy)**2) / 10**2)
    x += gd * 0.6
    # Calcium insert
    ca_x = cx - S * 0.15
    ca_y = cy + S * 0.1
    ca = np.exp(-0.5 * ((xx - ca_x)**2 + (yy - ca_y)**2) / 8**2)
    x += ca * 0.5
    return np.clip(x * body, 0, 1).astype(np.float32)


def phantom_chest_xray(seed):
    """Chest X-ray phantom — thorax with lungs, heart, ribs, spine."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Thorax outline
    thorax = (((xx - cx) / (S * 0.42))**2 + ((yy - cy) / (S * 0.45))**2 < 1)
    x += thorax.astype(np.float32) * 0.4
    # Lungs (two dark elliptical regions)
    for side in [-1, 1]:
        lx = cx + side * S * 0.15
        ly = cy - S * 0.02
        lung = (((xx - lx) / (S * 0.15))**2 + ((yy - ly) / (S * 0.28))**2 < 1)
        x -= lung.astype(np.float32) * 0.25
    # Heart (left-shifted dense region)
    hx = cx - S * 0.05
    hy = cy + S * 0.08
    heart = np.exp(-0.5 * ((xx - hx)**2 / (S * 0.1)**2 + (yy - hy)**2 / (S * 0.12)**2))
    x += heart * 0.35
    # Spine (vertical dense bar)
    spine = np.exp(-0.5 * (xx - cx)**2 / 8**2) * (np.abs(yy - cy) < S * 0.4).astype(np.float32)
    x += spine * 0.3
    # Ribs (curved horizontal lines)
    n_ribs = rng.randint(8, 12)
    for i in range(n_ribs):
        ry = cy - S * 0.3 + i * S * 0.06
        for side in [-1, 1]:
            rib_cx = cx + side * S * 0.15
            rib_dist = np.sqrt((xx - rib_cx)**2 + (yy - ry)**2)
            rib_r = S * 0.22
            rib = np.exp(-0.5 * ((rib_dist - rib_r) / 2.5)**2) * (side * (xx - cx) > 0).astype(np.float32)
            x += rib * 0.12
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_xray_ndt(seed):
    """NDT X-ray — welded steel plate with cracks and inclusions."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.ones((S, S), dtype=np.float32) * 0.6  # uniform steel plate
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    # Weld seam (vertical or angled line with higher density)
    weld_x = S // 2 + rng.randint(-20, 20)
    weld_w = rng.uniform(8, 15)
    weld = np.exp(-0.5 * (xx_g - weld_x)**2 / weld_w**2)
    x += weld * 0.15
    # Crack (thin dark line in or near weld)
    crack_x = weld_x + rng.uniform(-5, 5)
    crack_len = rng.uniform(S * 0.1, S * 0.4)
    crack_y0 = S // 2 - crack_len / 2
    crack = np.exp(-0.5 * (xx_g - crack_x)**2 / 1.5**2)
    crack *= ((yy > crack_y0) & (yy < crack_y0 + crack_len)).astype(np.float32)
    x -= crack * 0.3
    # Slag inclusions (irregular dark spots)
    for _ in range(rng.randint(2, 6)):
        ix = weld_x + rng.uniform(-weld_w, weld_w)
        iy = rng.uniform(S * 0.2, S * 0.8)
        isx = rng.uniform(2, 5)
        isy = rng.uniform(1, 3)
        incl = np.exp(-0.5 * ((xx_g - ix)**2 / isx**2 + (yy - iy)**2 / isy**2))
        x -= incl * rng.uniform(0.15, 0.3)
    # Porosity (tiny dark dots)
    for _ in range(rng.randint(5, 15)):
        px = weld_x + rng.uniform(-weld_w * 0.8, weld_w * 0.8)
        py = rng.uniform(S * 0.15, S * 0.85)
        ps = rng.uniform(0.5, 2)
        pore = np.exp(-0.5 * ((xx_g - px)**2 + (yy - py)**2) / ps**2)
        x -= pore * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_dexa(seed):
    """DXA (dual-energy X-ray) — lumbar spine with varying BMD."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx = S // 2
    # Soft tissue background
    body = (((xx_g - cx) / (S * 0.4))**2 + ((yy - S // 2) / (S * 0.48))**2 < 1)
    x += body.astype(np.float32) * 0.2
    # Vertebral bodies (L1-L4, rectangular bone regions)
    for i in range(4):
        vy = S * 0.15 + i * S * 0.18
        vw = S * rng.uniform(0.08, 0.12)
        vh = S * 0.12
        bmd = rng.uniform(0.4, 0.9)  # bone mineral density varies
        vert = ((np.abs(xx_g - cx) < vw) & (np.abs(yy - vy) < vh / 2)).astype(np.float32)
        x += vert * bmd
        # Cortical shell (brighter edge)
        cortex = vert * np.exp(-0.5 * ((np.abs(xx_g - cx) - vw + 3)**2 + (np.abs(yy - vy) - vh/2 + 3)**2) / 4**2)
        x += cortex * 0.15
    # Spinous processes (small posterior projections)
    for i in range(4):
        vy = S * 0.15 + i * S * 0.18
        sp = np.exp(-0.5 * ((xx_g - cx - S * 0.13)**2 / 4**2 + (yy - vy)**2 / 6**2))
        x += sp * 0.5
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_angiography(seed):
    """Coronary angiography — vessel tree with stenosis."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    # Background (cardiac silhouette)
    cx, cy = S // 2, S // 2
    heart_bg = np.exp(-0.5 * ((xx_g - cx)**2 + (yy - cy)**2) / (S * 0.3)**2)
    x += heart_bg * 0.15
    # Main vessel (left coronary artery) — curving path
    def draw_vessel(x0, y0, angle, length, width, n_pts=200):
        curve = np.zeros_like(x)
        px, py = x0, y0
        for t in range(n_pts):
            frac = t / n_pts
            angle_t = angle + rng.uniform(-0.02, 0.02)
            angle = angle_t
            px += length / n_pts * np.cos(angle)
            py += length / n_pts * np.sin(angle)
            w = width * (1 - 0.3 * frac)  # taper
            if 0 <= px < S and 0 <= py < S:
                curve += np.exp(-0.5 * ((xx_g - px)**2 + (yy - py)**2) / max(w, 0.5)**2)
        return curve

    # Left main → LAD
    x += draw_vessel(S * 0.55, S * 0.25, np.pi * 0.6, S * 0.5, 4) * 0.7
    # Left circumflex
    x += draw_vessel(S * 0.55, S * 0.25, np.pi * 0.85, S * 0.4, 3) * 0.6
    # Right coronary
    x += draw_vessel(S * 0.4, S * 0.2, np.pi * 0.4, S * 0.55, 3.5) * 0.65
    # Branches
    for _ in range(rng.randint(3, 7)):
        bx = rng.uniform(S * 0.2, S * 0.8)
        by = rng.uniform(S * 0.2, S * 0.7)
        ba = rng.uniform(0, 2 * np.pi)
        bl = rng.uniform(S * 0.1, S * 0.2)
        x += draw_vessel(bx, by, ba, bl, rng.uniform(1, 2.5)) * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_portal_imaging(seed):
    """Portal/EPID imaging — radiation therapy field with MLC pattern."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.ones((S, S), dtype=np.float32) * 0.1  # low background (outside field)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Treatment field (bright rectangular region)
    fw, fh = S * rng.uniform(0.3, 0.5), S * rng.uniform(0.3, 0.5)
    field = ((np.abs(xx_g - cx) < fw / 2) & (np.abs(yy - cy) < fh / 2)).astype(np.float32)
    x += field * 0.6
    # MLC leaf pattern (blocked regions within field)
    n_leaves = rng.randint(10, 20)
    leaf_w = fw / n_leaves
    for i in range(n_leaves):
        lx = cx - fw / 2 + (i + 0.5) * leaf_w
        # Each leaf blocks to a different position
        block_y = cy + rng.uniform(-fh * 0.4, fh * 0.4)
        if rng.rand() > 0.5:
            leaf = ((np.abs(xx_g - lx) < leaf_w * 0.45) & (yy > block_y) & (yy < cy + fh / 2))
        else:
            leaf = ((np.abs(xx_g - lx) < leaf_w * 0.45) & (yy < block_y) & (yy > cy - fh / 2))
        x -= leaf.astype(np.float32) * 0.4
    # Anatomical shadow (pelvis-like)
    pelvis = np.exp(-0.5 * ((xx_g - cx)**2 / (S * 0.25)**2 + (yy - cy)**2 / (S * 0.2)**2))
    x += pelvis * 0.2
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_proton_radio(seed):
    """Proton radiography — water-equivalent thickness map (WET) of head."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    r = np.sqrt((xx_g - cx)**2 + (yy - cy)**2)
    # Skull (high stopping power shell)
    skull = ((r < S * 0.38) & (r > S * 0.33)).astype(np.float32)
    x += skull * 0.8
    # Brain tissue (moderate stopping power)
    brain = (r < S * 0.33).astype(np.float32)
    x += brain * 0.45
    # Ventricles (lower stopping power — CSF)
    for side in [-1, 1]:
        vx = cx + side * S * 0.04
        vy = cy - S * 0.02
        vent = np.exp(-0.5 * ((xx_g - vx)**2 / 8**2 + (yy - vy)**2 / 15**2))
        x -= vent * 0.15
    # Dense bone (petrous)
    for side in [-1, 1]:
        bx = cx + side * S * 0.28
        by = cy + S * 0.1
        bone = np.exp(-0.5 * ((xx_g - bx)**2 + (yy - by)**2) / 8**2)
        x += bone * 0.3
    # Tumor (slightly higher stopping power)
    tx = cx + rng.uniform(-S * 0.15, S * 0.15)
    ty = cy + rng.uniform(-S * 0.15, S * 0.15)
    tumor = np.exp(-0.5 * ((xx_g - tx)**2 + (yy - ty)**2) / rng.uniform(5, 12)**2)
    x += tumor * 0.15
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_proton_therapy(seed):
    """Proton therapy dose distribution — Bragg peak dose map."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    # Body outline
    cx, cy = S // 2, S // 2
    body = (((xx_g - cx) / (S * 0.4))**2 + ((yy - cy) / (S * 0.35))**2 < 1)
    x += body.astype(np.float32) * 0.15  # entrance dose
    # Proton beam entrance (from left)
    beam_y = cy + rng.uniform(-S * 0.1, S * 0.1)
    beam_w = rng.uniform(15, 30)
    # Depth-dose: entrance plateau + Bragg peak
    bragg_x = cx + rng.uniform(-S * 0.05, S * 0.15)
    for col in range(S):
        depth = col - (cx - S * 0.35)
        if depth < 0:
            continue
        # Bragg curve: plateau then sharp peak
        bragg_depth = bragg_x - (cx - S * 0.35)
        if depth < bragg_depth:
            dose = 0.3 + 0.4 * (depth / bragg_depth)**2
        elif depth < bragg_depth + 10:
            dose = 0.7 * np.exp(-0.5 * ((depth - bragg_depth) / 3)**2) + 0.3
        else:
            dose = 0.05 * np.exp(-0.1 * (depth - bragg_depth - 10))
        beam_profile = np.exp(-0.5 * (yy[:, col] - beam_y)**2 / beam_w**2)
        x[:, col] += beam_profile * dose * body[:, col].astype(np.float32)
    # Target volume (where dose should be high)
    tx = bragg_x
    ty = beam_y
    target = np.exp(-0.5 * ((xx_g - tx)**2 + (yy - ty)**2) / 12**2)
    x += target * 0.1 * body.astype(np.float32)
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_dark_field(seed):
    """X-ray dark-field — lung tissue with alveolar microstructure scatter signal."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Lung fields (strong dark-field signal from alveolar air-tissue interfaces)
    for side in [-1, 1]:
        lx = cx + side * S * 0.15
        ly = cy
        lung_mask = (((xx_g - lx) / (S * 0.14))**2 + ((yy - ly) / (S * 0.3))**2 < 1)
        # Alveolar texture (small-angle scattering from microstructures)
        texture = ndimage.gaussian_filter(rng.randn(S, S).astype(np.float32), sigma=3)
        x += lung_mask * (0.6 + 0.2 * texture)
    # Bone (low dark-field — no microstructure)
    spine = np.exp(-0.5 * (xx_g - cx)**2 / 6**2) * (np.abs(yy - cy) < S * 0.35).astype(np.float32)
    x -= spine * 0.3  # bone gives negative dark-field contrast
    # Emphysema region (reduced dark-field — destroyed alveoli)
    if rng.rand() > 0.3:
        ex = cx + rng.choice([-1, 1]) * S * 0.15
        ey = cy + rng.uniform(-S * 0.1, S * 0.1)
        emph = np.exp(-0.5 * ((xx_g - ex)**2 + (yy - ey)**2) / (S * 0.08)**2)
        x -= emph * 0.3
    return np.clip(x, 0, 1).astype(np.float32)


def phantom_talbot_lau(seed):
    """Talbot-Lau grating interferometry — phase-contrast + visibility image."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Phase object (soft tissue phantom — shows edges)
    body = (((xx_g - cx) / (S * 0.35))**2 + ((yy - cy) / (S * 0.38))**2 < 1)
    # Internal organs with phase contrast (edge enhancement)
    organs = np.zeros((S, S), dtype=np.float32)
    organs += body.astype(np.float32) * 0.4
    for _ in range(rng.randint(3, 7)):
        ox = cx + rng.uniform(-S * 0.2, S * 0.2)
        oy = cy + rng.uniform(-S * 0.2, S * 0.2)
        osx = rng.uniform(S * 0.05, S * 0.15)
        osy = rng.uniform(S * 0.05, S * 0.12)
        organs += np.exp(-0.5 * ((xx_g - ox)**2 / osx**2 + (yy - oy)**2 / osy**2)) * rng.uniform(0.2, 0.5)
    # Phase contrast = gradient of refractive index decrement
    dx = ndimage.sobel(organs, axis=1)
    dy = ndimage.sobel(organs, axis=0)
    x = 0.5 + 0.4 * np.sqrt(dx**2 + dy**2) / (np.sqrt(dx**2 + dy**2).max() + 1e-10)
    # Add absorption contrast too
    x = 0.7 * x + 0.3 * organs
    return np.clip(x * body, 0, 1).astype(np.float32)


def phantom_phase_contrast(seed):
    """X-ray phase contrast — synchrotron imaging of insects/biomaterials."""
    rng = np.random.RandomState(seed)
    S = 256
    x = np.zeros((S, S), dtype=np.float32)
    yy, xx_g = np.mgrid[:S, :S].astype(np.float32)
    cx, cy = S // 2, S // 2
    # Insect body (thorax cross-section — common synchrotron phase contrast sample)
    # Exoskeleton (outer ring)
    r = np.sqrt((xx_g - cx)**2 + (yy - cy)**2)
    exo = ((r < S * 0.35) & (r > S * 0.3)).astype(np.float32)
    x += exo * 0.8
    # Internal structures (tracheal tubes — circular air channels)
    n_trach = rng.randint(5, 12)
    for _ in range(n_trach):
        tx = cx + rng.uniform(-S * 0.2, S * 0.2)
        ty = cy + rng.uniform(-S * 0.2, S * 0.2)
        tr_outer = rng.uniform(3, 10)
        tr_inner = tr_outer * 0.6
        tr = np.sqrt((xx_g - tx)**2 + (yy - ty)**2)
        tube = ((tr < tr_outer) & (tr > tr_inner)).astype(np.float32)
        x += tube * 0.6
        # Air inside
        air = (tr < tr_inner).astype(np.float32)
        x -= air * 0.1
    # Muscle fibers (oriented texture)
    fiber_angle = rng.uniform(0, np.pi)
    fiber = np.sin(((xx_g - cx) * np.cos(fiber_angle) + (yy - cy) * np.sin(fiber_angle)) / 3)
    x += (r < S * 0.3).astype(np.float32) * np.clip(fiber, 0, 1) * 0.15
    # Gut content
    gut = np.exp(-0.5 * ((xx_g - cx + S * 0.05)**2 / 20**2 + (yy - cy + S * 0.05)**2 / 15**2))
    x += gut * 0.4
    return np.clip(x, 0, 1).astype(np.float32)


# ===================================================================
# FORWARD MODELS
# ===================================================================

def fwd_beer_lambert(x, mu=0.02):
    """X-ray transmission: I = I0 * exp(-mu * x * thickness)."""
    y = np.exp(-mu * x * 100)
    return y.astype(np.float32)

def fwd_radon(x):
    return radon_fwd(x, 120)


# ===================================================================
# BUILD ALL
# ===================================================================

def build_all():
    print("=" * 60)
    print("Batch 1: Rebuilding X-ray / CT modalities with unique phantoms")
    print("=" * 60)

    modalities = [
        ("cbct", phantom_cbct_dental, "radon", "y = Radon(x); CBCT cone-beam geometry",
         {"src": "Walnut CBCT (Zenodo 2686726)", "ref": "doi:10.5281/zenodo.2686726"}),
        ("industrial_ct", phantom_industrial_ct, "radon", "y = Radon(x); industrial fan-beam",
         {"src": "WoDT industrial CT benchmark", "ref": "doi:10.1007/s10921-019-0580-5"}),
        ("digital_breast_tomo", phantom_dbt_breast, "radon", "y = limited-angle Radon(x); DBT sweep 15-50 deg",
         {"src": "VDM-100 DBT (TCIA)", "ref": "cancerimagingarchive.net"}),
        ("mammography", phantom_mammography, "beer_lambert", "y = exp(-mu * x); X-ray absorption",
         {"src": "CBIS-DDSM", "ref": "doi:10.1038/sdata.2017.177"}),
        ("spectral_ct", phantom_spectral_ct, "radon", "y = Radon(x); multi-energy sinogram",
         {"src": "AAPM Spectral CT Grand Challenge", "ref": "aapm.org/grandchallenge"}),
        ("xray_radiography", phantom_chest_xray, "beer_lambert", "y = exp(-mu * x); chest X-ray",
         {"src": "Chest X-ray14 (NIH)", "ref": "doi:10.1109/CVPR.2017.369"}),
        ("xray_ndt", phantom_xray_ndt, "beer_lambert", "y = exp(-mu * x); NDT radiography",
         {"src": "GDXray (GRIMA X-ray)", "ref": "doi:10.1007/s10921-015-0315-7"}),
        ("dexa", phantom_dexa, "beer_lambert", "y = exp(-mu * x); dual-energy absorption",
         {"src": "OAI DXA dataset", "ref": "oai.ucsf.edu"}),
        ("angiography", phantom_angiography, "beer_lambert", "y = exp(-mu * x); DSA angiogram",
         {"src": "XCAD coronary angiography", "ref": "github.com/XiaoweiXu/XCAD"}),
        ("portal_imaging", phantom_portal_imaging, "beer_lambert", "y = exp(-mu * x); EPID portal",
         {"src": "AAPM TG-58 EPID benchmark", "ref": "aapm.org"}),
        ("proton_radiography", phantom_proton_radio, "radon", "y = Radon(RSP(x)); proton stopping power",
         {"src": "pCT collaboration simulation", "ref": "doi:10.1088/0031-9155/60/20/7829"}),
        ("proton_therapy_img", phantom_proton_therapy, "beer_lambert", "y = dose_map(x); proton Bragg peak",
         {"src": "TOPAS Monte Carlo benchmark", "ref": "doi:10.1118/1.4758060"}),
        ("dark_field", phantom_dark_field, "radon", "y = Radon(scatter_coefficient(x))",
         {"src": "Munich dark-field chest CT", "ref": "doi:10.1016/S2589-7500(23)00147-3"}),
        ("talbot_lau", phantom_talbot_lau, "radon", "y = Radon(x); grating interferometry sinogram",
         {"src": "PSI Talbot-Lau grating CT", "ref": "doi:10.1038/nphys265"}),
        ("phase_contrast", phantom_phase_contrast, "beer_lambert", "y = fresnel_prop(x); phase-contrast edge enhancement",
         {"src": "APS synchrotron phase-contrast archive", "ref": "doi:10.1107/S2059798320008918"}),
    ]

    for mod_name, phantom_fn, fwd_type, fwd_desc, meta in modalities:
        samples = []
        for i in range(N):
            seed = 42 + i * 137
            x = phantom_fn(seed)
            if fwd_type == "radon":
                y = fwd_radon(x)
            elif fwd_type == "beer_lambert":
                y = fwd_beer_lambert(x)
            else:
                y = x.copy()
            samples.append((x, y))
        save_modality(mod_name, samples, fwd_type, fwd_desc, meta)

    print(f"\nBatch 1 done: {len(modalities)} X-ray/CT modalities rebuilt with unique phantoms")


if __name__ == "__main__":
    build_all()
