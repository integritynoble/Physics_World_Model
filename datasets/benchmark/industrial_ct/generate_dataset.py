#!/usr/bin/env python3
"""Generate Industrial CT benchmark dataset.

Forward model (2D parallel-beam industrial CT with beam hardening):
    y = Radon(x) * exp(-beam_hardening) + scatter + noise

where:
    x       : material attenuation map (ground truth, 256x256)
    Radon   : parallel-beam Radon transform (360 angles)
    beam_hardening : polychromatic beam hardening artifact  p_eff = p + bh*p^2 + bh2*p^3
    scatter : low-frequency scatter background (fraction of direct signal)
    noise   : Poisson noise (high-dose industrial setting) + readout

Objects are manufactured parts:
    - Metal cylinders (steel, aluminum, titanium)
    - Bolts / fasteners with threads
    - Welds with porosity / voids / cracks
    - Composite assemblies (metal + plastic + air gaps)

Key challenges:
    - Beam hardening artifacts from high-Z materials (streaks between dense regions)
    - Metal artifacts (photon starvation in metal shadow)
    - Internal voids / cracks in welds (defect detection)

Mismatch parameters (per tier):
    beam_hardening_order : BH polynomial order mismatch (0.0 - 0.30)
    scatter_fraction     : scatter / total (0.01 - 0.15)
    source_blur          : focal spot blur sigma (0.0 - 3.0 px)
    detector_efficiency  : relative detector gain variation (0.90 - 1.10)

Phantoms:
    Public  : 12 samples (4 cylinder + 4 bolt + 4 weld)
    Dev     : 20 samples (augmented variants with rotation/flip/zoom)
    Hidden  : 20 samples (adversarial: extreme BH, micro-cracks, multi-material)

Seed offsets:
    public  : 0
    dev     : 10000
    hidden  : 20000

Usage:
    python3 generate_dataset.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, zoom as nd_zoom

# Import radon_transform from pet.generate_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pet"))
from generate_dataset import radon_transform, fbp_reconstruct  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

# -- Geometry -----------------------------------------------------------------

IMAGE_SIZE = 256
N_ANGLES = 360
N_DET = 367  # ceil(sqrt(2) * 256)

# -- Material attenuation coefficients (normalised to [0, 1] scale) -----------
# Industrial CT uses much higher attenuation than medical CT.
# We work in normalised units where:
#   air     = 0.00
#   plastic = 0.10 - 0.20
#   aluminum = 0.35 - 0.50
#   titanium = 0.55 - 0.70
#   steel   = 0.70 - 0.90
#   tungsten = 0.90 - 1.00

MATERIALS = {
    "air":       (0.00, 0.00),
    "plastic":   (0.10, 0.20),
    "rubber":    (0.08, 0.15),
    "aluminum":  (0.35, 0.50),
    "titanium":  (0.55, 0.70),
    "steel":     (0.70, 0.90),
    "tungsten":  (0.90, 1.00),
    "copper":    (0.60, 0.75),
    "ceramic":   (0.25, 0.40),
}

# -- Mismatch ranges per tier -------------------------------------------------

SPEC = {
    "public": {
        "beam_hardening_order":  {"min": 0.00, "max": 0.08,  "unit": ""},
        "scatter_fraction":      {"min": 0.01, "max": 0.05,  "unit": ""},
        "source_blur":           {"min": 0.0,  "max": 1.0,   "unit": "pixels"},
        "detector_efficiency":   {"min": 0.95, "max": 1.05,  "unit": "relative"},
    },
    "dev": {
        "beam_hardening_order":  {"min": 0.00, "max": 0.15,  "unit": ""},
        "scatter_fraction":      {"min": 0.01, "max": 0.08,  "unit": ""},
        "source_blur":           {"min": 0.0,  "max": 2.0,   "unit": "pixels"},
        "detector_efficiency":   {"min": 0.92, "max": 1.08,  "unit": "relative"},
    },
    "hidden": {
        "beam_hardening_order":  {"min": 0.00, "max": 0.30,  "unit": ""},
        "scatter_fraction":      {"min": 0.01, "max": 0.15,  "unit": ""},
        "source_blur":           {"min": 0.0,  "max": 3.0,   "unit": "pixels"},
        "detector_efficiency":   {"min": 0.90, "max": 1.10,  "unit": "relative"},
    },
}


# -- Geometry helpers ---------------------------------------------------------

def _circle_mask(H: int, W: int, cx: float, cy: float,
                 r: float) -> np.ndarray:
    """Circle mask in normalised [-1, 1] coordinates."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    return ((xx - cx)**2 + (yy - cy)**2) <= r**2


def _ellipse_mask(H: int, W: int, cx: float, cy: float,
                  a: float, b: float, angle_deg: float = 0.0) -> np.ndarray:
    """Ellipse mask in normalised [-1, 1] coordinates."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (xr / a)**2 + (yr / b)**2 <= 1.0


def _rect_mask(H: int, W: int, cx: float, cy: float,
               hw: float, hh: float, angle_deg: float = 0.0) -> np.ndarray:
    """Rotated rectangle mask in normalised [-1, 1] coordinates."""
    yy = np.linspace(-1.0, 1.0, H)[:, None]
    xx = np.linspace(-1.0, 1.0, W)[None, :]
    ca, sa = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    yr = (yy - cy) * ca - (xx - cx) * sa
    xr = (yy - cy) * sa + (xx - cx) * ca
    return (np.abs(xr) <= hw) & (np.abs(yr) <= hh)


def _annulus_mask(H: int, W: int, cx: float, cy: float,
                  r_inner: float, r_outer: float) -> np.ndarray:
    """Annular ring mask."""
    return _circle_mask(H, W, cx, cy, r_outer) & ~_circle_mask(H, W, cx, cy, r_inner)


# -- Phantom generators: manufactured parts ----------------------------------

def make_cylinder_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Metal cylinder with internal features: holes, voids, inclusions.

    Simulates a machined metal cylinder (steel or aluminum) cross-section
    with drilled holes, casting voids, and possible inclusions.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)

    # Choose primary material
    materials = ["steel", "aluminum", "titanium"]
    mat = materials[variant % len(materials)]
    mat_lo, mat_hi = MATERIALS[mat]
    mat_val = rng.uniform(mat_lo, mat_hi)

    # Outer cylinder
    outer_r = rng.uniform(0.35, 0.45)
    outer = _circle_mask(H, W, 0.0, 0.0, outer_r)
    img[outer] = mat_val

    # Inner bore (hollow cylinder)
    if rng.random() < 0.6:
        bore_r = rng.uniform(0.08, 0.15)
        bore = _circle_mask(H, W, 0.0, 0.0, bore_r)
        img[bore] = 0.0  # air

    # Drilled holes (2-5)
    n_holes = rng.integers(2, 6)
    for _ in range(n_holes):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.12, outer_r - 0.05)
        hx = dist * np.cos(angle)
        hy = dist * np.sin(angle)
        hr = rng.uniform(0.015, 0.04)
        hole = _circle_mask(H, W, hx, hy, hr)
        img[hole & outer] = 0.0  # drilled through

    # Internal voids (casting defects) - 0-3 small voids
    n_voids = rng.integers(0, 4)
    for _ in range(n_voids):
        vx = rng.uniform(-outer_r * 0.6, outer_r * 0.6)
        vy = rng.uniform(-outer_r * 0.6, outer_r * 0.6)
        vr = rng.uniform(0.005, 0.02)
        void = _ellipse_mask(H, W, vx, vy, vr, vr * rng.uniform(0.5, 1.5),
                             rng.uniform(0, 180))
        img[void & outer] = rng.uniform(0.0, 0.03)  # near air (porosity)

    # Density variation (material inhomogeneity)
    noise = gaussian_filter(rng.standard_normal((H, W)), sigma=15) * 0.02
    img[outer] += noise[outer]
    img = np.clip(img, 0.0, 1.0)

    return img, f"cylinder_{mat}_{variant:02d}"


def make_bolt_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Cross-section of a bolt/fastener with threads and nut.

    Simulates hex bolt head + threaded shaft cross-section with
    potential thread damage or galling.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)

    mat_val = rng.uniform(*MATERIALS["steel"])

    # Bolt shaft (central circle)
    shaft_r = rng.uniform(0.10, 0.15)
    shaft = _circle_mask(H, W, 0.0, 0.0, shaft_r)
    img[shaft] = mat_val

    # Thread ridges (concentric rings)
    thread_pitch = rng.uniform(0.008, 0.015)
    n_threads = int((0.25 - shaft_r) / thread_pitch)
    for i in range(n_threads):
        r_inner = shaft_r + i * thread_pitch
        r_outer = r_inner + thread_pitch * 0.6
        ring = _annulus_mask(H, W, 0.0, 0.0, r_inner, r_outer)
        img[ring] = mat_val * rng.uniform(0.85, 1.0)

    # Hex head (if cross-section through head)
    if variant % 2 == 0:
        # Hexagonal approximation with 6 rectangles
        head_r = rng.uniform(0.28, 0.38)
        for k in range(6):
            angle = k * 60 + rng.uniform(-3, 3)
            rect = _rect_mask(H, W, 0.0, 0.0, head_r, head_r * 0.5, angle)
            hex_region = rect & _circle_mask(H, W, 0.0, 0.0, head_r)
            img[hex_region] = mat_val
    else:
        # Nut (annular with hex profile)
        nut_outer = rng.uniform(0.30, 0.40)
        nut_inner = shaft_r + n_threads * thread_pitch + 0.01
        nut = _annulus_mask(H, W, 0.0, 0.0, nut_inner, nut_outer)
        img[nut] = mat_val * rng.uniform(0.90, 1.0)

    # Washer (offset ring)
    if rng.random() < 0.4:
        w_cx = rng.uniform(-0.05, 0.05)
        w_cy = rng.uniform(0.30, 0.45)
        w_r_out = rng.uniform(0.08, 0.12)
        w_r_in = w_r_out * 0.5
        washer = _annulus_mask(H, W, w_cx, w_cy, w_r_in, w_r_out)
        img[washer] = mat_val * 0.95

    # Thread damage / galling (random local density changes)
    if rng.random() < 0.5:
        n_damage = rng.integers(1, 4)
        for _ in range(n_damage):
            angle = rng.uniform(0, 2 * np.pi)
            dist = shaft_r + rng.uniform(0.0, 0.08)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            dr = rng.uniform(0.005, 0.015)
            damage = _circle_mask(H, W, dx, dy, dr)
            img[damage] = mat_val * rng.uniform(0.3, 0.7)

    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0.0, 1.0)

    return img, f"bolt_{variant:02d}"


def make_weld_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Cross-section of a weld joint with porosity, cracks, and lack-of-fusion.

    Simulates butt weld, fillet weld, or lap joint cross-section
    with typical welding defects.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)

    mat_val = rng.uniform(*MATERIALS["steel"])

    weld_type = variant % 3  # 0: butt, 1: fillet, 2: lap

    if weld_type == 0:
        # Butt weld: two plates joined horizontally
        plate_h = rng.uniform(0.12, 0.18)
        # Left plate
        left = _rect_mask(H, W, -0.25, 0.0, 0.25, plate_h, 0)
        img[left] = mat_val
        # Right plate
        right = _rect_mask(H, W, 0.25, 0.0, 0.25, plate_h, 0)
        img[right] = mat_val
        # Weld bead (V-groove + cap)
        weld_zone = _ellipse_mask(H, W, 0.0, 0.0, 0.08, plate_h * 1.3, 0)
        img[weld_zone] = mat_val * rng.uniform(0.95, 1.05)
        # Root reinforcement
        root = _ellipse_mask(H, W, 0.0, plate_h * 0.8, 0.04, 0.03, 0)
        img[root] = mat_val * 1.02
        # Cap reinforcement
        cap = _ellipse_mask(H, W, 0.0, -plate_h * 0.8, 0.06, 0.04, 0)
        img[cap] = mat_val * 1.01
        defect_zone_cx, defect_zone_cy = 0.0, 0.0
        defect_zone_r = 0.08

    elif weld_type == 1:
        # Fillet weld: T-joint
        # Horizontal plate
        h_plate = _rect_mask(H, W, 0.0, 0.15, 0.40, 0.10, 0)
        img[h_plate] = mat_val
        # Vertical plate
        v_plate = _rect_mask(H, W, 0.0, -0.15, 0.08, 0.25, 0)
        img[v_plate] = mat_val
        # Fillet welds (triangular regions at junction)
        for sign in [-1, 1]:
            fx = sign * 0.06
            fillet = _ellipse_mask(H, W, fx, 0.06, 0.06, 0.06,
                                   sign * 45)
            img[fillet] = mat_val * rng.uniform(0.93, 1.03)
        defect_zone_cx, defect_zone_cy = 0.0, 0.06
        defect_zone_r = 0.08

    else:
        # Lap joint: overlapping plates
        # Upper plate (shifted left)
        upper = _rect_mask(H, W, -0.10, -0.08, 0.30, 0.08, 0)
        img[upper] = mat_val
        # Lower plate (shifted right)
        lower = _rect_mask(H, W, 0.10, 0.08, 0.30, 0.08, 0)
        img[lower] = mat_val
        # Fillet welds at overlap edges
        for cx, cy in [(0.10, -0.04), (-0.08, 0.04)]:
            weld = _ellipse_mask(H, W, cx, cy, 0.05, 0.05, 0)
            img[weld] = mat_val * rng.uniform(0.94, 1.04)
        defect_zone_cx, defect_zone_cy = 0.0, 0.0
        defect_zone_r = 0.12

    # Welding defects: porosity (gas pores)
    n_pores = rng.integers(3, 12)
    for _ in range(n_pores):
        px = defect_zone_cx + rng.uniform(-defect_zone_r, defect_zone_r)
        py = defect_zone_cy + rng.uniform(-defect_zone_r * 0.5, defect_zone_r * 0.5)
        pr = rng.uniform(0.003, 0.012)
        pore = _circle_mask(H, W, px, py, pr)
        img[pore] = rng.uniform(0.0, 0.05)  # gas void

    # Cracks (thin elongated voids)
    n_cracks = rng.integers(0, 3)
    for _ in range(n_cracks):
        cx = defect_zone_cx + rng.uniform(-defect_zone_r * 0.5, defect_zone_r * 0.5)
        cy = defect_zone_cy + rng.uniform(-defect_zone_r * 0.3, defect_zone_r * 0.3)
        crack_len = rng.uniform(0.02, 0.06)
        crack_w = rng.uniform(0.001, 0.004)
        crack_angle = rng.uniform(0, 180)
        crack = _ellipse_mask(H, W, cx, cy, crack_len, crack_w, crack_angle)
        img[crack] = 0.0

    # Lack of fusion (interface defect)
    if rng.random() < 0.3:
        lof_y = defect_zone_cy + rng.uniform(-0.02, 0.02)
        lof = _rect_mask(H, W, defect_zone_cx, lof_y, 0.04, 0.002, rng.uniform(-10, 10))
        img[lof] = rng.uniform(0.0, 0.02)

    # Slag inclusion
    if rng.random() < 0.25:
        sx = defect_zone_cx + rng.uniform(-0.03, 0.03)
        sy = defect_zone_cy + rng.uniform(-0.02, 0.02)
        slag = _ellipse_mask(H, W, sx, sy, rng.uniform(0.005, 0.015),
                             rng.uniform(0.003, 0.008), rng.uniform(0, 180))
        img[slag] = rng.uniform(0.15, 0.30)  # slag is less dense than metal

    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0.0, 1.0)

    return img, f"weld_{['butt', 'fillet', 'lap'][weld_type]}_{variant:02d}"


def make_composite_phantom(
    H: int, W: int, seed: int, variant: int = 0
) -> tuple[np.ndarray, str]:
    """Multi-material composite assembly: metal + plastic + ceramic.

    Simulates assembled part cross-section with different materials,
    fasteners, adhesive layers, and delamination defects.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W), dtype=np.float64)

    # Outer housing (aluminum or steel)
    housing_mat = "aluminum" if variant % 2 == 0 else "steel"
    housing_val = rng.uniform(*MATERIALS[housing_mat])
    housing_r = rng.uniform(0.38, 0.45)
    housing_inner_r = housing_r - rng.uniform(0.03, 0.06)
    housing = _annulus_mask(H, W, 0.0, 0.0, housing_inner_r, housing_r)
    img[housing] = housing_val

    # Internal plastic component
    plastic_val = rng.uniform(*MATERIALS["plastic"])
    plastic = _circle_mask(H, W, 0.0, 0.0, housing_inner_r - 0.01)
    img[plastic] = plastic_val

    # Ceramic insert
    ceramic_val = rng.uniform(*MATERIALS["ceramic"])
    ceramic_cx = rng.uniform(-0.10, 0.10)
    ceramic_cy = rng.uniform(-0.10, 0.10)
    ceramic = _ellipse_mask(H, W, ceramic_cx, ceramic_cy,
                            rng.uniform(0.06, 0.12), rng.uniform(0.05, 0.10),
                            rng.uniform(0, 180))
    img[ceramic & plastic] = ceramic_val

    # Metal core / shaft
    core_mat = "copper" if variant % 3 == 0 else "steel"
    core_val = rng.uniform(*MATERIALS[core_mat])
    core_r = rng.uniform(0.04, 0.08)
    core = _circle_mask(H, W, 0.0, 0.0, core_r)
    img[core] = core_val

    # Mounting bolts (small steel circles near housing)
    n_bolts = rng.integers(3, 7)
    for k in range(n_bolts):
        angle = 2 * np.pi * k / n_bolts + rng.uniform(-0.1, 0.1)
        dist = (housing_r + housing_inner_r) / 2
        bx = dist * np.cos(angle)
        by = dist * np.sin(angle)
        br = rng.uniform(0.012, 0.020)
        bolt = _circle_mask(H, W, bx, by, br)
        img[bolt] = rng.uniform(*MATERIALS["steel"])

    # Delamination (air gap between layers)
    if rng.random() < 0.4:
        delam_angle = rng.uniform(0, 360)
        delam_r = housing_inner_r
        delam = _annulus_mask(H, W, 0.0, 0.0, delam_r - 0.003, delam_r + 0.003)
        # Only a sector
        yy = np.linspace(-1.0, 1.0, H)[:, None]
        xx = np.linspace(-1.0, 1.0, W)[None, :]
        angles = np.degrees(np.arctan2(yy, xx)) % 360
        sector = (angles >= delam_angle) & (angles < delam_angle + rng.uniform(30, 90))
        img[delam & sector] = 0.0

    # Internal voids in plastic
    n_voids = rng.integers(0, 5)
    for _ in range(n_voids):
        vx = rng.uniform(-housing_inner_r * 0.5, housing_inner_r * 0.5)
        vy = rng.uniform(-housing_inner_r * 0.5, housing_inner_r * 0.5)
        vr = rng.uniform(0.004, 0.015)
        void = _circle_mask(H, W, vx, vy, vr)
        if img[void & plastic].size > 0:
            img[void & plastic] = 0.0

    img = gaussian_filter(img, sigma=0.5)
    img = np.clip(img, 0.0, 1.0)

    return img, f"composite_{housing_mat}_{variant:02d}"


# -- Phantom pools per tier ---------------------------------------------------

PHANTOM_GENERATORS = [make_cylinder_phantom, make_bolt_phantom,
                      make_weld_phantom, make_composite_phantom]


def generate_phantoms_public(n: int = 12) -> list[tuple[np.ndarray, str]]:
    """Generate public-tier phantoms: 3 cylinder + 3 bolt + 3 weld + 3 composite."""
    phantoms = []
    per_type = n // 4
    remainder = n % 4
    counts = [per_type] * 4
    for i in range(remainder):
        counts[i] += 1

    idx = 0
    for gen_idx, gen_fn in enumerate(PHANTOM_GENERATORS):
        for i in range(counts[gen_idx]):
            img, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=100 + idx, variant=i)
            phantoms.append((img, name))
            idx += 1
    return phantoms[:n]


def generate_phantoms_dev(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """Generate dev-tier phantoms with augmented diversity."""
    phantoms = []
    rng = np.random.default_rng(15000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 4]
        img, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=10500 + i, variant=i)
        # Augmentation: rotation, flip, zoom
        angle = float(rng.uniform(15, 345))
        img = nd_rotate(img, angle, reshape=False, mode="constant", cval=0.0)
        if rng.random() < 0.5:
            img = np.fliplr(img)
        if rng.random() < 0.3:
            img = np.flipud(img)
        zoom_f = float(rng.uniform(0.85, 1.15))
        if zoom_f != 1.0:
            img = _zoom_crop(img, zoom_f, IMAGE_SIZE)
        img = np.clip(img, 0.0, 1.0)
        phantoms.append((img, f"dev_{name}"))
    return phantoms


def generate_phantoms_hidden(n: int = 20) -> list[tuple[np.ndarray, str]]:
    """Generate hidden-tier phantoms with adversarial modifications."""
    phantoms = []
    rng = np.random.default_rng(25000)
    for i in range(n):
        gen_fn = PHANTOM_GENERATORS[i % 4]
        img, name = gen_fn(IMAGE_SIZE, IMAGE_SIZE, seed=20500 + i, variant=i + 10)

        # Aggressive augmentation
        angle = float(rng.uniform(20, 340))
        img = nd_rotate(img, angle, reshape=False, mode="constant", cval=0.0)
        if rng.random() < 0.7:
            img = np.fliplr(img)
        if rng.random() < 0.5:
            img = np.flipud(img)

        zoom_f = float(rng.uniform(0.70, 1.30))
        img = _zoom_crop(img, zoom_f, IMAGE_SIZE)

        # Adversarial: add micro-cracks and extra dense inclusions
        n_micro = rng.integers(3, 8)
        for _ in range(n_micro):
            cy_px = rng.integers(30, IMAGE_SIZE - 30)
            cx_px = rng.integers(30, IMAGE_SIZE - 30)
            if img[cy_px, cx_px] > 0.1:  # only in material regions
                if rng.random() < 0.5:
                    # Micro-void
                    r = rng.integers(1, 4)
                    yy, xx = np.ogrid[-r:r+1, -r:r+1]
                    circle = (yy**2 + xx**2 <= r**2)
                    y0 = max(0, cy_px - r)
                    y1 = min(IMAGE_SIZE, cy_px + r + 1)
                    x0 = max(0, cx_px - r)
                    x1 = min(IMAGE_SIZE, cx_px + r + 1)
                    cy0 = r - (cy_px - y0)
                    cy1 = r + (y1 - cy_px)
                    cx0 = r - (cx_px - x0)
                    cx1 = r + (x1 - cx_px)
                    patch = circle[cy0:cy1, cx0:cx1]
                    img[y0:y1, x0:x1] = np.where(
                        patch, 0.0, img[y0:y1, x0:x1])
                else:
                    # Dense inclusion (tungsten particle)
                    r = rng.integers(1, 3)
                    yy, xx = np.ogrid[-r:r+1, -r:r+1]
                    circle = (yy**2 + xx**2 <= r**2)
                    y0 = max(0, cy_px - r)
                    y1 = min(IMAGE_SIZE, cy_px + r + 1)
                    x0 = max(0, cx_px - r)
                    x1 = min(IMAGE_SIZE, cx_px + r + 1)
                    cy0 = r - (cy_px - y0)
                    cy1 = r + (y1 - cy_px)
                    cx0 = r - (cx_px - x0)
                    cx1 = r + (x1 - cx_px)
                    patch = circle[cy0:cy1, cx0:cx1]
                    img[y0:y1, x0:x1] = np.where(
                        patch, rng.uniform(0.90, 1.0),
                        img[y0:y1, x0:x1])

        img = np.clip(img, 0.0, 1.0)
        phantoms.append((img, f"hidden_{name}"))
    return phantoms


def _zoom_crop(arr: np.ndarray, zoom_f: float, size: int) -> np.ndarray:
    """Zoom and crop/pad to target size."""
    zoomed = nd_zoom(arr, zoom_f, order=1)
    H, W = zoomed.shape
    if H >= size and W >= size:
        y0 = (H - size) // 2
        x0 = (W - size) // 2
        return zoomed[y0:y0 + size, x0:x0 + size]
    else:
        out = np.zeros((size, size), dtype=arr.dtype)
        y0 = (size - H) // 2
        x0 = (size - W) // 2
        out[y0:y0 + H, x0:x0 + W] = zoomed
        return out


# -- Industrial CT Forward Model ----------------------------------------------

def industrial_ct_forward(
    x_true: np.ndarray,
    theta_deg: np.ndarray,
    beam_hardening_order: float,
    scatter_fraction: float,
    source_blur: float,
    detector_efficiency: float,
    rng: np.random.Generator,
    I0: float = 50_000.0,  # high dose (industrial CT)
    sigma_readout: float = 3.0,
) -> dict:
    """Apply industrial CT forward model with beam hardening + scatter + noise.

    Forward model:
        1. Radon transform (parallel-beam projection)
        2. Beam hardening: p_eff = p + bh * p^2 + bh2 * p^3
        3. Source blur (focal spot size)
        4. Scatter (low-frequency background)
        5. Beer-Lambert + Poisson noise + readout noise
        6. Detector efficiency variation

    Args:
        x_true:                material attenuation map [0, 1]
        theta_deg:             projection angles in degrees
        beam_hardening_order:  beam hardening polynomial coefficient
        scatter_fraction:      scatter / direct signal ratio
        source_blur:           focal spot blur sigma in detector pixels
        detector_efficiency:   relative detector gain
        rng:                   random generator
        I0:                    incident photon count (high for industrial)
        sigma_readout:         readout noise sigma

    Returns:
        dict with sinogram_ideal, sinogram_measured, etc.
    """
    # 1. Ideal sinogram (Radon transform)
    sino_ideal = radon_transform(x_true, theta_deg)
    sino_ideal = np.maximum(sino_ideal, 0.0)
    n_det = sino_ideal.shape[1]

    # Scale sinogram to physical attenuation range
    # Industrial CT: higher attenuation than medical
    mu_scale = 0.10  # pixel-density to nepers scaling
    sino_phys = sino_ideal * mu_scale

    # 2. Beam hardening: p_eff = p + bh * p^2 + (bh/2) * p^3
    # This models the polychromatic X-ray source effect where
    # higher-Z materials cause more severe hardening
    bh = beam_hardening_order
    sino_bh = sino_phys + bh * sino_phys**2 + (bh * 0.5) * sino_phys**3

    # 3. Source blur (focal spot broadening)
    if source_blur > 0.1:
        for i in range(sino_bh.shape[0]):
            sino_bh[i] = gaussian_filter(sino_bh[i], sigma=source_blur)

    # 4. Scatter contribution
    # Scatter is a smooth, low-frequency background proportional to total signal
    if scatter_fraction > 0.001:
        scatter_base = gaussian_filter(sino_bh, sigma=[3.0, 15.0])
        scatter = scatter_fraction * scatter_base
        scatter += rng.standard_normal(sino_bh.shape) * scatter.mean() * 0.05
        scatter = np.maximum(scatter, 0.0)
    else:
        scatter = np.zeros_like(sino_bh)

    # 5. Beer-Lambert + Poisson noise
    sino_total = sino_bh + scatter
    sino_clamped = np.clip(sino_total, 0.0, 30.0)
    I_expected = I0 * detector_efficiency * np.exp(-sino_clamped)
    I_noisy = rng.poisson(np.maximum(I_expected, 1e-3)).astype(np.float64)
    I_noisy += rng.normal(0.0, sigma_readout, I_noisy.shape)
    I_noisy = np.maximum(I_noisy, 1.0)

    # Convert back to line integrals (nepers)
    sino_measured = -np.log(I_noisy / (I0 * detector_efficiency))

    return {
        "sinogram_ideal": sino_phys.astype(np.float32),
        "sinogram_measured": sino_measured.astype(np.float32),
        "scatter": scatter.astype(np.float32),
        "beam_hardening_sino": sino_bh.astype(np.float32),
        "I0": float(I0),
        "mu_scale": float(mu_scale),
    }


# -- Metrics ------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-12:
        return 0.0
    return float(10 * np.log10(data_range ** 2 / mse))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    gt = gt.astype(np.float64)
    recon = recon.astype(np.float64)
    data_range = gt.max() - gt.min()
    if data_range < 1e-12:
        return 0.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = gt.mean()
    mu_y = recon.mean()
    var_x = gt.var()
    var_y = recon.var()
    cov_xy = np.mean((gt - mu_x) * (recon - mu_y))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim)


# -- Image helpers ------------------------------------------------------------

def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(
        np.clip(_norm(arr) * 255, 0, 255).astype(np.uint8), "L"
    ).save(str(path))


def _save_overview(x_true, sino_ideal, sino_meas, recon_fbp, path: Path) -> None:
    """4-panel overview: GT | ideal sino | measured sino | FBP recon."""
    th, tw = 128, 128

    def _r(a):
        pil = Image.fromarray(np.clip(_norm(a) * 255, 0, 255).astype(np.uint8), "L")
        return np.array(pil.resize((tw, th), Image.LANCZOS)) / 255.0

    ov = np.zeros((th, 4 * tw), dtype=np.float32)
    ov[:, 0:tw] = _r(x_true)
    ov[:, tw:2*tw] = _r(sino_ideal)
    ov[:, 2*tw:3*tw] = _r(sino_meas)
    ov[:, 3*tw:4*tw] = _r(recon_fbp)
    _save_png(ov, path)


# -- Tier generation ----------------------------------------------------------

def sample_mismatch(rng: np.random.Generator, spec: dict) -> dict:
    return {k: float(rng.uniform(v["min"], v["max"])) for k, v in spec.items()}


def generate_tier(
    tier: str,
    phantoms: list[tuple[np.ndarray, str]],
    base_seed: int,
) -> None:
    """Generate one tier of the industrial CT benchmark."""
    spec_ranges = SPEC[tier]
    tier_dir = BENCHMARK_DIR / tier
    images_dir = tier_dir / "images"
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    theta_deg = np.linspace(0, 180, N_ANGLES, endpoint=False).astype(np.float64)

    h5_path = tier_dir / f"industrial_ct_challenge_{tier}.h5"
    rng = np.random.default_rng(base_seed)
    true_specs = {}
    all_psnrs = []
    all_ssims = []

    with h5py.File(h5_path, "w") as f:
        f.attrs["description"] = (
            f"PWM Industrial CT benchmark -- {tier} tier "
            f"(parallel-beam Radon + beam hardening + scatter + Poisson noise)"
        )
        f.attrs["spec_ranges"] = json.dumps(spec_ranges)
        f.attrs["geometry"] = json.dumps({
            "image_size": IMAGE_SIZE,
            "n_angles": N_ANGLES,
            "n_det": N_DET,
            "angle_range_deg": [0, 180],
        })
        f.attrs["forward_model"] = (
            "y = -log(Poisson(I0 * eta * exp(-(Radon(x)*mu + bh*p^2 + bh2*p^3 + scatter))) / (I0*eta))"
        )

        for idx, (x_true, scene_name) in enumerate(phantoms):
            key = f"sample_{idx:02d}"
            print(f"  [{tier}] Generating {key} ({scene_name})...", end="", flush=True)

            mis = sample_mismatch(rng, spec_ranges)
            true_specs[key] = mis

            # Apply forward model
            result = industrial_ct_forward(
                x_true, theta_deg,
                beam_hardening_order=mis["beam_hardening_order"],
                scatter_fraction=mis["scatter_fraction"],
                source_blur=mis["source_blur"],
                detector_efficiency=mis["detector_efficiency"],
                rng=rng,
            )

            sino_ideal = result["sinogram_ideal"]
            sino_measured = result["sinogram_measured"]
            n_det = sino_ideal.shape[1]

            # FBP reconstruction from measured sinogram
            # Rescale measured sinogram from nepers back to projection domain
            mu_scale = result["mu_scale"]
            sino_for_fbp = sino_measured / mu_scale if mu_scale > 0 else sino_measured
            recon_fbp = fbp_reconstruct(sino_for_fbp, theta_deg, IMAGE_SIZE)
            recon_fbp = np.maximum(recon_fbp, 0.0).astype(np.float32)

            psnr = compute_psnr(x_true, recon_fbp)
            ssim = compute_ssim(x_true, recon_fbp)
            all_psnrs.append(psnr)
            all_ssims.append(ssim)

            # Save to HDF5
            grp = f.create_group(key)
            grp.create_dataset("x_true", data=x_true.astype(np.float32),
                               compression="gzip")
            grp.create_dataset("y", data=sino_measured, compression="gzip")
            grp.create_dataset("sinogram_ideal", data=sino_ideal,
                               compression="gzip")
            grp.create_dataset("sinogram_measured", data=sino_measured,
                               compression="gzip")
            grp.create_dataset("angles_deg", data=theta_deg.astype(np.float32))
            grp.create_dataset("reconstruction_fbp", data=recon_fbp,
                               compression="gzip")
            grp.attrs["metadata"] = json.dumps({
                "scene": scene_name,
                "shape": list(x_true.shape),
                "n_angles": N_ANGLES,
                "n_det": int(n_det),
                "psnr_fbp": float(psnr),
                "ssim_fbp": float(ssim),
            })
            grp.attrs["true_spec"] = json.dumps(mis)
            grp.attrs["spec_ranges"] = json.dumps(spec_ranges)

            # Save images
            sample_dir = images_dir / f"sample_{idx:02d}_{scene_name}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _save_png(x_true, sample_dir / "ground_truth.png")
            _save_png(sino_ideal, sample_dir / "sinogram_ideal.png")
            _save_png(sino_measured, sample_dir / "sinogram_measured.png")
            _save_png(recon_fbp, sample_dir / "reconstruction_fbp.png")
            _save_overview(x_true, sino_ideal, sino_measured, recon_fbp,
                           sample_dir / "overview.png")
            with open(sample_dir / "spec.json", "w") as sf:
                json.dump({"scene": scene_name, "spec_ranges": spec_ranges,
                           "true_spec": mis, "psnr_fbp": psnr, "ssim_fbp": ssim},
                          sf, indent=2)

            print(f"  PSNR={psnr:.2f} dB, SSIM={ssim:.3f}  "
                  f"bh={mis['beam_hardening_order']:.3f}  "
                  f"scatter={mis['scatter_fraction']:.3f}  "
                  f"blur={mis['source_blur']:.2f}  "
                  f"eta={mis['detector_efficiency']:.3f}")

    # Save spec files
    with open(tier_dir / "spec.json", "w") as sf:
        json.dump(spec_ranges, sf, indent=2)
    with open(tier_dir / "true_spec.json", "w") as tf:
        json.dump(true_specs, tf, indent=2)

    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    print(f"  [{tier}] {len(phantoms)} samples | "
          f"Mean PSNR={mean_psnr:.2f} dB | Mean SSIM={mean_ssim:.3f}")
    print(f"  [{tier}] HDF5 -> {h5_path.name}")


# -- Gallery images -----------------------------------------------------------

def generate_gallery_images() -> None:
    """Generate gallery images for the platform benchmark page."""
    gallery_base = (BENCHMARK_DIR.parent.parent.parent /
                    "platform" / "pwm_platform" / "static" / "img" /
                    "benchmark_gallery" / "industrial_ct")

    h5_path = BENCHMARK_DIR / "public" / "industrial_ct_challenge_public.h5"
    if not h5_path.exists():
        print("  [gallery] Public HDF5 not found, skipping gallery generation.")
        return

    gallery_sample_indices = [0, 3, 6, 9]

    with h5py.File(h5_path, "r") as f:
        for scene_idx, sample_idx in enumerate(gallery_sample_indices):
            scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)

            key = f"sample_{sample_idx:02d}"
            if key not in f:
                print(f"  [gallery] {key} not found in HDF5, skipping.")
                continue

            grp = f[key]
            x_true = grp["x_true"][:]
            sino_ideal = grp["sinogram_ideal"][:]
            sino_meas = grp["sinogram_measured"][:]
            recon_fbp = grp["reconstruction_fbp"][:]

            _save_png(x_true, scene_dir / "gt.png")
            _save_png(sino_meas, scene_dir / "measurement_I.png")
            _save_png(sino_ideal, scene_dir / "measurement_II.png")
            _save_png(recon_fbp, scene_dir / "recon_I.png")

            # Beam hardening artifact visualization (difference)
            diff = np.abs(sino_meas - sino_ideal)
            _save_png(diff, scene_dir / "recon_II.png")

            # Error map
            err = np.abs(x_true - recon_fbp)
            _save_png(err, scene_dir / "recon_III.png")

            print(f"  [gallery] scene_{scene_idx:02d} images saved to {scene_dir}")


# -- Main ---------------------------------------------------------------------

def main() -> None:
    print("Industrial CT Benchmark Dataset Generator")
    print("=" * 68)
    print(f"Output: {BENCHMARK_DIR}")
    print(f"Geometry: {N_ANGLES} angles, {IMAGE_SIZE}x{IMAGE_SIZE} images\n")

    # -- Public tier (12 samples) -----------------------------------------
    print("Generating public tier (12 samples)...")
    public_phantoms = generate_phantoms_public(12)
    generate_tier("public", public_phantoms, base_seed=0)

    # -- Dev tier (20 samples) -------------------------------------------
    print("\nGenerating dev tier (20 samples)...")
    dev_phantoms = generate_phantoms_dev(20)
    generate_tier("dev", dev_phantoms, base_seed=10000)

    # -- Hidden tier (20 samples) ----------------------------------------
    print("\nGenerating hidden tier (20 samples)...")
    hidden_phantoms = generate_phantoms_hidden(20)
    generate_tier("hidden", hidden_phantoms, base_seed=20000)

    # -- Gallery images --------------------------------------------------
    print("\nGenerating gallery images...")
    generate_gallery_images()

    print(f"\n{'=' * 68}")
    print("Industrial CT benchmark dataset generation complete!")
    print(f"Output: {BENCHMARK_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
