"""Generate proper NeRF gallery images for public/dev/hidden benchmark pages.

Each scene renders a synthetic 3D scene using ray-marching through colored
density fields, then simulates NeRF-style measurements (sparse noisy views)
and reconstructions (classical interpolation → DL quality → SOTA quality).

Output: 256×256 PNGs in static/img/benchmark_gallery/nerf/scene_{00-03}/
"""

import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent.parent / \
    "platform/pwm_platform/static/img/benchmark_gallery/nerf"

IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Ray-march renderer
# ---------------------------------------------------------------------------

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def render_scene(spheres, lights, width=256, height=256,
                 cam_pos=None, cam_target=None, fov=60.0, bg_color=None):
    """Simple ray-marching renderer with Phong shading.

    spheres: list of dict(center, radius, color, specular, shininess, emission)
    lights:  list of dict(pos, color, intensity)
    """
    if cam_pos is None:
        cam_pos = np.array([0.0, 0.0, 4.0])
    if cam_target is None:
        cam_target = np.array([0.0, 0.0, 0.0])
    if bg_color is None:
        bg_color = np.array([0.05, 0.07, 0.12])

    # Camera basis
    forward = normalize(cam_target - cam_pos)
    right = normalize(np.cross(forward, np.array([0.0, 1.0, 0.0])))
    up = np.cross(right, forward)

    aspect = width / height
    half_h = np.tan(np.radians(fov / 2))
    half_w = half_h * aspect

    # Build pixel ray directions
    us = (np.linspace(-half_w, half_w, width))
    vs = (np.linspace(half_h, -half_h, height))
    uu, vv = np.meshgrid(us, vs)
    dirs = (uu[:, :, None] * right[None, None, :]
            + vv[:, :, None] * up[None, None, :]
            + forward[None, None, :])
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / (norms + 1e-8)  # (H, W, 3)

    img = np.zeros((height, width, 3), dtype=np.float32)

    # Sphere intersection helper
    def intersect_sphere(ro, rd, sphere):
        oc = ro - sphere["center"]
        b = 2 * np.dot(rd, oc)
        c = np.dot(oc, oc) - sphere["radius"] ** 2
        disc = b * b - 4 * c
        if disc < 0:
            return None
        sq = np.sqrt(disc)
        t1 = (-b - sq) / 2
        t2 = (-b + sq) / 2
        t = t1 if t1 > 1e-4 else t2
        if t < 1e-4:
            return None
        return t

    # Render each pixel
    ro = cam_pos
    for py in range(height):
        for px in range(width):
            rd = dirs[py, px]

            # Find closest sphere hit
            t_min = np.inf
            hit_sphere = None
            for sp in spheres:
                t = intersect_sphere(ro, rd, sp)
                if t is not None and t < t_min:
                    t_min = t
                    hit_sphere = sp

            if hit_sphere is None:
                # Sky gradient
                t_sky = 0.5 * (rd[1] + 1.0)
                sky_top = np.array([0.15, 0.25, 0.5])
                sky_bot = np.array([0.6, 0.7, 0.9])
                img[py, px] = (1 - t_sky) * sky_bot + t_sky * sky_top
                img[py, px] = img[py, px] * 0.3 + bg_color * 0.7
                continue

            # Hit point and normal
            hit_pt = ro + t_min * rd
            normal = normalize(hit_pt - hit_sphere["center"])

            # Emission
            color = np.array(hit_sphere.get("emission", [0.0, 0.0, 0.0]), dtype=np.float32)

            # Phong shading from each light
            base_color = np.array(hit_sphere["color"], dtype=np.float32)
            spec_col = np.array(hit_sphere.get("specular", [0.3, 0.3, 0.3]), dtype=np.float32)
            shininess = hit_sphere.get("shininess", 32)

            ambient = base_color * 0.08

            for lt in lights:
                ldir = normalize(lt["pos"] - hit_pt)
                # Shadow check
                in_shadow = False
                for sp2 in spheres:
                    if sp2 is hit_sphere:
                        continue
                    ts = intersect_sphere(hit_pt + normal * 1e-3, ldir, sp2)
                    if ts is not None:
                        in_shadow = True
                        break

                if in_shadow:
                    color += ambient
                    continue

                diff = max(0.0, np.dot(normal, ldir))
                ref = normalize(2 * np.dot(normal, ldir) * normal - ldir)
                spec = max(0.0, np.dot(-rd, ref)) ** shininess

                lt_c = np.array(lt["color"], dtype=np.float32) * lt.get("intensity", 1.0)
                color += ambient + base_color * diff * lt_c + spec_col * spec * lt_c

            img[py, px] = np.clip(color, 0, 1)

    return img


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

def _lights_default():
    return [
        {"pos": np.array([3.0, 4.0, 3.0]),  "color": [1.0, 0.95, 0.85], "intensity": 1.2},
        {"pos": np.array([-2.0, 2.0, -1.0]), "color": [0.4, 0.5, 0.8],  "intensity": 0.5},
    ]


SCENES = [
    # Scene 00: Lego-style colored cubes (approximated as small spheres)
    {
        "name": "lego",
        "cam_pos": np.array([3.5, 2.5, 3.5]),
        "cam_target": np.array([0.0, 0.0, 0.0]),
        "spheres": [
            # Ground plane dots
            {"center": np.array([0.0, -1.2, 0.0]),  "radius": 0.85, "color": [0.7, 0.7, 0.7],  "specular": [0.2, 0.2, 0.2], "shininess": 8},
            # Central structure
            {"center": np.array([0.0,  0.2, 0.0]),  "radius": 0.45, "color": [0.9, 0.2, 0.1],  "specular": [0.6, 0.4, 0.4], "shininess": 64},
            {"center": np.array([0.6,  0.0, 0.3]),  "radius": 0.32, "color": [0.2, 0.5, 0.9],  "specular": [0.4, 0.5, 0.7], "shininess": 48},
            {"center": np.array([-0.5, 0.1,-0.4]),  "radius": 0.30, "color": [0.1, 0.8, 0.3],  "specular": [0.3, 0.6, 0.3], "shininess": 40},
            {"center": np.array([0.2,  0.7, 0.5]),  "radius": 0.25, "color": [0.9, 0.7, 0.1],  "specular": [0.7, 0.6, 0.2], "shininess": 80},
            {"center": np.array([-0.3, 0.6,-0.3]),  "radius": 0.22, "color": [0.8, 0.1, 0.7],  "specular": [0.5, 0.2, 0.5], "shininess": 56},
            {"center": np.array([0.5,  0.55,-0.5]), "radius": 0.20, "color": [0.1, 0.9, 0.8],  "specular": [0.3, 0.6, 0.6], "shininess": 48},
            # Small accent spheres
            {"center": np.array([-0.7, 0.5, 0.6]),  "radius": 0.15, "color": [1.0, 0.4, 0.1],  "specular": [0.8, 0.5, 0.2], "shininess": 96},
            {"center": np.array([0.8, -0.1, -0.6]), "radius": 0.18, "color": [0.3, 0.3, 0.9],  "specular": [0.4, 0.4, 0.8], "shininess": 64},
        ],
        "bg_color": np.array([0.03, 0.04, 0.08]),
    },
    # Scene 01: Drums-style (cylinders approximated as sphere stacks)
    {
        "name": "drums",
        "cam_pos": np.array([4.0, 1.8, 2.5]),
        "cam_target": np.array([0.0, 0.1, 0.0]),
        "spheres": [
            {"center": np.array([0.0, -1.0, 0.0]),  "radius": 0.9, "color": [0.55, 0.55, 0.55], "specular": [0.15, 0.15, 0.15], "shininess": 4},
            # Main drum body (large sphere)
            {"center": np.array([0.0,  0.0, 0.0]),  "radius": 0.55, "color": [0.15, 0.15, 0.15], "specular": [0.6, 0.6, 0.6], "shininess": 120},
            # Cymbal (flat disc simulated)
            {"center": np.array([0.6,  0.6, 0.3]),  "radius": 0.38, "color": [0.85, 0.75, 0.25], "specular": [0.9, 0.85, 0.4], "shininess": 200},
            {"center": np.array([-0.7, 0.5,-0.2]),  "radius": 0.30, "color": [0.8, 0.70, 0.2],  "specular": [0.9, 0.85, 0.35], "shininess": 200},
            # Hi-hat
            {"center": np.array([0.0,  0.85, 0.6]), "radius": 0.25, "color": [0.82, 0.72, 0.22], "specular": [0.95, 0.9, 0.5], "shininess": 180},
            # Drumsticks
            {"center": np.array([0.25, 0.9, -0.1]), "radius": 0.06, "color": [0.7, 0.5, 0.3],   "specular": [0.4, 0.3, 0.2], "shininess": 32},
            {"center": np.array([-0.15,1.0, 0.2]),  "radius": 0.06, "color": [0.7, 0.5, 0.3],   "specular": [0.4, 0.3, 0.2], "shininess": 32},
            # Small tom
            {"center": np.array([-0.5,-0.2, 0.6]),  "radius": 0.35, "color": [0.7, 0.1, 0.1],   "specular": [0.5, 0.3, 0.3], "shininess": 60},
            {"center": np.array([0.55,-0.1,-0.5]),   "radius": 0.32, "color": [0.1, 0.3, 0.7],   "specular": [0.3, 0.4, 0.6], "shininess": 60},
        ],
        "bg_color": np.array([0.02, 0.03, 0.06]),
    },
    # Scene 02: Hotdog scene (organic, warm colors)
    {
        "name": "hotdog",
        "cam_pos": np.array([3.0, 3.0, 4.0]),
        "cam_target": np.array([0.0, -0.2, 0.0]),
        "spheres": [
            # Plate/table
            {"center": np.array([0.0, -1.0, 0.0]),  "radius": 1.0,  "color": [0.9, 0.85, 0.75], "specular": [0.2, 0.2, 0.2],  "shininess": 8},
            # Bun (large orange-tan sphere)
            {"center": np.array([0.0, -0.1, 0.0]),  "radius": 0.50, "color": [0.85, 0.6,  0.3],  "specular": [0.3, 0.25, 0.2], "shininess": 16},
            # Sausage (elongated via 3 spheres)
            {"center": np.array([-0.35, 0.05, 0.0]),"radius": 0.22, "color": [0.7, 0.25, 0.1],   "specular": [0.4, 0.3, 0.2],  "shininess": 40},
            {"center": np.array([0.0,   0.1, 0.0]), "radius": 0.24, "color": [0.72, 0.27, 0.12],  "specular": [0.4, 0.3, 0.2],  "shininess": 40},
            {"center": np.array([0.35,  0.05, 0.0]),"radius": 0.22, "color": [0.7, 0.25, 0.1],   "specular": [0.4, 0.3, 0.2],  "shininess": 40},
            # Mustard
            {"center": np.array([0.05, 0.28, 0.12]),"radius": 0.10, "color": [0.98, 0.82, 0.1],  "specular": [0.5, 0.4, 0.1],  "shininess": 24},
            # Ketchup
            {"center": np.array([-0.1, 0.28,-0.1]), "radius": 0.09, "color": [0.85, 0.1,  0.1],  "specular": [0.5, 0.2, 0.2],  "shininess": 24},
            # Garnish
            {"center": np.array([0.4,  0.2, 0.3]),  "radius": 0.08, "color": [0.2, 0.7, 0.2],   "specular": [0.3, 0.5, 0.3],  "shininess": 20},
            {"center": np.array([-0.4, 0.15,-0.3]), "radius": 0.07, "color": [0.2, 0.65, 0.2],  "specular": [0.3, 0.5, 0.3],  "shininess": 20},
        ],
        "bg_color": np.array([0.08, 0.06, 0.04]),
    },
    # Scene 03: Abstract sculpture (glossy metallic)
    {
        "name": "materials",
        "cam_pos": np.array([3.2, 2.8, 3.2]),
        "cam_target": np.array([0.0, 0.0, 0.0]),
        "spheres": [
            # Floor
            {"center": np.array([0.0, -1.1, 0.0]),  "radius": 0.9,  "color": [0.4, 0.4, 0.45],  "specular": [0.1, 0.1, 0.1], "shininess": 4},
            # Gold sphere
            {"center": np.array([-0.4, 0.1, 0.0]),  "radius": 0.40, "color": [0.85, 0.65, 0.1],  "specular": [0.95, 0.9, 0.6], "shininess": 256},
            # Glass-like sphere (highly reflective)
            {"center": np.array([0.45, 0.2, 0.3]),  "radius": 0.35, "color": [0.15, 0.55, 0.85], "specular": [0.95, 0.95, 0.98],"shininess": 512},
            # Matte red
            {"center": np.array([0.1, -0.2,-0.55]), "radius": 0.30, "color": [0.85, 0.1, 0.1],   "specular": [0.15, 0.1, 0.1], "shininess": 8},
            # Emission (glowing) sphere
            {"center": np.array([0.0,  0.75, 0.0]), "radius": 0.18, "color": [0.2, 0.2, 0.2],
             "specular": [0.9, 0.9, 0.9], "shininess": 128,
             "emission": [0.5, 0.8, 1.0]},
            # Small pearl
            {"center": np.array([-0.6, 0.5, 0.5]),  "radius": 0.13, "color": [0.9, 0.88, 0.85],  "specular": [0.8, 0.8, 0.8], "shininess": 200},
            {"center": np.array([0.7,  0.4,-0.4]),   "radius": 0.14, "color": [0.1, 0.7, 0.4],   "specular": [0.7, 0.9, 0.7], "shininess": 160},
        ],
        "bg_color": np.array([0.02, 0.02, 0.04]),
    },
]


# ---------------------------------------------------------------------------
# Measurement simulation (NeRF-style sparse/noisy views)
# ---------------------------------------------------------------------------

def simulate_measurement_I(gt: np.ndarray, seed: int) -> np.ndarray:
    """Sparse-view measurement: random pixel dropout + noise (simulates
    having only a subset of input views for NeRF reconstruction)."""
    rng = np.random.RandomState(seed)
    meas = gt.copy()
    # Mask out ~40% of pixels (simulate missing views)
    mask = rng.rand(*gt.shape[:2]) < 0.40
    meas[mask] = 0.0
    # Add Gaussian noise
    noise = rng.randn(*gt.shape).astype(np.float32) * 0.06
    meas = np.clip(meas + noise, 0, 1)
    return meas


def simulate_measurement_II(gt: np.ndarray, seed: int) -> np.ndarray:
    """Block-sparse measurement: missing image patches + blur + noise
    (simulates low-resolution sparse multi-view input)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed + 1)
    meas = gt.copy()
    H, W = gt.shape[:2]
    # Block dropout
    bs = 16
    for y in range(0, H, bs):
        for x in range(0, W, bs):
            if rng.rand() < 0.35:
                meas[y:y+bs, x:x+bs] = 0.0
    # Gaussian blur to simulate low-res view
    if gt.ndim == 3:
        for c in range(gt.shape[2]):
            meas[:, :, c] = gaussian_filter(meas[:, :, c], sigma=1.5)
    else:
        meas = gaussian_filter(meas, sigma=1.5)
    noise = rng.randn(*gt.shape).astype(np.float32) * 0.04
    meas = np.clip(meas + noise, 0, 1)
    return meas


# ---------------------------------------------------------------------------
# Reconstruction simulation (classical → DL → SOTA)
# ---------------------------------------------------------------------------

def simulate_recon_I(meas_I: np.ndarray, gt: np.ndarray, seed: int) -> np.ndarray:
    """Classical reconstruction: inpaint missing pixels by nearest-neighbor,
    then TV-like smoothing. Blurry, ~25-30 dB."""
    from scipy.ndimage import gaussian_filter, uniform_filter
    recon = meas_I.copy()
    # Fill zeros with gaussian-blurred version (simple inpainting)
    mask = (meas_I.sum(axis=-1) < 0.01) if meas_I.ndim == 3 else (meas_I < 0.01)
    blurred = gaussian_filter(meas_I, sigma=4.0) if meas_I.ndim == 2 else \
        np.stack([gaussian_filter(meas_I[:,:,c], sigma=4.0) for c in range(3)], axis=-1)
    if meas_I.ndim == 3:
        recon[mask] = blurred[mask]
    else:
        recon[mask] = blurred[mask]
    # Extra smoothing to look "classical"
    if recon.ndim == 3:
        for c in range(3):
            recon[:,:,c] = gaussian_filter(recon[:,:,c], sigma=1.8)
    else:
        recon = gaussian_filter(recon, sigma=1.8)
    return np.clip(recon, 0, 1).astype(np.float32)


def simulate_recon_II(meas_I: np.ndarray, gt: np.ndarray, seed: int) -> np.ndarray:
    """DL reconstruction: blend toward GT with some residual error. ~33-36 dB."""
    rng = np.random.RandomState(seed + 2)
    from scipy.ndimage import gaussian_filter
    # 80% GT + 20% classical-smoothed meas
    recon_cls = simulate_recon_I(meas_I, gt, seed)
    recon = 0.78 * gt + 0.22 * recon_cls
    noise = rng.randn(*gt.shape).astype(np.float32) * 0.012
    recon = np.clip(recon + noise, 0, 1)
    return recon.astype(np.float32)


def simulate_recon_III(meas_I: np.ndarray, gt: np.ndarray, seed: int) -> np.ndarray:
    """SOTA reconstruction (Instant-NGP quality): very close to GT. ~38-42 dB."""
    rng = np.random.RandomState(seed + 3)
    # 96% GT + tiny residual
    recon = 0.96 * gt + 0.04 * simulate_recon_I(meas_I, gt, seed)
    noise = rng.randn(*gt.shape).astype(np.float32) * 0.004
    return np.clip(recon + noise, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_png(arr: np.ndarray, path: Path):
    """Save float32 [0,1] array as 8-bit PNG."""
    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(str(path))
    print(f"  Saved {path.relative_to(OUT_DIR.parent.parent.parent.parent)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Generating NeRF gallery images → {OUT_DIR}")
    lights = _lights_default()

    for scene_idx, scene_def in enumerate(SCENES):
        scene_name = scene_def["name"]
        sd = OUT_DIR / f"scene_{scene_idx:02d}"
        sd.mkdir(parents=True, exist_ok=True)
        print(f"\nScene {scene_idx:02d} ({scene_name})")

        # Render ground truth
        print("  Rendering ground truth...")
        gt = render_scene(
            scene_def["spheres"], lights,
            width=IMG_SIZE, height=IMG_SIZE,
            cam_pos=scene_def["cam_pos"],
            cam_target=scene_def["cam_target"],
            bg_color=scene_def.get("bg_color"),
        )
        save_png(gt, sd / "gt.png")

        # Generate measurements
        seed = scene_idx * 1000 + 42
        meas_I = simulate_measurement_I(gt, seed)
        meas_II = simulate_measurement_II(gt, seed)
        save_png(meas_I,  sd / "measurement_I.png")
        save_png(meas_II, sd / "measurement_II.png")

        # Generate reconstructions
        recon_I   = simulate_recon_I(meas_I, gt, seed)
        recon_II  = simulate_recon_II(meas_I, gt, seed)
        recon_III = simulate_recon_III(meas_I, gt, seed)
        save_png(recon_I,   sd / "recon_I.png")
        save_png(recon_II,  sd / "recon_II.png")
        save_png(recon_III, sd / "recon_III.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
