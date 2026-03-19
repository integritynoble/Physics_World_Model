"""
Procedural high-FPS video generator for CACTI / SCI-video benchmarks.

Generates ground-truth video clips x_{1:T} ∈ [0,1]^{512×512} procedurally,
with NO external datasets required. Each clip is fully determined by a seed
+ recipe_id, so PWM can regenerate any sample deterministically.

Recipe families (scene types):
  0  urban      — rectangle blobs + edges + grid patterns
  1  nature     — smooth textures + large soft objects
  2  textile    — near-periodic textures (hard for reconstruction)
  3  particles  — many tiny moving dots (hard for T=32)
  4  thin_struct — lines, wires, strokes (stress test)
  5  occlusion  — layered objects crossing paths
  6  cam_shake  — mild objects + strong global camera motion

Dev uses mostly: urban, nature, occlusion (easy-medium)
Hidden adds more: textile, particles, thin_struct, cam_shake (hard)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, affine_transform


# ── Public API ───────────────────────────────────────────────────────────────

RECIPE_NAMES = [
    "urban", "nature", "textile", "particles",
    "thin_struct", "occlusion", "cam_shake",
]

# Dev recipes: mostly easy-medium (20 samples)
DEV_RECIPES = [0, 1, 5, 0, 1, 5, 0, 1, 6, 5, 0, 1, 5, 6, 0, 1, 5, 0, 1, 6]
# Hidden recipes: includes harder types (20 samples)
HIDDEN_RECIPES = [2, 3, 4, 6, 2, 3, 4, 0, 1, 5, 2, 3, 4, 6, 2, 3, 4, 0, 1, 5]


def generate_video(seed: int, T: int, recipe_id: int,
                   size: int = 512,
                   difficulty: str = "dev") -> np.ndarray:
    """Generate a procedural video clip.

    Args:
        seed: Random seed for reproducibility.
        T: Number of temporal frames (8, 16, or 32).
        recipe_id: Scene type (0-6), see RECIPE_NAMES.
        size: Spatial dimension (default 512).
        difficulty: "dev" for mild motion/occlusion, "hidden" for hard.

    Returns:
        x: np.ndarray of shape (size, size, T), values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Scale motion magnitude with T so long sequences stay solvable
    motion_scale = 8.0 / T  # smaller per-frame motion for larger T

    if difficulty == "hidden":
        motion_scale *= 1.5  # faster overall for hidden

    recipe_fn = _RECIPE_DISPATCH[recipe_id]
    x = recipe_fn(rng, T, size, motion_scale, difficulty)

    return np.clip(x, 0.0, 1.0)


# ── Shared primitives: backgrounds ──────────────────────────────────────────

def _fbm_noise(rng: np.random.Generator, H: int, W: int,
               octaves: int = 5, persistence: float = 0.5) -> np.ndarray:
    """Fractional Brownian Motion noise (multi-scale Perlin-like)."""
    result = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    for o in range(octaves):
        freq = 2 ** o
        noise = rng.standard_normal((max(H // freq, 2), max(W // freq, 2)))
        from PIL import Image
        img = Image.fromarray(noise)
        img = img.resize((W, H), Image.BILINEAR)
        result += amplitude * np.array(img, dtype=np.float64)
        amplitude *= persistence
    result -= result.min()
    mx = result.max()
    if mx > 1e-8:
        result /= mx
    return result


def _smooth_gradient(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    """Random smooth illumination gradient."""
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    yy = yy / H - 0.5
    xx = xx / W - 0.5
    grad = 0.5 + 0.4 * (np.cos(angle) * xx + np.sin(angle) * yy)
    return np.clip(grad, 0, 1)


def _make_background(rng: np.random.Generator, H: int, W: int,
                     style: str = "fbm") -> np.ndarray:
    """Generate a static background texture with many pattern options."""
    if style == "fbm":
        bg = _fbm_noise(rng, H, W, octaves=5)
        grad = _smooth_gradient(rng, H, W)
        return np.clip(0.6 * bg + 0.4 * grad, 0, 1)

    elif style == "grid":
        period = rng.integers(16, 64)
        yy, xx = np.mgrid[0:H, 0:W]
        grid = ((yy % period < period // 2) ^ (xx % period < period // 2)).astype(np.float64)
        grid = gaussian_filter(grid, sigma=1.0)
        base = _fbm_noise(rng, H, W, octaves=3) * 0.3
        return np.clip(0.7 * grid + 0.3 * base, 0, 1)

    elif style == "stripe":
        angle = rng.uniform(0, np.pi)
        freq = rng.uniform(0.02, 0.1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        pattern = 0.5 + 0.5 * np.sin(freq * (np.cos(angle) * xx + np.sin(angle) * yy))
        base = _fbm_noise(rng, H, W, octaves=3) * 0.2
        return np.clip(pattern + base, 0, 1)

    elif style == "radial":
        # Concentric rings emanating from a random center
        cy = rng.uniform(0.2 * H, 0.8 * H)
        cx = rng.uniform(0.2 * W, 0.8 * W)
        freq = rng.uniform(0.03, 0.12)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rings = 0.5 + 0.5 * np.sin(freq * dist)
        base = _fbm_noise(rng, H, W, octaves=3) * 0.2
        return np.clip(0.7 * rings + 0.3 * base, 0, 1)

    elif style == "dots":
        # Regular dot pattern
        spacing = rng.integers(20, 60)
        dot_r = rng.uniform(3, spacing * 0.3)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        # Distance to nearest grid point
        gy = (yy % spacing) - spacing / 2
        gx = (xx % spacing) - spacing / 2
        dist = np.sqrt(gy**2 + gx**2)
        dots = np.clip(1.0 - dist / dot_r, 0, 1)
        base = _fbm_noise(rng, H, W, octaves=3) * 0.3
        bg_val = rng.uniform(0.1, 0.4)
        return np.clip(bg_val + 0.6 * dots + 0.2 * base, 0, 1)

    elif style == "brick":
        # Brick / herringbone pattern
        bh = rng.integers(16, 40)
        bw = rng.integers(30, 80)
        yy, xx = np.mgrid[0:H, 0:W]
        row = yy // bh
        # Offset every other row
        x_shifted = xx + (row % 2) * (bw // 2)
        mortar_h = (yy % bh < 2).astype(np.float64)
        mortar_v = (x_shifted % bw < 2).astype(np.float64)
        mortar = np.maximum(mortar_h, mortar_v)
        brick_noise = _fbm_noise(rng, H, W, octaves=4) * 0.3
        brick_base = rng.uniform(0.4, 0.7)
        return np.clip(brick_base + brick_noise - 0.4 * mortar, 0, 1)

    elif style == "voronoi":
        # Voronoi-like cells via nearest-point distance
        n_pts = rng.integers(20, 80)
        pts_y = rng.uniform(0, H, n_pts)
        pts_x = rng.uniform(0, W, n_pts)
        pts_val = rng.uniform(0.2, 0.9, n_pts)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        result = np.zeros((H, W), dtype=np.float64)
        min_dist = np.full((H, W), 1e9)
        for i in range(n_pts):
            d = np.sqrt((yy - pts_y[i])**2 + (xx - pts_x[i])**2)
            closer = d < min_dist
            result[closer] = pts_val[i]
            min_dist[closer] = d[closer]
        # Add edge lines
        # Compute second-nearest distance for edge detection
        edge = gaussian_filter(result, sigma=1.0)
        edge_detect = np.abs(result - edge)
        result = result * (1 - 3 * edge_detect)
        return np.clip(result, 0, 1)

    elif style == "hexgrid":
        # Hexagonal grid pattern
        cell_size = rng.integers(20, 50)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        # Hex coordinates
        q = (2.0/3 * xx) / cell_size
        r_hex = (-1.0/3 * xx + np.sqrt(3)/3 * yy) / cell_size
        # Round to nearest hex center
        qr = np.round(q)
        rr = np.round(r_hex)
        sr = np.round(-q - r_hex)
        # Fix rounding
        q_diff = np.abs(qr - q)
        r_diff = np.abs(rr - r_hex)
        s_diff = np.abs(sr - (-q - r_hex))
        mask_q = (q_diff > r_diff) & (q_diff > s_diff)
        mask_r = ~mask_q & (r_diff > s_diff)
        qr[mask_q] = -rr[mask_q] - sr[mask_q]
        rr[mask_r] = -qr[mask_r] - sr[mask_r]
        # Color by hex cell
        cell_id = (qr * 7 + rr * 13) % 5
        pattern = cell_id / 5.0
        base = _fbm_noise(rng, H, W, octaves=3) * 0.2
        return np.clip(0.3 + 0.5 * pattern + base, 0, 1)

    else:
        return _fbm_noise(rng, H, W)


# ── Shared primitives: shapes ───────────────────────────────────────────────

def _render_circle(canvas: np.ndarray, cy: float, cx: float, r: float,
                   value: float, alpha: float = 1.0):
    """Render a filled circle with alpha blending."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask = np.clip(1.0 - (dist - r) / max(r * 0.1, 1.0), 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_rect(canvas: np.ndarray, cy: float, cx: float, h: float, w: float,
                 angle: float, value: float, alpha: float = 1.0):
    """Render a rotated rectangle with alpha blending."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy = yy - cy
    dx = xx - cx
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ly = cos_a * dy + sin_a * dx
    lx = -sin_a * dy + cos_a * dx
    inside = (np.abs(ly) < h/2) & (np.abs(lx) < w/2)
    dist_y = np.clip(1.0 - (np.abs(ly) - h/2 + 2) / 2, 0, 1)
    dist_x = np.clip(1.0 - (np.abs(lx) - w/2 + 2) / 2, 0, 1)
    mask = dist_y * dist_x * inside.astype(np.float64)
    mask = np.clip(mask, 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_ellipse(canvas: np.ndarray, cy: float, cx: float,
                    ry: float, rx: float, angle: float,
                    value: float, alpha: float = 1.0):
    """Render a rotated ellipse with alpha blending."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ly = (cos_a * dy + sin_a * dx) / max(ry, 1)
    lx = (-sin_a * dy + cos_a * dx) / max(rx, 1)
    dist = np.sqrt(ly**2 + lx**2)
    mask = np.clip(1.0 - (dist - 1.0) * min(ry, rx) * 0.5, 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_ring(canvas: np.ndarray, cy: float, cx: float,
                 r_outer: float, r_inner: float,
                 value: float, alpha: float = 1.0):
    """Render an annulus (ring) with alpha blending."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    outer = np.clip(1.0 - (dist - r_outer) / max(r_outer * 0.05, 1.0), 0, 1)
    inner = np.clip(1.0 - (r_inner - dist) / max(r_inner * 0.05, 1.0), 0, 1)
    mask = outer * inner
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_triangle(canvas: np.ndarray, cy: float, cx: float,
                     size_t: float, angle: float,
                     value: float, alpha: float = 1.0):
    """Render a filled equilateral triangle with alpha blending."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ly = cos_a * dy + sin_a * dx
    lx = -sin_a * dy + cos_a * dx
    # Three half-planes define an equilateral triangle
    h = size_t * np.sqrt(3) / 2
    d1 = ly + h / 3           # bottom edge
    d2 = -0.5 * ly + np.sqrt(3)/2 * lx + h / 3
    d3 = -0.5 * ly - np.sqrt(3)/2 * lx + h / 3
    inside = np.minimum(np.minimum(d1, d2), d3)
    mask = np.clip(inside / max(size_t * 0.05, 1.0), 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_cross(canvas: np.ndarray, cy: float, cx: float,
                  arm_len: float, arm_w: float, angle: float,
                  value: float, alpha: float = 1.0):
    """Render a rotated cross/plus shape."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ly = cos_a * dy + sin_a * dx
    lx = -sin_a * dy + cos_a * dx
    # Horizontal arm
    h_arm = (np.abs(ly) < arm_w / 2) & (np.abs(lx) < arm_len / 2)
    # Vertical arm
    v_arm = (np.abs(lx) < arm_w / 2) & (np.abs(ly) < arm_len / 2)
    inside = (h_arm | v_arm).astype(np.float64)
    mask = gaussian_filter(inside, sigma=1.0)
    mask = np.clip(mask, 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_arc(canvas: np.ndarray, cy: float, cx: float,
                r: float, width: float, start_angle: float,
                sweep: float, value: float, alpha: float = 0.9):
    """Render an arc (partial ring)."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # Radial band
    radial = np.clip(1.0 - np.abs(dist - r) / max(width / 2, 1.0), 0, 1)
    # Angular constraint
    angle = np.arctan2(yy - cy, xx - cx)
    # Normalize angle relative to start
    rel = (angle - start_angle + np.pi) % (2 * np.pi) - np.pi
    angular = (np.abs(rel) < sweep / 2).astype(np.float64)
    mask = radial * angular
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


def _render_line(canvas: np.ndarray, y0: float, x0: float,
                 y1: float, x1: float, width: float, value: float):
    """Render a line segment with given width."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2) + 1e-8
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (length**2), 0, 1)
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    dist = np.sqrt((xx - proj_x)**2 + (yy - proj_y)**2)
    mask = np.clip(1.0 - (dist - width/2) / max(width * 0.2, 1.0), 0, 1)
    canvas[:] = np.maximum(canvas, value * mask)


def _render_star(canvas: np.ndarray, cy: float, cx: float,
                 r_outer: float, r_inner: float, n_points: int,
                 angle: float, value: float, alpha: float = 1.0):
    """Render a star polygon via angular distance modulation."""
    H, W = canvas.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - cy, xx - cx
    dist = np.sqrt(dy**2 + dx**2) + 1e-8
    ang = np.arctan2(dy, dx) - angle
    # Modulate radius between inner and outer based on angle
    phase = (ang * n_points) % (2 * np.pi)
    r_boundary = r_inner + (r_outer - r_inner) * (0.5 + 0.5 * np.cos(phase))
    mask = np.clip(1.0 - (dist - r_boundary) / max(r_outer * 0.05, 1.0), 0, 1)
    canvas[:] = canvas * (1 - alpha * mask) + value * alpha * mask


# ── Shared primitives: motion trajectories ──────────────────────────────────

def _motion_linear(t, vy, vx):
    """Constant-velocity linear motion."""
    return vy * t, vx * t


def _motion_sinusoidal(t, amplitude_y, amplitude_x, freq_y, freq_x,
                       phase_y=0.0, phase_x=0.0):
    """Sinusoidal / oscillatory motion."""
    dy = amplitude_y * np.sin(2 * np.pi * freq_y * t + phase_y)
    dx = amplitude_x * np.sin(2 * np.pi * freq_x * t + phase_x)
    return dy, dx


def _motion_spiral(t, center_vy, center_vx, radius, angular_speed):
    """Spiral: linear drift + circular orbit."""
    dy = center_vy * t + radius * np.sin(angular_speed * t)
    dx = center_vx * t + radius * np.cos(angular_speed * t)
    return dy, dx


def _motion_bounce(t, vy, vx, bounds_y, bounds_x, T):
    """Bouncing motion within bounds (triangular wave)."""
    # Triangular wave: period = 2 * range / speed
    def _tri(pos, speed, lo, hi):
        rng = hi - lo
        if abs(speed) < 1e-8:
            return 0.0
        period = 2 * rng / abs(speed)
        phase = (pos + speed * t) % period if period > 0 else 0.0
        # Triangular wave
        if isinstance(phase, np.ndarray):
            return np.where(phase < period / 2,
                            phase * abs(speed) / abs(speed),
                            (period - phase) * abs(speed) / abs(speed))
        return phase if phase < period / 2 else period - phase
    # Simpler: just reflect via modular arithmetic
    raw_y = vy * t
    raw_x = vx * t
    # Fold into [0, 2*bounds] then reflect
    by, bx = bounds_y, bounds_x
    if by > 0:
        raw_y = raw_y % (2 * by)
        if raw_y > by:
            raw_y = 2 * by - raw_y
    if bx > 0:
        raw_x = raw_x % (2 * bx)
        if raw_x > bx:
            raw_x = 2 * bx - raw_x
    return raw_y, raw_x


def _motion_lissajous(t, amp_y, amp_x, freq_y, freq_x, phase_diff=0.0):
    """Lissajous curve motion (figure-8, etc.)."""
    dy = amp_y * np.sin(2 * np.pi * freq_y * t)
    dx = amp_x * np.sin(2 * np.pi * freq_x * t + phase_diff)
    return dy, dx


def _motion_pendulum(t, length, amplitude, phase=0.0):
    """Pendulum-like oscillation (mostly horizontal)."""
    theta = amplitude * np.sin(2 * np.pi * t / (2 * np.pi * np.sqrt(length)) + phase)
    dx = length * np.sin(theta)
    dy = length * (1 - np.cos(theta))
    return dy, dx


# ── Shared primitives: transforms ───────────────────────────────────────────

def _apply_global_transform(frame: np.ndarray, dx: float, dy: float,
                            angle: float, zoom: float) -> np.ndarray:
    """Apply affine camera transform (translate + rotate + zoom)."""
    H, W = frame.shape
    cy, cx = H / 2.0, W / 2.0
    cos_a = np.cos(angle) * zoom
    sin_a = np.sin(angle) * zoom
    matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    offset = np.array([
        cy - cos_a * cy - sin_a * cx + dy,
        cx + sin_a * cy - cos_a * cx + dx,
    ])
    return affine_transform(frame, matrix, offset=offset, order=1, mode='reflect')


def _apply_photometric(frame: np.ndarray, t: int, T: int,
                       rng: np.random.Generator,
                       flicker_amp: float = 0.05,
                       phase: float = 0.0) -> np.ndarray:
    """Apply exposure flicker."""
    brightness = 1.0 + flicker_amp * np.sin(2 * np.pi * t / T + phase)
    return np.clip(frame * brightness, 0, 1)


# ── Helper: pick random shape and render ────────────────────────────────────

def _random_shape_obj(rng, size, motion_scale, difficulty):
    """Create a random shape descriptor with a random motion model."""
    shape_types = ["rect", "circle", "ellipse", "triangle", "ring",
                   "cross", "star", "arc"]
    motion_types = ["linear", "sinusoidal", "spiral", "lissajous"]

    shape = rng.choice(shape_types)
    motion = rng.choice(motion_types)

    obj = {
        "shape": shape,
        "motion": motion,
        "cy": rng.uniform(60, size - 60),
        "cx": rng.uniform(60, size - 60),
        "value": rng.uniform(0.2, 0.95),
        "alpha": rng.uniform(0.5, 1.0),
        "angle": rng.uniform(0, 2 * np.pi),
        "va": rng.uniform(-0.03, 0.03) * motion_scale,
        # Shape params
        "r": rng.uniform(15, 80),
        "ry": rng.uniform(20, 80),
        "rx": rng.uniform(20, 80),
        "h": rng.uniform(25, 120),
        "w": rng.uniform(25, 120),
        "size_t": rng.uniform(30, 100),  # triangle size
        "arm_len": rng.uniform(30, 100),
        "arm_w": rng.uniform(8, 25),
        "r_outer": rng.uniform(30, 80),
        "r_inner": rng.uniform(10, 40),
        "n_points": int(rng.integers(4, 9)),
        "arc_start": rng.uniform(0, 2 * np.pi),
        "arc_sweep": rng.uniform(np.pi / 3, np.pi * 1.5),
        "line_width": rng.uniform(3, 12),
    }
    # Ensure r_inner < r_outer for ring/star
    if obj["r_inner"] >= obj["r_outer"]:
        obj["r_inner"] = obj["r_outer"] * 0.5

    # Motion params
    speed = rng.uniform(1, 5) * motion_scale
    if motion == "linear":
        obj["vy"] = rng.uniform(-1, 1) * speed
        obj["vx"] = rng.uniform(-1, 1) * speed
    elif motion == "sinusoidal":
        obj["amp_y"] = rng.uniform(10, 60) * motion_scale
        obj["amp_x"] = rng.uniform(10, 60) * motion_scale
        obj["freq_y"] = rng.uniform(0.02, 0.15)
        obj["freq_x"] = rng.uniform(0.02, 0.15)
        obj["phase_y"] = rng.uniform(0, 2 * np.pi)
        obj["phase_x"] = rng.uniform(0, 2 * np.pi)
    elif motion == "spiral":
        obj["center_vy"] = rng.uniform(-1, 1) * speed * 0.3
        obj["center_vx"] = rng.uniform(-1, 1) * speed * 0.3
        obj["orbit_r"] = rng.uniform(10, 50) * motion_scale
        obj["angular_speed"] = rng.uniform(0.1, 0.5)
    elif motion == "lissajous":
        obj["amp_y"] = rng.uniform(15, 70) * motion_scale
        obj["amp_x"] = rng.uniform(15, 70) * motion_scale
        obj["freq_y"] = rng.choice([1, 2, 3]) * 0.05
        obj["freq_x"] = rng.choice([1, 2, 3]) * 0.05
        obj["phase_diff"] = rng.uniform(0, np.pi)

    return obj


def _get_obj_position(obj, t):
    """Compute position offset (dy, dx) for object at time t."""
    motion = obj["motion"]
    if motion == "linear":
        return obj["vy"] * t, obj["vx"] * t
    elif motion == "sinusoidal":
        return _motion_sinusoidal(t, obj["amp_y"], obj["amp_x"],
                                  obj["freq_y"], obj["freq_x"],
                                  obj["phase_y"], obj["phase_x"])
    elif motion == "spiral":
        return _motion_spiral(t, obj["center_vy"], obj["center_vx"],
                              obj["orbit_r"], obj["angular_speed"])
    elif motion == "lissajous":
        return _motion_lissajous(t, obj["amp_y"], obj["amp_x"],
                                 obj["freq_y"], obj["freq_x"],
                                 obj["phase_diff"])
    return 0.0, 0.0


def _draw_obj(canvas, obj, t):
    """Draw a shape object on canvas at time t."""
    dy, dx = _get_obj_position(obj, t)
    cy = obj["cy"] + dy
    cx = obj["cx"] + dx
    a = obj["angle"] + obj["va"] * t
    v, al = obj["value"], obj["alpha"]
    shape = obj["shape"]

    if shape == "rect":
        _render_rect(canvas, cy, cx, obj["h"], obj["w"], a, v, al)
    elif shape == "circle":
        _render_circle(canvas, cy, cx, obj["r"], v, al)
    elif shape == "ellipse":
        _render_ellipse(canvas, cy, cx, obj["ry"], obj["rx"], a, v, al)
    elif shape == "triangle":
        _render_triangle(canvas, cy, cx, obj["size_t"], a, v, al)
    elif shape == "ring":
        _render_ring(canvas, cy, cx, obj["r_outer"], obj["r_inner"], v, al)
    elif shape == "cross":
        _render_cross(canvas, cy, cx, obj["arm_len"], obj["arm_w"], a, v, al)
    elif shape == "star":
        _render_star(canvas, cy, cx, obj["r_outer"], obj["r_inner"],
                     obj["n_points"], a, v, al)
    elif shape == "arc":
        _render_arc(canvas, cy, cx, obj["r"], obj["line_width"],
                    obj["arc_start"] + a, obj["arc_sweep"], v, al)


# ── Recipe implementations ───────────────────────────────────────────────────

def _recipe_urban(rng, T, size, motion_scale, difficulty):
    """Urban: rectangle blobs + edges + grid/brick/hex patterns + diverse shapes."""
    bg_style = rng.choice(["grid", "brick", "hexgrid"])
    bg = _make_background(rng, size, size, bg_style)
    n_obj = rng.integers(5, 12) if difficulty == "dev" else rng.integers(8, 20)

    objects = []
    for _ in range(n_obj):
        obj = _random_shape_obj(rng, size, motion_scale, difficulty)
        # Bias toward rectangular/structural shapes for urban
        obj["shape"] = rng.choice(["rect", "rect", "cross", "triangle",
                                   "ellipse", "circle", "ring"])
        objects.append(obj)

    # Add a few line structures (roads/edges)
    n_lines = rng.integers(2, 6)
    line_objs = []
    for _ in range(n_lines):
        line_objs.append({
            "y0": rng.uniform(0, size), "x0": rng.uniform(0, size),
            "y1": rng.uniform(0, size), "x1": rng.uniform(0, size),
            "width": rng.uniform(2, 8),
            "value": rng.uniform(0.1, 0.4),
            "vy": rng.uniform(-1, 1) * motion_scale,
            "vx": rng.uniform(-1, 1) * motion_scale,
        })

    flicker_phase = rng.uniform(0, 2 * np.pi)
    cam_dx = rng.uniform(-0.8, 0.8) * motion_scale
    cam_dy = rng.uniform(-0.8, 0.8) * motion_scale
    cam_rot_speed = rng.uniform(-0.002, 0.002) * motion_scale

    frames = []
    for t in range(T):
        frame = bg.copy()
        # Draw line structures
        for ln in line_objs:
            _render_line(frame,
                         ln["y0"] + ln["vy"]*t, ln["x0"] + ln["vx"]*t,
                         ln["y1"] + ln["vy"]*t, ln["x1"] + ln["vx"]*t,
                         ln["width"], ln["value"])
        # Draw objects
        for obj in objects:
            _draw_obj(frame, obj, t)
        frame = _apply_global_transform(frame, cam_dx * t, cam_dy * t,
                                        cam_rot_speed * t, 1.0)
        frame = _apply_photometric(frame, t, T, rng, 0.03, flicker_phase)
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_nature(rng, T, size, motion_scale, difficulty):
    """Nature: smooth textures + soft objects with organic motion."""
    bg_style = rng.choice(["fbm", "radial", "voronoi"])
    bg = _make_background(rng, size, size, bg_style)
    n_obj = rng.integers(3, 8) if difficulty == "dev" else rng.integers(5, 12)

    objects = []
    for _ in range(n_obj):
        obj = _random_shape_obj(rng, size, motion_scale, difficulty)
        # Bias toward organic shapes
        obj["shape"] = rng.choice(["circle", "circle", "ellipse", "ellipse",
                                   "ring", "arc", "star"])
        # Bias toward oscillatory/spiral motion for organic feel
        obj["motion"] = rng.choice(["sinusoidal", "sinusoidal", "spiral",
                                    "lissajous", "linear"])
        # Re-init motion params for chosen type
        speed = rng.uniform(1, 4) * motion_scale
        if obj["motion"] == "sinusoidal":
            obj["amp_y"] = rng.uniform(10, 50) * motion_scale
            obj["amp_x"] = rng.uniform(10, 50) * motion_scale
            obj["freq_y"] = rng.uniform(0.02, 0.1)
            obj["freq_x"] = rng.uniform(0.02, 0.1)
            obj["phase_y"] = rng.uniform(0, 2 * np.pi)
            obj["phase_x"] = rng.uniform(0, 2 * np.pi)
        elif obj["motion"] == "spiral":
            obj["center_vy"] = rng.uniform(-0.5, 0.5) * speed
            obj["center_vx"] = rng.uniform(-0.5, 0.5) * speed
            obj["orbit_r"] = rng.uniform(8, 40) * motion_scale
            obj["angular_speed"] = rng.uniform(0.05, 0.3)
        elif obj["motion"] == "lissajous":
            obj["amp_y"] = rng.uniform(10, 50) * motion_scale
            obj["amp_x"] = rng.uniform(10, 50) * motion_scale
            obj["freq_y"] = rng.choice([1, 2, 3]) * 0.04
            obj["freq_x"] = rng.choice([1, 2, 3]) * 0.04
            obj["phase_diff"] = rng.uniform(0, np.pi)
        else:
            obj["vy"] = rng.uniform(-2, 2) * motion_scale
            obj["vx"] = rng.uniform(-2, 2) * motion_scale
        objects.append(obj)

    flicker_phase = rng.uniform(0, 2 * np.pi)

    frames = []
    for t in range(T):
        frame = bg.copy()
        # Slowly vary background
        drift = 0.04 * np.sin(2 * np.pi * t / T + flicker_phase)
        frame = np.clip(frame + drift, 0, 1)

        for obj in objects:
            _draw_obj(frame, obj, t)

        frame = gaussian_filter(frame, sigma=0.8)
        frame = _apply_photometric(frame, t, T, rng, 0.05, flicker_phase)
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_textile(rng, T, size, motion_scale, difficulty):
    """Textile: multi-layered periodic textures + diverse moving objects."""
    bg_style = rng.choice(["stripe", "dots", "hexgrid"])
    bg = _make_background(rng, size, size, bg_style)

    # Overlay a second periodic pattern
    overlay_style = rng.choice(["stripe", "dots", "grid"])
    overlay = _make_background(rng, size, size, overlay_style)
    blend = rng.uniform(0.3, 0.6)
    bg = blend * bg + (1 - blend) * overlay

    n_obj = rng.integers(3, 8) if difficulty == "dev" else rng.integers(6, 14)
    objects = []
    for _ in range(n_obj):
        obj = _random_shape_obj(rng, size, motion_scale, difficulty)
        # Mix of shapes for textile
        obj["shape"] = rng.choice(["rect", "circle", "ellipse", "triangle",
                                   "cross", "star"])
        objects.append(obj)

    flicker_phase = rng.uniform(0, 2 * np.pi)
    # Background pattern scrolls
    scroll_vy = rng.uniform(-1.5, 1.5) * motion_scale
    scroll_vx = rng.uniform(-1.5, 1.5) * motion_scale

    frames = []
    for t in range(T):
        frame = bg.copy()
        # Scroll the background texture
        sy, sx = int(scroll_vy * t), int(scroll_vx * t)
        if sy != 0:
            frame = np.roll(frame, sy, axis=0)
        if sx != 0:
            frame = np.roll(frame, sx, axis=1)

        for obj in objects:
            _draw_obj(frame, obj, t)

        frame = _apply_photometric(frame, t, T, rng, 0.04, flicker_phase)
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_particles(rng, T, size, motion_scale, difficulty):
    """Particles: many tiny moving dots + varied sizes and trajectories."""
    bg_style = rng.choice(["fbm", "radial"])
    bg = _make_background(rng, size, size, bg_style)
    bg = gaussian_filter(bg, sigma=3.0)  # smooth background

    n_particles = rng.integers(40, 120) if difficulty == "dev" else rng.integers(80, 250)

    particles = []
    for _ in range(n_particles):
        motion = rng.choice(["linear", "linear", "sinusoidal", "spiral"])
        p = {
            "cy": rng.uniform(0, size),
            "cx": rng.uniform(0, size),
            "r": rng.uniform(2, 10) if difficulty == "dev" else rng.uniform(1, 7),
            "value": rng.uniform(0.4, 1.0),
            "shape": rng.choice(["circle", "circle", "circle", "star", "triangle"]),
            "motion": motion,
            "angle": rng.uniform(0, 2 * np.pi),
            "va": 0,
            "alpha": rng.uniform(0.7, 1.0),
            # Shape params (for star/triangle)
            "r_outer": rng.uniform(3, 8),
            "r_inner": rng.uniform(1, 4),
            "n_points": int(rng.integers(4, 7)),
            "size_t": rng.uniform(3, 10),
        }
        if p["r_inner"] >= p["r_outer"]:
            p["r_inner"] = p["r_outer"] * 0.4
        speed = rng.uniform(2, 7) * motion_scale
        if motion == "linear":
            p["vy"] = rng.uniform(-1, 1) * speed
            p["vx"] = rng.uniform(-1, 1) * speed
            p["ay"] = rng.uniform(-0.3, 0.3) * motion_scale
            p["ax"] = rng.uniform(-0.3, 0.3) * motion_scale
        elif motion == "sinusoidal":
            p["amp_y"] = rng.uniform(5, 30) * motion_scale
            p["amp_x"] = rng.uniform(5, 30) * motion_scale
            p["freq_y"] = rng.uniform(0.03, 0.2)
            p["freq_x"] = rng.uniform(0.03, 0.2)
            p["phase_y"] = rng.uniform(0, 2 * np.pi)
            p["phase_x"] = rng.uniform(0, 2 * np.pi)
        elif motion == "spiral":
            p["center_vy"] = rng.uniform(-0.5, 0.5) * speed
            p["center_vx"] = rng.uniform(-0.5, 0.5) * speed
            p["orbit_r"] = rng.uniform(3, 15) * motion_scale
            p["angular_speed"] = rng.uniform(0.1, 0.6)
        particles.append(p)

    # A few larger "emitter" circles that particles orbit
    n_emitters = rng.integers(0, 4)
    emitters = []
    for _ in range(n_emitters):
        emitters.append({
            "cy": rng.uniform(100, size - 100),
            "cx": rng.uniform(100, size - 100),
            "r": rng.uniform(20, 50),
            "value": rng.uniform(0.15, 0.35),
            "alpha": 0.4,
        })

    frames = []
    for t in range(T):
        frame = bg.copy()

        # Draw emitters
        for em in emitters:
            _render_circle(frame, em["cy"], em["cx"], em["r"],
                           em["value"], em["alpha"])

        # Draw particles
        for p in particles:
            if p["motion"] == "linear":
                dy = p["vy"] * t + 0.5 * p.get("ay", 0) * t * t
                dx = p["vx"] * t + 0.5 * p.get("ax", 0) * t * t
            else:
                dy, dx = _get_obj_position(p, t)
            cy = (p["cy"] + dy) % size
            cx = (p["cx"] + dx) % size

            if p["shape"] == "circle":
                _render_circle(frame, cy, cx, p["r"], p["value"], p["alpha"])
            elif p["shape"] == "star":
                _render_star(frame, cy, cx, p["r_outer"], p["r_inner"],
                             p["n_points"], p["angle"], p["value"], p["alpha"])
            elif p["shape"] == "triangle":
                _render_triangle(frame, cy, cx, p["size_t"], p["angle"],
                                 p["value"], p["alpha"])

        frame = _apply_photometric(frame, t, T, rng, 0.02, rng.uniform(0, 2*np.pi))
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_thin_struct(rng, T, size, motion_scale, difficulty):
    """Thin structures: lines, arcs, wires + circles, crosses (stress test)."""
    bg_style = rng.choice(["fbm", "voronoi"])
    bg = _make_background(rng, size, size, bg_style)
    bg = bg * 0.3 + 0.1  # dark background

    n_lines = rng.integers(8, 20) if difficulty == "dev" else rng.integers(15, 40)
    n_arcs = rng.integers(3, 8) if difficulty == "dev" else rng.integers(5, 15)

    lines = []
    for _ in range(n_lines):
        motion = rng.choice(["linear", "sinusoidal"])
        ln = {
            "y0": rng.uniform(0, size), "x0": rng.uniform(0, size),
            "y1": rng.uniform(0, size), "x1": rng.uniform(0, size),
            "width": rng.uniform(0.8, 5) if difficulty == "dev" else rng.uniform(0.5, 3),
            "value": rng.uniform(0.5, 1.0),
            "motion": motion,
        }
        if motion == "linear":
            ln["vy"] = rng.uniform(-3, 3) * motion_scale
            ln["vx"] = rng.uniform(-3, 3) * motion_scale
        else:
            ln["amp_y"] = rng.uniform(5, 30) * motion_scale
            ln["amp_x"] = rng.uniform(5, 30) * motion_scale
            ln["freq"] = rng.uniform(0.03, 0.15)
            ln["phase"] = rng.uniform(0, 2 * np.pi)
        lines.append(ln)

    arcs = []
    for _ in range(n_arcs):
        arcs.append({
            "cy": rng.uniform(50, size - 50),
            "cx": rng.uniform(50, size - 50),
            "r": rng.uniform(30, 150),
            "width": rng.uniform(1, 5),
            "start": rng.uniform(0, 2 * np.pi),
            "sweep": rng.uniform(np.pi / 4, np.pi * 1.5),
            "value": rng.uniform(0.5, 1.0),
            "vy": rng.uniform(-2, 2) * motion_scale,
            "vx": rng.uniform(-2, 2) * motion_scale,
            "vr": rng.uniform(-0.05, 0.05) * motion_scale,  # rotation
        })

    # Some crosses and rings for variety
    n_extra = rng.integers(3, 8)
    extras = []
    for _ in range(n_extra):
        obj = _random_shape_obj(rng, size, motion_scale, difficulty)
        obj["shape"] = rng.choice(["cross", "ring", "circle", "star"])
        obj["r"] = rng.uniform(10, 40)
        obj["r_outer"] = rng.uniform(15, 50)
        obj["r_inner"] = obj["r_outer"] * rng.uniform(0.3, 0.7)
        obj["arm_len"] = rng.uniform(15, 50)
        obj["arm_w"] = rng.uniform(2, 8)
        extras.append(obj)

    frames = []
    for t in range(T):
        frame = bg.copy()

        for obj in extras:
            _draw_obj(frame, obj, t)

        for ln in lines:
            if ln["motion"] == "linear":
                oy, ox = ln["vy"] * t, ln["vx"] * t
            else:
                oy = ln["amp_y"] * np.sin(2*np.pi*ln["freq"]*t + ln["phase"])
                ox = ln["amp_x"] * np.sin(2*np.pi*ln["freq"]*t + ln["phase"] + np.pi/3)
            _render_line(frame,
                         ln["y0"] + oy, ln["x0"] + ox,
                         ln["y1"] + oy, ln["x1"] + ox,
                         ln["width"], ln["value"])

        for arc in arcs:
            _render_arc(frame,
                        arc["cy"] + arc["vy"]*t,
                        arc["cx"] + arc["vx"]*t,
                        arc["r"], arc["width"],
                        arc["start"] + arc["vr"]*t,
                        arc["sweep"], arc["value"])

        frame = _apply_photometric(frame, t, T, rng, 0.03, 0)
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_occlusion(rng, T, size, motion_scale, difficulty):
    """Occlusion: diverse layered objects with crossing trajectories."""
    bg_style = rng.choice(["fbm", "voronoi", "radial", "brick"])
    bg = _make_background(rng, size, size, bg_style)
    n_obj = rng.integers(5, 12) if difficulty == "dev" else rng.integers(8, 20)

    objects = []
    for k in range(n_obj):
        obj = _random_shape_obj(rng, size, motion_scale, difficulty)
        # Ensure crossing trajectories by alternating direction bias
        speed = rng.uniform(2, 6) * motion_scale
        direction = 1 if k % 2 == 0 else -1
        # Override motion for crossing pattern
        motion_choice = rng.choice(["linear", "sinusoidal", "lissajous"])
        obj["motion"] = motion_choice
        if motion_choice == "linear":
            obj["vy"] = rng.uniform(0.5, 1.0) * speed * direction
            obj["vx"] = rng.uniform(0.5, 1.0) * speed * (-direction)
        elif motion_choice == "sinusoidal":
            obj["amp_y"] = rng.uniform(20, 80) * motion_scale
            obj["amp_x"] = rng.uniform(20, 80) * motion_scale
            obj["freq_y"] = rng.uniform(0.02, 0.1)
            obj["freq_x"] = rng.uniform(0.02, 0.1)
            obj["phase_y"] = k * np.pi / n_obj  # spread phases
            obj["phase_x"] = k * np.pi / n_obj + np.pi/2
        elif motion_choice == "lissajous":
            obj["amp_y"] = rng.uniform(20, 80) * motion_scale
            obj["amp_x"] = rng.uniform(20, 80) * motion_scale
            obj["freq_y"] = rng.choice([1, 2, 3]) * 0.04
            obj["freq_x"] = rng.choice([1, 2, 3]) * 0.04
            obj["phase_diff"] = k * np.pi / 4
        obj["alpha"] = rng.uniform(0.6, 1.0)
        obj["depth"] = k
        objects.append(obj)

    # Sort by depth for correct occlusion
    objects.sort(key=lambda o: o["depth"])

    flicker_phase = rng.uniform(0, 2 * np.pi)

    frames = []
    for t in range(T):
        frame = bg.copy()
        for obj in objects:
            _draw_obj(frame, obj, t)
        frame = _apply_photometric(frame, t, T, rng, 0.04, flicker_phase)
        frames.append(frame)

    return np.stack(frames, axis=-1)


def _recipe_cam_shake(rng, T, size, motion_scale, difficulty):
    """Camera shake dominant: diverse objects + strong global motion."""
    bg_style = rng.choice(["fbm", "grid", "brick", "voronoi"])
    bg = _make_background(rng, size, size, bg_style)

    n_obj = rng.integers(4, 10) if difficulty == "dev" else rng.integers(6, 15)
    objects = []
    for _ in range(n_obj):
        obj = _random_shape_obj(rng, size, motion_scale * 0.3, difficulty)
        # Objects move slowly — camera dominates
        obj["motion"] = rng.choice(["linear", "sinusoidal"])
        if obj["motion"] == "linear":
            obj["vy"] = rng.uniform(-0.5, 0.5) * motion_scale
            obj["vx"] = rng.uniform(-0.5, 0.5) * motion_scale
        else:
            obj["amp_y"] = rng.uniform(3, 15) * motion_scale
            obj["amp_x"] = rng.uniform(3, 15) * motion_scale
            obj["freq_y"] = rng.uniform(0.02, 0.08)
            obj["freq_x"] = rng.uniform(0.02, 0.08)
            obj["phase_y"] = rng.uniform(0, 2*np.pi)
            obj["phase_x"] = rng.uniform(0, 2*np.pi)
        objects.append(obj)

    # Strong camera motion: random walk
    shake_scale = 3.0 * motion_scale if difficulty == "dev" else 6.0 * motion_scale
    cam_traj_x = np.cumsum(rng.standard_normal(T) * shake_scale)
    cam_traj_y = np.cumsum(rng.standard_normal(T) * shake_scale)
    cam_rot = np.cumsum(rng.standard_normal(T) * 0.004 * motion_scale)
    cam_zoom = 1.0 + np.cumsum(rng.standard_normal(T) * 0.003 * motion_scale)
    cam_zoom = np.clip(cam_zoom, 0.93, 1.07)

    frames = []
    for t in range(T):
        frame = bg.copy()
        for obj in objects:
            _draw_obj(frame, obj, t)

        frame = _apply_global_transform(
            frame, cam_traj_x[t], cam_traj_y[t],
            cam_rot[t], cam_zoom[t]
        )
        frame = _apply_photometric(frame, t, T, rng, 0.06, rng.uniform(0, 2*np.pi))
        frames.append(frame)

    return np.stack(frames, axis=-1)


# ── Dispatch ─────────────────────────────────────────────────────────────────

_RECIPE_DISPATCH = [
    _recipe_urban,
    _recipe_nature,
    _recipe_textile,
    _recipe_particles,
    _recipe_thin_struct,
    _recipe_occlusion,
    _recipe_cam_shake,
]
