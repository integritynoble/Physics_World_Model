"""Procedural 3D phantom generator for CBCT benchmark dev/hidden tiers.

Generates fully synthetic 3D attenuation volumes mu in [0, 1] with
high structural complexity.  No external datasets required.

All recipes are explicitly inspired by the structural characteristics found
in recent, high-impact CBCT/CT benchmark datasets:
  - AAPM Low-Dose CT (Mayo, 2017) — chest/abdomen anatomy
  - LIDC-IDRI / ICASSP 2024 3D CBCT Challenge — lung nodules, airways
  - CBCTLiTS (2024) — liver tumors, portal vasculature, organ boundaries
  - MMDental (2025) / CTooth+ — dental arch, tooth roots, metal crowns
  - 2DeteCT (2023) — industrial multi-material, beam hardening modes
  - Helsinki Tomography Challenge (2022) — small-object, limited-angle
  - Walnut CT (CWI, 2019) / PCCT (2025) — shell micro-structure, porosity
  - CQ500 (2018) — head anatomy, skull vault, hemorrhage
  - DM4CT (2025) — rock micro-structure, Turing-like porosity patterns

Computation layers per phantom:
  1) Multi-scale 3D fBm noise fields (8 octaves, multiple passes)
  2) Worley/cellular noise for tissue microstructure
  3) Signed-distance-field geometry (ellipsoids, superquadrics, tori)
  4) L-system-inspired vascular/bronchial branching trees
  5) Gyroid / TPMS minimal surfaces for trabecular bone & scaffolds
  6) Elastic deformation fields for organic boundary irregularity
  7) Anisotropic directional textures (muscle fibers, cortical bone)
  8) Multi-material compositing with realistic HU-based attenuation
  9) Post-process: blur cascade, contrast, ring artifact simulation

Dev recipes (10 types, anatomy-inspired, medium complexity):
  head_cranial (CQ500), torso_thorax (AAPM/ICASSP), abdomen_organs
  (CBCTLiTS), extremity_bone (LoDoPaB), dental_arch (MMDental/CTooth+),
  pelvis_hip (AAPM), shoulder_complex (AAPM), knee_joint (LoDoPaB),
  spine_segment (AAPM), hand_wrist (HTC)

Hidden recipes (10 types, adversarial stress-tests, extreme complexity):
  trabecular_micro (Walnut/PCCT), multi_metal (AAPM+ortho),
  vascular_tree (CBCTLiTS), lung_parenchyma (LIDC/ICASSP),
  fractal_membrane (CBCTLiTS), gyroid_scaffold (2DeteCT/DM4CT),
  dental_metal (MMDental/CTooth+), cardiac_chambers (AAPM cardiac),
  multi_contrast (2DeteCT), reaction_diffusion (DM4CT)

Usage:
    from simulate_phantoms import generate_cbct_phantom
    mu, recipe = generate_cbct_phantom(seed=42, mode="dev")
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Low-level noise primitives
# ═══════════════════════════════════════════════════════════════════════════════

def fbm_noise_3d(
    shape: tuple[int, int, int],
    octaves: int = 8,
    persistence: float = 0.55,
    base_sigma: float = 2.0,
    lacunarity: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """High-octave fractional Brownian motion 3D noise."""
    rng = rng or np.random.default_rng()
    D, H, W = shape
    out = np.zeros(shape, dtype=np.float64)
    amp, total, sigma = 1.0, 0.0, base_sigma
    for _ in range(octaves):
        n = rng.standard_normal(shape).astype(np.float64)
        out += amp * gaussian_filter(n, sigma=max(sigma, 0.5))
        total += amp
        amp *= persistence
        sigma /= lacunarity
    out /= max(total, 1e-8)
    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out.astype(np.float32)


def worley_noise_3d(
    shape: tuple[int, int, int],
    n_pts: int = 200,
    mode: str = "F1",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Worley (cellular) noise — F1 (nearest) or F2-F1 (cell edges).

    Computationally intensive: evaluates distance to n_pts for every voxel
    via a block-decomposition approach for tractability.
    """
    rng = rng or np.random.default_rng()
    D, H, W = shape
    pts = np.column_stack([
        rng.uniform(0, D, n_pts),
        rng.uniform(0, H, n_pts),
        rng.uniform(0, W, n_pts),
    ]).astype(np.float32)

    # Block decomposition for memory efficiency
    block = 64
    f1 = np.full(shape, 1e9, dtype=np.float32)
    f2 = np.full(shape, 1e9, dtype=np.float32)

    for zs in range(0, D, block):
        ze = min(zs + block, D)
        for ys in range(0, H, block):
            ye = min(ys + block, H)
            for xs in range(0, W, block):
                xe = min(xs + block, W)
                zz, yy, xx = np.mgrid[zs:ze, ys:ye, xs:xe].astype(np.float32)
                for i in range(n_pts):
                    d = np.sqrt(
                        (zz - pts[i, 0])**2 +
                        (yy - pts[i, 1])**2 +
                        (xx - pts[i, 2])**2
                    )
                    update_f2 = d < f2[zs:ze, ys:ye, xs:xe]
                    f2[zs:ze, ys:ye, xs:xe] = np.where(update_f2, d, f2[zs:ze, ys:ye, xs:xe])
                    swap = f2[zs:ze, ys:ye, xs:xe] < f1[zs:ze, ys:ye, xs:xe]
                    tmp = f1[zs:ze, ys:ye, xs:xe].copy()
                    f1[zs:ze, ys:ye, xs:xe] = np.where(swap, f2[zs:ze, ys:ye, xs:xe], f1[zs:ze, ys:ye, xs:xe])
                    f2[zs:ze, ys:ye, xs:xe] = np.where(swap, tmp, f2[zs:ze, ys:ye, xs:xe])

    if mode == "F1":
        out = f1
    else:  # F2-F1 → cell edges
        out = f2 - f1

    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out.astype(np.float32)


def anisotropic_noise_3d(
    shape: tuple[int, int, int],
    direction: tuple[float, float, float] = (1, 0, 0),
    stretch: float = 4.0,
    octaves: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Directional noise for fiber-like textures (muscle, cortical bone)."""
    rng = rng or np.random.default_rng()
    d = np.array(direction, dtype=np.float64)
    d /= np.linalg.norm(d) + 1e-8
    sigma_along = 1.0 * stretch
    sigma_perp = 1.0
    # Compute anisotropic sigmas in each axis
    sigmas = [max(sigma_perp + (sigma_along - sigma_perp) * abs(d[i]), 0.5)
              for i in range(3)]
    out = np.zeros(shape, dtype=np.float64)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        n = rng.standard_normal(shape)
        n = gaussian_filter(n, sigma=sigmas)
        out += amp * n
        total += amp
        amp *= 0.5
        sigmas = [max(s * 0.7, 0.4) for s in sigmas]
    out /= max(total, 1e-8)
    out -= out.min()
    mx = out.max()
    if mx > 1e-8:
        out /= mx
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Geometric primitives & SDF operations
# ═══════════════════════════════════════════════════════════════════════════════

def _coords(shape):
    """Return (zz, yy, xx) coordinate grids."""
    D, H, W = shape
    return np.mgrid[0:D, 0:H, 0:W].astype(np.float32)


def sdf_ellipsoid(shape, center, radii):
    """Signed distance to ellipsoid surface (negative inside)."""
    zz, yy, xx = _coords(shape)
    cz, cy, cx = center
    rz, ry, rx = [max(r, 1) for r in radii]
    q = np.sqrt(((zz-cz)/rz)**2 + ((yy-cy)/ry)**2 + ((xx-cx)/rx)**2)
    return (q - 1.0) * min(rz, ry, rx)


def sdf_cylinder(shape, center, radius, half_length, axis=0):
    """SDF for axis-aligned cylinder."""
    zz, yy, xx = _coords(shape)
    cz, cy, cx = center
    if axis == 0:
        radial = np.sqrt((yy-cy)**2 + (xx-cx)**2) - radius
        axial = np.abs(zz-cz) - half_length
    elif axis == 1:
        radial = np.sqrt((zz-cz)**2 + (xx-cx)**2) - radius
        axial = np.abs(yy-cy) - half_length
    else:
        radial = np.sqrt((zz-cz)**2 + (yy-cy)**2) - radius
        axial = np.abs(xx-cx) - half_length
    return np.maximum(radial, axial)


def sdf_torus(shape, center, R, r, axis=0):
    """SDF for torus with major radius R, minor radius r."""
    zz, yy, xx = _coords(shape)
    cz, cy, cx = center
    if axis == 0:
        q = np.sqrt((yy-cy)**2 + (xx-cx)**2) - R
        return np.sqrt(q**2 + (zz-cz)**2) - r
    elif axis == 1:
        q = np.sqrt((zz-cz)**2 + (xx-cx)**2) - R
        return np.sqrt(q**2 + (yy-cy)**2) - r
    else:
        q = np.sqrt((zz-cz)**2 + (yy-cy)**2) - R
        return np.sqrt(q**2 + (xx-cx)**2) - r


def sdf_superquadric(shape, center, radii, epsilon1=1.0, epsilon2=1.0):
    """SDF approximation for superquadric (box-to-sphere morph)."""
    zz, yy, xx = _coords(shape)
    cz, cy, cx = center
    rz, ry, rx = [max(r, 1) for r in radii]
    e1 = max(epsilon1, 0.1)
    e2 = max(epsilon2, 0.1)
    t1 = np.abs((xx-cx)/rx)**(2.0/e2) + np.abs((yy-cy)/ry)**(2.0/e2)
    q = t1**(e2/e1) + np.abs((zz-cz)/rz)**(2.0/e1)
    return (q**(e1/2.0) - 1.0) * min(rz, ry, rx)


def sdf_smooth_union(d1, d2, k=4.0):
    """Smooth (polynomial) union of two SDF fields."""
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def sdf_to_mask(sdf, edge_width=1.5):
    """Convert SDF to smooth [0,1] mask (1 inside, 0 outside)."""
    return np.clip(0.5 - sdf / max(edge_width, 0.5), 0, 1).astype(np.float32)


def composite(vol, mask, value, alpha=1.0):
    """Alpha-composite a value into volume using mask."""
    m = mask * alpha
    return np.clip(vol * (1 - m) + value * m, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Structural generators
# ═══════════════════════════════════════════════════════════════════════════════

def elastic_deform(vol, rng, strength=8.0, sigma=20.0):
    """Apply random elastic deformation for organic boundary irregularity."""
    shape = vol.shape
    dz = gaussian_filter(rng.standard_normal(shape), sigma) * strength
    dy = gaussian_filter(rng.standard_normal(shape), sigma) * strength
    dx = gaussian_filter(rng.standard_normal(shape), sigma) * strength
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    coords = [
        np.clip(zz + dz, 0, D - 1),
        np.clip(yy + dy, 0, H - 1),
        np.clip(xx + dx, 0, W - 1),
    ]
    return map_coordinates(vol, coords, order=1, mode='reflect').astype(np.float32)


def gyroid_surface(shape, period=30.0, thickness=2.0):
    """Gyroid triply-periodic minimal surface — used for trabecular scaffolds."""
    D, H, W = shape
    zz, yy, xx = _coords(shape)
    s = 2 * np.pi / period
    g = (np.sin(s * xx) * np.cos(s * yy) +
         np.sin(s * yy) * np.cos(s * zz) +
         np.sin(s * zz) * np.cos(s * xx))
    return (np.abs(g) < thickness).astype(np.float32)


def schwarz_p_surface(shape, period=30.0, thickness=2.0):
    """Schwarz-P triply-periodic minimal surface."""
    D, H, W = shape
    zz, yy, xx = _coords(shape)
    s = 2 * np.pi / period
    p = np.cos(s * xx) + np.cos(s * yy) + np.cos(s * zz)
    return (np.abs(p) < thickness).astype(np.float32)


def branching_tree(
    shape, root, direction, length, radius,
    depth=5, branch_angle=0.5, decay=0.7,
    rng=None, segments=None,
):
    """L-system-inspired recursive branching tree (vessels, bronchi).

    Returns list of (p0, p1, radius) tube segments.
    """
    rng = rng or np.random.default_rng()
    if segments is None:
        segments = []
    if depth <= 0 or radius < 0.5 or length < 2:
        return segments

    root = np.array(root, dtype=np.float64)
    direction = np.array(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-8

    end = root + direction * length
    # Clip to volume bounds
    D, H, W = shape
    end = np.clip(end, [2, 2, 2], [D-3, H-3, W-3])
    segments.append((tuple(root), tuple(end), radius))

    # Branch into 2-3 children
    n_children = rng.integers(2, 4)
    for _ in range(n_children):
        # Random perturbation of direction
        perturb = rng.standard_normal(3) * branch_angle
        child_dir = direction + perturb
        child_dir /= np.linalg.norm(child_dir) + 1e-8

        child_length = length * decay * rng.uniform(0.6, 1.0)
        child_radius = radius * decay * rng.uniform(0.6, 0.9)  # Murray's law approx

        branching_tree(
            shape, end, child_dir, child_length, child_radius,
            depth=depth - 1, branch_angle=branch_angle * 1.1,
            decay=decay, rng=rng, segments=segments,
        )
    return segments


def render_tube(vol, p0, p1, radius, value, alpha=0.9):
    """Render a tube segment into volume (optimized with bounding box)."""
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    d = p1 - p0
    length = np.linalg.norm(d) + 1e-8
    if length < 0.5:
        return vol
    d_hat = d / length

    D, H, W = vol.shape
    # Bounding box for efficiency
    margin = radius + 3
    lo = np.maximum(np.minimum(p0, p1) - margin, 0).astype(int)
    hi = np.minimum(np.maximum(p0, p1) + margin, [D, H, W]).astype(int)
    if np.any(hi <= lo):
        return vol

    zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    vz, vy, vx = zz - p0[0], yy - p0[1], xx - p0[2]
    t = np.clip(vz*d_hat[0] + vy*d_hat[1] + vx*d_hat[2], 0, length)
    pz, py, px = p0[0]+t*d_hat[0], p0[1]+t*d_hat[1], p0[2]+t*d_hat[2]
    dist = np.sqrt((zz-pz)**2 + (yy-py)**2 + (xx-px)**2)
    mask = np.clip(1.0 - (dist - radius) / max(radius*0.15, 0.8), 0, 1)

    sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    vol[sl] = vol[sl] * (1 - alpha * mask) + value * alpha * mask
    return vol


def render_tree_segments(vol, segments, value, alpha=0.9):
    """Render all tube segments from branching_tree into volume."""
    for p0, p1, r in segments:
        vol = render_tube(vol, p0, p1, r, value, alpha)
    return np.clip(vol, 0, 1).astype(np.float32)


def reaction_diffusion_3d(shape, n_steps=300, rng=None):
    """Gray-Scott reaction-diffusion for Turing-pattern textures.

    Computationally intensive: runs n_steps PDE iterations on full 3D grid.
    """
    rng = rng or np.random.default_rng()
    D, H, W = shape
    # Use coarser grid for speed, then upsample
    scale = 2
    sD, sH, sW = D // scale, H // scale, W // scale
    U = np.ones((sD, sH, sW), dtype=np.float64)
    V = np.zeros((sD, sH, sW), dtype=np.float64)
    # Seed patches
    n_seeds = rng.integers(5, 15)
    for _ in range(n_seeds):
        cz = rng.integers(5, sD - 5)
        cy = rng.integers(5, sH - 5)
        cx = rng.integers(5, sW - 5)
        r = rng.integers(2, 6)
        V[max(cz-r,0):cz+r, max(cy-r,0):cy+r, max(cx-r,0):cx+r] = 1.0
        U[max(cz-r,0):cz+r, max(cy-r,0):cy+r, max(cx-r,0):cx+r] = 0.5

    Du, Dv = 0.16, 0.08
    f, k = 0.035, 0.065
    dt = 1.0

    for _ in range(n_steps):
        lu = gaussian_filter(U, sigma=1.0) - U  # Laplacian approx
        lv = gaussian_filter(V, sigma=1.0) - V
        uvv = U * V * V
        U += dt * (Du * lu - uvv + f * (1 - U))
        V += dt * (Dv * lv + uvv - (f + k) * V)
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    # Upsample to full resolution
    from scipy.ndimage import zoom
    V_full = zoom(V, scale, order=1).astype(np.float32)
    # Crop/pad to exact shape
    V_out = np.zeros(shape, dtype=np.float32)
    dz = min(V_full.shape[0], D)
    dy = min(V_full.shape[1], H)
    dx = min(V_full.shape[2], W)
    V_out[:dz, :dy, :dx] = V_full[:dz, :dy, :dx]
    return V_out


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Attenuation values (normalized HU to [0,1])
# ═══════════════════════════════════════════════════════════════════════════════
# Map:  air=0.0, fat=0.18, water=0.22, soft_tissue=0.25, muscle=0.28,
#        blood=0.26, cartilage=0.30, cancellous_bone=0.55,
#        cortical_bone=0.80, enamel=0.90, metal=1.0

MU = {
    "air": 0.0, "lung": 0.06, "fat": 0.18, "water": 0.22,
    "soft": 0.25, "blood": 0.26, "muscle": 0.28, "liver": 0.27,
    "kidney": 0.26, "cartilage": 0.30, "cancellous": 0.55,
    "cortical": 0.80, "dentin": 0.85, "enamel": 0.92, "metal": 1.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Dev recipes (10 types — anatomy-inspired, medium-to-high complexity)
#
# Each recipe is inspired by structural characteristics found in recent CBCT/CT
# benchmark datasets. The mapping:
#   head_cranial      → CQ500 (2018) head CT anatomy
#   torso_thorax      → AAPM Low-Dose CT (2017) / ICASSP 2024 3D CBCT Challenge
#   abdomen_organs    → CBCTLiTS (2024) liver/abdomen synthetic CBCT
#   extremity_bone    → LoDoPaB-CT (2021) long bone / extremity anatomy
#   dental_arch       → MMDental (2025) / CTooth+ dental CBCT
#   pelvis_hip        → AAPM Low-Dose CT (2017) pelvis anatomy
#   shoulder_complex  → AAPM Low-Dose CT (2017) shoulder anatomy
#   knee_joint        → LoDoPaB-CT (2021) joint anatomy
#   spine_segment     → AAPM Low-Dose CT (2017) spine anatomy
#   hand_wrist        → Helsinki Tomography Challenge (2022) small-object geometry
# ═══════════════════════════════════════════════════════════════════════════════

def _recipe_head_cranial(rng, shape):
    """Skull vault + brain with cortical folds + ventricles + sinuses.

    Inspired by CQ500 (2018) head CT anatomy — 491 non-contrast head CTs."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Skull outer surface (deformed ellipsoid)
    skull_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.36))
    skull_inner = sdf_ellipsoid(shape, c, (D*.38, H*.34, W*.32))
    skull_mask = sdf_to_mask(skull_sdf) * (1 - sdf_to_mask(skull_inner))
    vol = composite(vol, skull_mask, MU["cortical"], 0.95)

    # Brain matter — fBm texture for gray/white matter contrast
    brain_mask = sdf_to_mask(skull_inner, edge_width=2.0)
    brain_tex = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=1.5, rng=rng)
    # Cortical folds: high-frequency noise thresholded
    folds = fbm_noise_3d(shape, octaves=8, persistence=0.65, base_sigma=0.8, rng=rng)
    gray_matter = (folds > 0.45).astype(np.float32) * 0.04  # slight boost
    brain_val = MU["soft"] + 0.03 * brain_tex + gray_matter
    vol = composite(vol, brain_mask, brain_val, 0.85)

    # Ventricles (CSF-filled cavities)
    for offset in [(-D*0.02, 0, -W*0.06), (-D*0.02, 0, W*0.06)]:
        vc = (c[0]+offset[0], c[1]+offset[1], c[2]+offset[2])
        v_sdf = sdf_ellipsoid(shape, vc, (D*0.08, H*0.04, W*0.02))
        vol = composite(vol, sdf_to_mask(v_sdf), MU["water"], 0.9)
    # Third ventricle
    v3_sdf = sdf_ellipsoid(shape, c, (D*0.06, H*0.02, W*0.01))
    vol = composite(vol, sdf_to_mask(v3_sdf), MU["water"], 0.9)

    # Sinuses (air cavities in frontal bone)
    for sy in [-1, 1]:
        sc = (c[0]-D*0.15, c[1]-H*0.20, c[2]+sy*W*0.08)
        s_sdf = sdf_ellipsoid(shape, sc, (D*0.06, H*0.05, W*0.04))
        vol = composite(vol, sdf_to_mask(s_sdf), MU["air"], 0.95)

    # Eyes
    for sy in [-1, 1]:
        ec = (c[0]-D*0.05, c[1]-H*0.28, c[2]+sy*W*0.12)
        e_sdf = sdf_ellipsoid(shape, ec, (12, 12, 12))
        vol = composite(vol, sdf_to_mask(e_sdf), MU["water"], 0.85)

    # Scalp soft tissue
    scalp_sdf = sdf_ellipsoid(shape, c, (D*.45, H*.41, W*.39))
    scalp_mask = sdf_to_mask(scalp_sdf) * (1 - sdf_to_mask(skull_sdf))
    vol = composite(vol, scalp_mask, MU["soft"] - 0.02, 0.7)

    # Micro-texture overlay
    tex = fbm_noise_3d(shape, octaves=6, persistence=0.5, base_sigma=1.0, rng=rng)
    body_mask = sdf_to_mask(scalp_sdf)
    vol = vol + 0.02 * tex * body_mask

    vol = elastic_deform(vol, rng, strength=3.0, sigma=25.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_torso_thorax(rng, shape):
    """Ribs + spine + lung parenchyma + heart + major vessels.

    Inspired by AAPM Low-Dose CT (2017) / ICASSP 2024 3D CBCT Challenge."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Body outline
    body_sdf = sdf_ellipsoid(shape, c, (D*.44, H*.36, W*.42))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["fat"], 0.9)

    # Muscle/soft tissue inner
    inner_sdf = sdf_ellipsoid(shape, c, (D*.40, H*.32, W*.38))
    vol = composite(vol, sdf_to_mask(inner_sdf), MU["muscle"], 0.7)

    # Lungs (two air-filled regions with parenchymal texture)
    for side in [-1, 1]:
        lc = (c[0], c[1]-H*0.02, c[2]+side*W*0.14)
        lung_sdf = sdf_ellipsoid(shape, lc, (D*0.30, H*0.22, W*0.15))
        lung_mask = sdf_to_mask(lung_sdf, edge_width=2.0)
        # Parenchymal texture with vessels
        lung_tex = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=1.0, rng=rng)
        lung_val = MU["lung"] + 0.06 * lung_tex
        vol = composite(vol, lung_mask, lung_val, 0.9)

    # Heart (central, slightly left)
    heart_c = (c[0], c[1]-H*0.05, c[2]-W*0.04)
    heart_sdf = sdf_superquadric(shape, heart_c, (D*0.12, H*0.10, W*0.10), 0.8, 0.8)
    vol = composite(vol, sdf_to_mask(heart_sdf), MU["blood"], 0.85)
    # Heart wall
    heart_inner = sdf_superquadric(shape, heart_c, (D*0.09, H*0.07, W*0.07), 0.8, 0.8)
    wall_mask = sdf_to_mask(heart_sdf) * (1 - sdf_to_mask(heart_inner))
    vol = composite(vol, wall_mask, MU["muscle"], 0.9)

    # Spine (posterior cylinder stack)
    spine_c = (c[0], c[1]+H*0.22, c[2])
    for zi in range(8):
        zpos = D*0.15 + zi * D*0.085
        vert_sdf = sdf_superquadric(shape, (zpos, spine_c[1], spine_c[2]),
                                     (D*0.03, H*0.04, W*0.035), 0.6, 0.6)
        vol = composite(vol, sdf_to_mask(vert_sdf), MU["cortical"], 0.9)
        # Cancellous core
        core_sdf = sdf_ellipsoid(shape, (zpos, spine_c[1], spine_c[2]),
                                  (D*0.02, H*0.025, W*0.02))
        vol = composite(vol, sdf_to_mask(core_sdf), MU["cancellous"], 0.8)

    # Ribs (curved cylinders, 12 pairs approximated as tori)
    for i in range(6):
        zpos = D*0.20 + i * D*0.10
        for side in [-1, 1]:
            rib_c = (zpos, c[1]+H*0.05, c[2])
            R_major = W * 0.28
            r_minor = rng.uniform(3, 5)
            rib_sdf = sdf_torus(shape, rib_c, R_major, r_minor, axis=0)
            # Only keep posterior half
            zz, yy, xx = _coords(shape)
            half = (yy > c[1] - H*0.15).astype(np.float32)
            rib_mask = sdf_to_mask(rib_sdf, edge_width=1.0) * half
            vol = composite(vol, rib_mask, MU["cortical"], 0.85)

    # Aorta (major vessel, vertical tube)
    aorta_segs = branching_tree(
        shape, (D*0.2, c[1]+H*0.05, c[2]-W*0.02), (1, 0, 0),
        length=D*0.6, radius=6, depth=3, branch_angle=0.4, decay=0.65, rng=rng,
    )
    vol = render_tree_segments(vol, aorta_segs, MU["blood"], alpha=0.85)

    tex = fbm_noise_3d(shape, octaves=6, persistence=0.5, base_sigma=1.5, rng=rng)
    vol = vol + 0.015 * tex * sdf_to_mask(body_sdf)
    vol = elastic_deform(vol, rng, strength=2.5, sigma=30.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_abdomen_organs(rng, shape):
    """Liver + kidneys + spleen + bowel loops + vertebrae + fat layers.

    Inspired by CBCTLiTS (2024) — 201 synthetic paired CBCT/CT volumes."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.40))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["fat"], 0.85)
    inner_sdf = sdf_ellipsoid(shape, c, (D*.38, H*.34, W*.36))
    vol = composite(vol, sdf_to_mask(inner_sdf), MU["soft"], 0.75)

    # Liver (large right-side organ)
    liver_c = (c[0]-D*0.05, c[1]-H*0.05, c[2]+W*0.10)
    liver_sdf = sdf_superquadric(shape, liver_c, (D*0.14, H*0.12, W*0.16), 0.7, 0.7)
    liver_tex = fbm_noise_3d(shape, octaves=7, persistence=0.55, base_sigma=1.2, rng=rng)
    vol = composite(vol, sdf_to_mask(liver_sdf), MU["liver"] + 0.02*liver_tex, 0.85)

    # Kidneys (bilateral bean-shaped)
    for side in [-1, 1]:
        kc = (c[0]+D*0.05, c[1]+H*0.08, c[2]+side*W*0.15)
        k_sdf = sdf_ellipsoid(shape, kc, (D*0.06, H*0.04, W*0.035))
        vol = composite(vol, sdf_to_mask(k_sdf), MU["kidney"], 0.85)
        # Renal pelvis
        kp_sdf = sdf_ellipsoid(shape, kc, (D*0.03, H*0.02, W*0.015))
        vol = composite(vol, sdf_to_mask(kp_sdf), MU["water"], 0.7)

    # Spleen (left side)
    sp_c = (c[0]-D*0.03, c[1]-H*0.02, c[2]-W*0.18)
    sp_sdf = sdf_ellipsoid(shape, sp_c, (D*0.07, H*0.05, W*0.04))
    vol = composite(vol, sdf_to_mask(sp_sdf), MU["soft"]+0.02, 0.8)

    # Bowel loops (multiple small deformed tubes)
    for _ in range(rng.integers(8, 15)):
        bc = (c[0]+rng.uniform(-D*0.15, D*0.15),
              c[1]+rng.uniform(-H*0.10, H*0.10),
              c[2]+rng.uniform(-W*0.15, W*0.15))
        br = (rng.uniform(D*0.02, D*0.06), rng.uniform(H*0.02, H*0.04),
              rng.uniform(W*0.02, W*0.05))
        b_sdf = sdf_ellipsoid(shape, bc, br)
        # Gas/fluid inside
        vol = composite(vol, sdf_to_mask(b_sdf), rng.uniform(MU["air"], MU["water"]), 0.7)
        # Bowel wall
        b_outer = sdf_ellipsoid(shape, bc, tuple(r+2 for r in br))
        wall = sdf_to_mask(b_outer) * (1 - sdf_to_mask(b_sdf))
        vol = composite(vol, wall, MU["soft"], 0.75)

    # Spine
    for zi in range(5):
        zpos = D*0.25 + zi * D*0.10
        vert_sdf = sdf_ellipsoid(shape, (zpos, c[1]+H*0.25, c[2]),
                                  (D*0.03, H*0.04, W*0.035))
        vol = composite(vol, sdf_to_mask(vert_sdf), MU["cortical"], 0.9)

    # Mesenteric vessels
    segs = branching_tree(shape, (D*0.3, c[1], c[2]), (1,0,0),
                          length=D*0.4, radius=4, depth=4, decay=0.65, rng=rng)
    vol = render_tree_segments(vol, segs, MU["blood"], 0.8)

    tex = fbm_noise_3d(shape, octaves=6, persistence=0.5, base_sigma=2.0, rng=rng)
    vol = vol + 0.015 * tex * sdf_to_mask(body_sdf)
    vol = elastic_deform(vol, rng, strength=3.0, sigma=25.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_extremity_bone(rng, shape):
    """Long bone + cortex/marrow + muscle bundles + fat + periosteum.

    Inspired by LoDoPaB-CT (2021) extremity anatomy characteristics."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Limb outline
    limb_sdf = sdf_cylinder(shape, c, W*0.35, D*0.42, axis=0)
    vol = composite(vol, sdf_to_mask(limb_sdf), MU["fat"], 0.85)

    # Muscle compartments (4 bundles around bone, with fiber texture)
    for angle_idx in range(4):
        angle = angle_idx * np.pi / 2 + rng.uniform(-0.2, 0.2)
        mc = (c[0], c[1]+H*0.12*np.cos(angle), c[2]+W*0.12*np.sin(angle))
        m_sdf = sdf_cylinder(shape, mc, rng.uniform(15, 28), D*0.38, axis=0)
        # Anisotropic fiber texture along bone axis
        fiber = anisotropic_noise_3d(shape, direction=(1, 0, 0), stretch=5.0,
                                      octaves=6, rng=rng)
        m_val = MU["muscle"] + 0.03 * fiber
        vol = composite(vol, sdf_to_mask(m_sdf, edge_width=2.0), m_val, 0.75)

    # Cortical bone shaft
    bone_sdf = sdf_cylinder(shape, c, rng.uniform(12, 20), D*0.40, axis=0)
    cortex_tex = anisotropic_noise_3d(shape, (1, 0, 0), stretch=6.0, octaves=5, rng=rng)
    vol = composite(vol, sdf_to_mask(bone_sdf), MU["cortical"] + 0.02*cortex_tex, 0.95)

    # Marrow cavity
    marrow_r = rng.uniform(6, 12)
    marrow_sdf = sdf_cylinder(shape, c, marrow_r, D*0.35, axis=0)
    marrow_tex = fbm_noise_3d(shape, octaves=5, persistence=0.5, base_sigma=2.0, rng=rng)
    vol = composite(vol, sdf_to_mask(marrow_sdf), MU["fat"]+0.05*marrow_tex, 0.9)

    # Nutrient vessels in cortex
    n_canals = rng.integers(8, 20)
    for _ in range(n_canals):
        p0 = (rng.uniform(D*0.15, D*0.85), c[1]+rng.uniform(-8, 8), c[2]+rng.uniform(-8, 8))
        p1 = (p0[0]+rng.uniform(-20, 20), c[1]+rng.uniform(-15, 15), c[2]+rng.uniform(-15, 15))
        vol = render_tube(vol, p0, p1, rng.uniform(0.8, 2.0), MU["blood"], 0.8)

    # Subcutaneous fat layer texture
    fat_tex = worley_noise_3d(shape, n_pts=100, mode="F2-F1", rng=rng)
    limb_mask = sdf_to_mask(limb_sdf)
    vol = vol + 0.02 * fat_tex * limb_mask

    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_dental_arch(rng, shape):
    """Full dental arch with individual teeth + roots + mandible + soft tissue.

    Inspired by MMDental (2025, 660 patients) and CTooth+ dental CBCT."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Mandible (U-shaped bone)
    mand_sdf = sdf_torus(shape, (c[0], c[1]-H*0.05, c[2]), W*0.25, 18, axis=0)
    zz, yy, xx = _coords(shape)
    anterior = (yy < c[1]+H*0.05).astype(np.float32)
    vol = composite(vol, sdf_to_mask(mand_sdf)*anterior, MU["cortical"], 0.9)
    # Cancellous interior
    mand_inner = sdf_torus(shape, (c[0], c[1]-H*0.05, c[2]), W*0.25, 12, axis=0)
    vol = composite(vol, sdf_to_mask(mand_inner)*anterior, MU["cancellous"], 0.8)

    # Individual teeth along the arch (16 teeth per arch)
    n_teeth = 16
    for i in range(n_teeth):
        theta = np.pi * 0.1 + i * np.pi * 0.8 / (n_teeth - 1)
        tz = c[0] + rng.uniform(-2, 2)
        ty = c[1] - H*0.05 - 25 * np.cos(theta)
        tx = c[2] + W*0.25 * np.sin(theta - np.pi/2)

        # Crown (enamel outer)
        crown_h = rng.uniform(8, 14)
        crown_r = rng.uniform(4, 7)
        t_sdf = sdf_cylinder(shape, (tz-crown_h/2, ty, tx), crown_r, crown_h, axis=0)
        vol = composite(vol, sdf_to_mask(t_sdf), MU["enamel"], 0.95)
        # Dentin inner
        d_sdf = sdf_cylinder(shape, (tz-crown_h/2, ty, tx), crown_r*0.7, crown_h*0.9, axis=0)
        vol = composite(vol, sdf_to_mask(d_sdf), MU["dentin"], 0.9)
        # Pulp chamber
        p_sdf = sdf_cylinder(shape, (tz-crown_h/2, ty, tx), crown_r*0.25, crown_h*0.7, axis=0)
        vol = composite(vol, sdf_to_mask(p_sdf), MU["soft"], 0.85)
        # Root
        root_h = rng.uniform(12, 22)
        root_r = rng.uniform(2, 4)
        r_sdf = sdf_cylinder(shape, (tz+crown_h/2+root_h/2, ty, tx),
                              root_r, root_h, axis=0)
        vol = composite(vol, sdf_to_mask(r_sdf), MU["dentin"], 0.9)

    # Gingival soft tissue
    gum_sdf = sdf_torus(shape, (c[0], c[1]-H*0.05, c[2]), W*0.25, 25, axis=0)
    gum_mask = sdf_to_mask(gum_sdf)*anterior * (1 - sdf_to_mask(mand_sdf)*anterior)
    vol = composite(vol, gum_mask, MU["soft"], 0.6)

    # Tongue
    tongue_sdf = sdf_ellipsoid(shape, (c[0], c[1]-H*0.15, c[2]), (D*0.06, H*0.08, W*0.10))
    vol = composite(vol, sdf_to_mask(tongue_sdf), MU["muscle"], 0.8)

    # Inferior alveolar nerve canals
    for side in [-1, 1]:
        nc = (c[0]+D*0.08, c[1]+H*0.02, c[2]+side*W*0.12)
        ne = (c[0]+D*0.08, c[1]-H*0.22, c[2]+side*W*0.02)
        vol = render_tube(vol, nc, ne, 1.5, MU["soft"], 0.8)

    vol = elastic_deform(vol, rng, strength=1.5, sigma=18.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_pelvis_hip(rng, shape):
    """Hip bones + sacrum + femoral heads + muscle layers + bladder.

    Inspired by AAPM Low-Dose CT (2017) pelvis anatomy."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.40, H*.38, W*.42))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["fat"], 0.8)
    vol = composite(vol, sdf_to_mask(sdf_ellipsoid(shape, c, (D*.36, H*.34, W*.38))),
                    MU["muscle"], 0.7)

    # Iliac bones (two large curved plates)
    for side in [-1, 1]:
        ic = (c[0]-D*0.05, c[1]+H*0.05, c[2]+side*W*0.18)
        il_sdf = sdf_superquadric(shape, ic, (D*0.18, H*0.15, W*0.06), 0.5, 1.5)
        vol = composite(vol, sdf_to_mask(il_sdf), MU["cortical"], 0.9)
        il_inner = sdf_superquadric(shape, ic, (D*0.15, H*0.12, W*0.03), 0.5, 1.5)
        canc_tex = fbm_noise_3d(shape, octaves=6, persistence=0.6, base_sigma=1.0, rng=rng)
        vol = composite(vol, sdf_to_mask(il_inner), MU["cancellous"]+0.05*canc_tex, 0.8)

    # Sacrum
    sac_sdf = sdf_superquadric(shape, (c[0]+D*0.05, c[1]+H*0.22, c[2]),
                                 (D*0.10, H*0.06, W*0.08), 0.7, 0.7)
    vol = composite(vol, sdf_to_mask(sac_sdf), MU["cortical"], 0.9)

    # Femoral heads
    for side in [-1, 1]:
        fh_c = (c[0]+D*0.15, c[1]-H*0.02, c[2]+side*W*0.16)
        fh_sdf = sdf_ellipsoid(shape, fh_c, (18, 18, 18))
        vol = composite(vol, sdf_to_mask(fh_sdf), MU["cortical"], 0.9)
        fh_inner = sdf_ellipsoid(shape, fh_c, (13, 13, 13))
        trab = fbm_noise_3d(shape, octaves=7, persistence=0.6, base_sigma=0.8, rng=rng)
        vol = composite(vol, sdf_to_mask(fh_inner), MU["cancellous"]+0.08*trab, 0.85)

    # Bladder
    bl_sdf = sdf_ellipsoid(shape, (c[0]+D*0.08, c[1]-H*0.12, c[2]),
                             (D*0.07, H*0.06, W*0.06))
    vol = composite(vol, sdf_to_mask(bl_sdf), MU["water"], 0.85)

    # Iliac vessels
    for side in [-1, 1]:
        segs = branching_tree(shape, (D*0.15, c[1]+H*0.05, c[2]+side*W*0.08),
                              (1, -0.3, side*0.2), D*0.3, 4, depth=3, rng=rng)
        vol = render_tree_segments(vol, segs, MU["blood"], 0.8)

    vol = elastic_deform(vol, rng, strength=2.5, sigma=22.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_shoulder_complex(rng, shape):
    """Humerus head + scapula + clavicle + rotator cuff muscles + vessels.

    Inspired by AAPM Low-Dose CT (2017) upper extremity anatomy."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.40))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["fat"], 0.8)
    vol = composite(vol, sdf_to_mask(sdf_ellipsoid(shape, c, (D*.38, H*.34, W*.36))),
                    MU["muscle"], 0.65)

    # Humeral head
    hh_c = (c[0], c[1]-H*0.05, c[2]-W*0.22)
    hh_sdf = sdf_ellipsoid(shape, hh_c, (22, 22, 22))
    vol = composite(vol, sdf_to_mask(hh_sdf), MU["cortical"], 0.9)
    hh_inner = sdf_ellipsoid(shape, hh_c, (17, 17, 17))
    vol = composite(vol, sdf_to_mask(hh_inner), MU["cancellous"], 0.85)

    # Humeral shaft
    shaft_sdf = sdf_cylinder(shape, (c[0]+D*0.15, c[1]-H*0.05, c[2]-W*0.22),
                              10, D*0.25, axis=0)
    vol = composite(vol, sdf_to_mask(shaft_sdf), MU["cortical"], 0.9)

    # Glenoid (scapula socket)
    gl_c = (c[0], c[1]+H*0.02, c[2]-W*0.15)
    gl_sdf = sdf_superquadric(shape, gl_c, (D*0.08, H*0.12, W*0.03), 0.6, 1.2)
    vol = composite(vol, sdf_to_mask(gl_sdf), MU["cortical"], 0.9)

    # Scapula body
    sc_c = (c[0]+D*0.05, c[1]+H*0.15, c[2]-W*0.10)
    sc_sdf = sdf_superquadric(shape, sc_c, (D*0.12, H*0.14, W*0.02), 0.5, 1.5)
    vol = composite(vol, sdf_to_mask(sc_sdf), MU["cortical"], 0.85)

    # Clavicle
    vol = render_tube(vol, (c[0]-D*0.12, c[1]-H*0.18, c[2]-W*0.08),
                      (c[0]-D*0.12, c[1]-H*0.18, c[2]+W*0.15), 5, MU["cortical"], 0.9)

    # Rotator cuff muscles (4 muscles wrapping around joint)
    for k in range(4):
        angle = k * np.pi / 2 + rng.uniform(-0.3, 0.3)
        mc = (c[0]+rng.uniform(-10, 10),
              hh_c[1]+25*np.cos(angle),
              hh_c[2]+25*np.sin(angle))
        m_sdf = sdf_ellipsoid(shape, mc, (D*0.08, rng.uniform(12, 20), rng.uniform(8, 15)))
        fiber = anisotropic_noise_3d(shape, (0, np.cos(angle), np.sin(angle)),
                                      stretch=4.0, octaves=5, rng=rng)
        vol = composite(vol, sdf_to_mask(m_sdf), MU["muscle"]+0.02*fiber, 0.7)

    # Vessels
    segs = branching_tree(shape, (c[0]-D*0.2, c[1]-H*0.1, c[2]-W*0.05),
                          (0, -0.2, -1), W*0.25, 3.5, depth=4, rng=rng)
    vol = render_tree_segments(vol, segs, MU["blood"], 0.8)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_knee_joint(rng, shape):
    """Femur/tibia condyles + menisci + cartilage + ligaments + muscle.

    Inspired by LoDoPaB-CT (2021) joint anatomy characteristics."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    limb_sdf = sdf_cylinder(shape, c, W*0.32, D*0.44, axis=0)
    vol = composite(vol, sdf_to_mask(limb_sdf), MU["fat"], 0.8)

    # Muscle compartments
    for k in range(6):
        angle = k * np.pi / 3
        mc = (c[0], c[1]+H*0.14*np.cos(angle), c[2]+W*0.14*np.sin(angle))
        m_sdf = sdf_cylinder(shape, mc, rng.uniform(12, 22), D*0.38, axis=0)
        vol = composite(vol, sdf_to_mask(m_sdf), MU["muscle"], 0.7)

    # Femur (upper half)
    f_sdf = sdf_cylinder(shape, (c[0]-D*0.15, c[1], c[2]), 14, D*0.28, axis=0)
    vol = composite(vol, sdf_to_mask(f_sdf), MU["cortical"], 0.92)
    # Femoral condyles (two bulges at joint line)
    for side in [-1, 1]:
        fc = (c[0]+D*0.02, c[1]+H*0.01, c[2]+side*8)
        fc_sdf = sdf_ellipsoid(shape, fc, (15, 16, 14))
        vol = composite(vol, sdf_to_mask(fc_sdf), MU["cortical"], 0.9)
        # Subchondral cancellous
        fc_inner = sdf_ellipsoid(shape, fc, (11, 12, 10))
        trab = fbm_noise_3d(shape, octaves=7, persistence=0.6, base_sigma=0.8, rng=rng)
        vol = composite(vol, sdf_to_mask(fc_inner), MU["cancellous"]+0.06*trab, 0.85)

    # Tibia (lower half)
    t_sdf = sdf_cylinder(shape, (c[0]+D*0.15, c[1]+H*0.01, c[2]), 12, D*0.28, axis=0)
    vol = composite(vol, sdf_to_mask(t_sdf), MU["cortical"], 0.92)
    # Tibial plateau
    tp_sdf = sdf_ellipsoid(shape, (c[0]+D*0.02, c[1]+H*0.01, c[2]), (8, 20, 18))
    vol = composite(vol, sdf_to_mask(tp_sdf), MU["cortical"], 0.9)

    # Articular cartilage
    cart_sdf = sdf_ellipsoid(shape, (c[0]+D*0.01, c[1], c[2]), (5, 22, 20))
    vol = composite(vol, sdf_to_mask(cart_sdf), MU["cartilage"], 0.75)

    # Menisci (two C-shaped fibrocartilage pads)
    for side in [-1, 1]:
        men_c = (c[0]+D*0.01, c[1], c[2]+side*8)
        men_sdf = sdf_torus(shape, men_c, 10, 3, axis=0)
        zz = _coords(shape)[0]
        half = (zz > c[0]-3).astype(np.float32) * (zz < c[0]+3).astype(np.float32)
        vol = composite(vol, sdf_to_mask(men_sdf)*half, MU["cartilage"]+0.02, 0.8)

    # Patella
    pat_sdf = sdf_ellipsoid(shape, (c[0], c[1]-H*0.18, c[2]), (8, 10, 12))
    vol = composite(vol, sdf_to_mask(pat_sdf), MU["cortical"], 0.9)

    # Popliteal vessels
    segs = branching_tree(shape, (c[0]-D*0.2, c[1]+H*0.12, c[2]),
                          (1, 0, 0), D*0.4, 3, depth=3, rng=rng)
    vol = render_tree_segments(vol, segs, MU["blood"], 0.8)

    vol = elastic_deform(vol, rng, strength=1.5, sigma=18.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_spine_segment(rng, shape):
    """3-4 vertebral bodies + intervertebral discs + spinal canal + processes.

    Inspired by AAPM Low-Dose CT (2017) spine anatomy."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.44, H*.38, W*.36))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["fat"], 0.7)
    vol = composite(vol, sdf_to_mask(sdf_ellipsoid(shape, c, (D*.40, H*.34, W*.32))),
                    MU["muscle"], 0.65)

    n_vert = rng.integers(3, 5)
    spacing = D * 0.7 / n_vert
    for i in range(n_vert):
        zpos = D*0.15 + i * spacing

        # Vertebral body (anterior)
        vb_c = (zpos, c[1]-H*0.02, c[2])
        vb_sdf = sdf_superquadric(shape, vb_c, (spacing*0.35, H*0.08, W*0.09), 0.6, 0.6)
        vol = composite(vol, sdf_to_mask(vb_sdf), MU["cortical"], 0.9)
        # Cancellous core with trabecular texture
        vb_inner = sdf_superquadric(shape, vb_c, (spacing*0.28, H*0.06, W*0.07), 0.6, 0.6)
        trab = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=0.6, rng=rng)
        vol = composite(vol, sdf_to_mask(vb_inner), MU["cancellous"]+0.08*trab, 0.85)

        # Pedicles + laminae (posterior arch)
        for side in [-1, 1]:
            ped_c = (zpos, c[1]+H*0.06, c[2]+side*W*0.05)
            ped_sdf = sdf_cylinder(shape, ped_c, 5, H*0.08, axis=1)
            vol = composite(vol, sdf_to_mask(ped_sdf), MU["cortical"], 0.9)

        # Spinous process
        sp_c = (zpos, c[1]+H*0.16, c[2])
        sp_sdf = sdf_ellipsoid(shape, sp_c, (spacing*0.15, H*0.06, W*0.02))
        vol = composite(vol, sdf_to_mask(sp_sdf), MU["cortical"], 0.88)

        # Transverse processes
        for side in [-1, 1]:
            tp_c = (zpos, c[1]+H*0.08, c[2]+side*W*0.12)
            tp_sdf = sdf_ellipsoid(shape, tp_c, (spacing*0.10, H*0.02, W*0.05))
            vol = composite(vol, sdf_to_mask(tp_sdf), MU["cortical"], 0.88)

        # Intervertebral disc (between vertebrae)
        if i < n_vert - 1:
            disc_z = zpos + spacing * 0.5
            disc_sdf = sdf_cylinder(shape, (disc_z, c[1]-H*0.02, c[2]),
                                     W*0.08, spacing*0.15, axis=0)
            vol = composite(vol, sdf_to_mask(disc_sdf), MU["cartilage"], 0.8)

    # Spinal canal (CSF + cord)
    canal_sdf = sdf_cylinder(shape, c, 6, D*0.6, axis=0)
    vol = composite(vol, sdf_to_mask(canal_sdf), MU["water"], 0.85)
    cord_sdf = sdf_cylinder(shape, c, 3, D*0.6, axis=0)
    vol = composite(vol, sdf_to_mask(cord_sdf), MU["soft"], 0.9)

    # Paraspinal muscles with fiber texture
    for side in [-1, 1]:
        pm_c = (c[0], c[1]+H*0.10, c[2]+side*W*0.08)
        pm_sdf = sdf_cylinder(shape, pm_c, 18, D*0.6, axis=0)
        fiber = anisotropic_noise_3d(shape, (1, 0, 0), stretch=5.0, octaves=5, rng=rng)
        vol = composite(vol, sdf_to_mask(pm_sdf), MU["muscle"]+0.02*fiber, 0.65)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_hand_wrist(rng, shape):
    """Carpal bones + metacarpals + phalanges + tendons + joint spaces.

    Inspired by Helsinki Tomography Challenge (2022) small-object geometry."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Hand outline (flat ellipsoid)
    hand_sdf = sdf_ellipsoid(shape, c, (D*0.10, H*0.35, W*0.40))
    vol = composite(vol, sdf_to_mask(hand_sdf), MU["soft"], 0.8)

    # 8 carpal bones (small irregular bones at wrist)
    carpal_positions = [
        (-D*0.02, -H*0.08, -W*0.10), (-D*0.02, -H*0.08, -W*0.03),
        (-D*0.02, -H*0.08, W*0.03),  (-D*0.02, -H*0.08, W*0.10),
        (D*0.02, -H*0.04, -W*0.08),  (D*0.02, -H*0.04, -W*0.01),
        (D*0.02, -H*0.04, W*0.05),   (D*0.02, -H*0.04, W*0.10),
    ]
    for dx, dy, dz in carpal_positions:
        bc = (c[0]+dx, c[1]+dy, c[2]+dz)
        r = rng.uniform(5, 9)
        b_sdf = sdf_superquadric(shape, bc, (r, r*0.9, r*0.8),
                                   rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0))
        vol = composite(vol, sdf_to_mask(b_sdf), MU["cortical"], 0.9)
        b_inner = sdf_ellipsoid(shape, bc, (r*0.6, r*0.5, r*0.5))
        vol = composite(vol, sdf_to_mask(b_inner), MU["cancellous"], 0.8)

    # 5 metacarpals + phalanges (finger rays)
    for finger in range(5):
        fx = c[2] - W*0.16 + finger * W*0.08
        # Metacarpal
        mc_start = (c[0], c[1]-H*0.02, fx)
        mc_end = (c[0], c[1]+H*0.12, fx)
        vol = render_tube(vol, mc_start, mc_end, rng.uniform(3, 5), MU["cortical"], 0.9)
        # Proximal phalanx
        pp_end = (c[0], c[1]+H*0.22, fx)
        vol = render_tube(vol, mc_end, pp_end, rng.uniform(2.5, 4), MU["cortical"], 0.9)
        # Distal phalanx (skip for thumb simplicity)
        if finger != 0:
            dp_end = (c[0], c[1]+H*0.28, fx)
            vol = render_tube(vol, pp_end, dp_end, rng.uniform(2, 3.5), MU["cortical"], 0.88)

    # Tendons (multiple thin tubes along each finger)
    for finger in range(5):
        fx = c[2] - W*0.16 + finger * W*0.08
        for offset in [-2, 2]:
            t_start = (c[0]+offset, c[1]-H*0.15, fx)
            t_end = (c[0]+offset, c[1]+H*0.25, fx)
            vol = render_tube(vol, t_start, t_end, 1.2, MU["soft"]+0.03, 0.7)

    # Ulna and radius (forearm bones at wrist)
    vol = render_tube(vol, (c[0]-3, c[1]-H*0.35, c[2]-W*0.06),
                      (c[0]-3, c[1]-H*0.05, c[2]-W*0.06), 6, MU["cortical"], 0.9)
    vol = render_tube(vol, (c[0]+3, c[1]-H*0.35, c[2]+W*0.06),
                      (c[0]+3, c[1]-H*0.05, c[2]+W*0.06), 5, MU["cortical"], 0.9)

    tex = fbm_noise_3d(shape, octaves=5, persistence=0.5, base_sigma=1.5, rng=rng)
    vol = vol + 0.01 * tex * sdf_to_mask(hand_sdf)
    vol = elastic_deform(vol, rng, strength=1.0, sigma=15.0)
    return np.clip(vol, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — Hidden recipes (10 types — adversarial stress-tests)
#
# Each recipe takes the most challenging features from recent CBCT datasets
# and amplifies them to create extreme adversarial phantoms:
#   trabecular_micro    → Walnut CT (CWI, 2019) / PCCT (2025) micro-structure
#   multi_metal         → AAPM Low-Dose CT + orthopedic metal implants
#   vascular_tree       → CBCTLiTS (2024) portal vasculature, extreme branching
#   lung_parenchyma     → LIDC-IDRI / ICASSP 2024 3D CBCT Challenge
#   fractal_membrane    → CBCTLiTS (2024) organ boundaries, thin membranes
#   gyroid_scaffold     → 2DeteCT (2023) / DM4CT (2025) industrial/metamaterial
#   dental_metal        → MMDental (2025) / CTooth+ extreme metal artifacts
#   cardiac_chambers    → AAPM Low-Dose CT cardiac anatomy
#   multi_contrast      → 2DeteCT (2023) multi-beam mode dynamic range
#   reaction_diffusion  → DM4CT (2025) complex micro-structure patterns
# ═══════════════════════════════════════════════════════════════════════════════

def _recipe_trabecular_micro(rng, shape):
    """Fine trabecular bone: gyroid + Worley + multi-scale porosity.

    Inspired by Walnut CT (CWI, 2019) / PCCT (2025) shell micro-structure."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.zeros(shape, dtype=np.float32)

    # Outer cortical shell with texture
    outer_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.38))
    inner_sdf = sdf_ellipsoid(shape, c, (D*.37, H*.33, W*.33))
    cortex_mask = sdf_to_mask(outer_sdf) * (1 - sdf_to_mask(inner_sdf))
    cortex_tex = anisotropic_noise_3d(shape, (0, 1, 0), stretch=5.0, octaves=6, rng=rng)
    vol = composite(vol, cortex_mask, MU["cortical"] + 0.03*cortex_tex, 0.95)

    interior = sdf_to_mask(inner_sdf)

    # Layer 1: Gyroid trabecular scaffold
    period = rng.uniform(12, 25)
    thickness = rng.uniform(1.2, 2.5)
    g1 = gyroid_surface(shape, period=period, thickness=thickness)
    vol = np.maximum(vol, g1 * MU["cancellous"] * interior * 0.9)

    # Layer 2: Schwarz-P at different scale
    p1 = schwarz_p_surface(shape, period=period*1.7, thickness=thickness*0.8)
    vol = np.maximum(vol, p1 * (MU["cancellous"]+0.05) * interior * 0.85)

    # Layer 3: Worley noise for irregular porosity
    w = worley_noise_3d(shape, n_pts=150, mode="F2-F1", rng=rng)
    trabecular_mask = (w > 0.15).astype(np.float32)
    vol = np.maximum(vol, trabecular_mask * MU["cancellous"] * interior * 0.8)

    # Layer 4: Ultra-fine fBm micro-texture
    micro = fbm_noise_3d(shape, octaves=8, persistence=0.65, base_sigma=0.5, rng=rng)
    fine_trab = (micro > rng.uniform(0.4, 0.55)).astype(np.float32)
    vol = np.maximum(vol, fine_trab * 0.5 * interior * 0.7)

    # Marrow fat between trabeculae
    marrow = fbm_noise_3d(shape, octaves=5, persistence=0.5, base_sigma=3.0, rng=rng)
    marrow_mask = interior * (vol < 0.3).astype(np.float32)
    vol = np.maximum(vol, MU["fat"] * marrow * marrow_mask * 0.6)

    # Haversian-like canals in cortex
    n_canals = rng.integers(15, 35)
    for _ in range(n_canals):
        p0 = (rng.uniform(D*0.1, D*0.9), rng.uniform(H*0.1, H*0.9), rng.uniform(W*0.1, W*0.9))
        angle = rng.uniform(0, 2*np.pi)
        length = rng.uniform(15, 50)
        p1 = (p0[0]+length*rng.uniform(-0.3, 0.3),
              p0[1]+length*np.cos(angle), p0[2]+length*np.sin(angle))
        vol = render_tube(vol, p0, p1, rng.uniform(0.5, 1.5), MU["blood"], 0.7)

    vol = elastic_deform(vol, rng, strength=1.5, sigma=15.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_multi_metal(rng, shape):
    """Multiple metal implant types + bone + tissue — extreme dynamic range.

    Inspired by AAPM Low-Dose CT + orthopedic implant scenarios."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.40))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["soft"], 0.85)

    # Multiple bone structures
    for _ in range(rng.integers(3, 7)):
        bc = tuple(c[i] + rng.uniform(-0.2, 0.2)*shape[i] for i in range(3))
        br = tuple(rng.uniform(10, 35) for _ in range(3))
        b_sdf = sdf_ellipsoid(shape, bc, br)
        vol = composite(vol, sdf_to_mask(b_sdf), MU["cortical"], 0.9)

    # Titanium hip prosthesis (stem + head + cup)
    stem_c = (c[0]+D*0.1, c[1]-H*0.05, c[2]+W*0.15)
    stem_sdf = sdf_cylinder(shape, stem_c, 6, D*0.3, axis=0)
    vol = composite(vol, sdf_to_mask(stem_sdf), MU["metal"], 1.0)
    head_sdf = sdf_ellipsoid(shape, (stem_c[0]-D*0.15, stem_c[1], stem_c[2]), (14, 14, 14))
    vol = composite(vol, sdf_to_mask(head_sdf), MU["metal"], 1.0)

    # Screws (multiple thin high-contrast cylinders)
    n_screws = rng.integers(3, 8)
    for _ in range(n_screws):
        sc = tuple(c[i] + rng.uniform(-0.25, 0.25)*shape[i] for i in range(3))
        axis = int(rng.integers(0, 3))
        length = rng.uniform(20, 60)
        r = rng.uniform(1.5, 4)
        s_sdf = sdf_cylinder(shape, sc, r, length, axis)
        vol = composite(vol, sdf_to_mask(s_sdf, edge_width=0.8), MU["metal"], 1.0)

    # Metal plate
    plate_c = tuple(c[i] + rng.uniform(-0.15, 0.15)*shape[i] for i in range(3))
    plate_sdf = sdf_superquadric(shape, plate_c,
                                   (rng.uniform(2, 4), rng.uniform(15, 40), rng.uniform(15, 40)),
                                   0.3, 0.3)
    vol = composite(vol, sdf_to_mask(plate_sdf), MU["metal"], 1.0)

    # Wire cerclage (thin ring)
    wire_c = tuple(c[i] + rng.uniform(-0.1, 0.1)*shape[i] for i in range(3))
    wire_sdf = sdf_torus(shape, wire_c, rng.uniform(15, 30), rng.uniform(0.8, 2.0),
                          axis=int(rng.integers(0, 3)))
    vol = composite(vol, sdf_to_mask(wire_sdf, edge_width=0.8), MU["metal"], 1.0)

    # Dental amalgam filling (small but very bright)
    for _ in range(rng.integers(1, 4)):
        fc = tuple(c[i] + rng.uniform(-0.3, 0.3)*shape[i] for i in range(3))
        f_sdf = sdf_ellipsoid(shape, fc, (rng.uniform(2, 5), rng.uniform(2, 5), rng.uniform(2, 5)))
        vol = composite(vol, sdf_to_mask(f_sdf), MU["metal"], 1.0)

    # Tissue textures
    tex = fbm_noise_3d(shape, octaves=6, persistence=0.55, base_sigma=2.0, rng=rng)
    vol = vol + 0.02 * tex * sdf_to_mask(body_sdf)
    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_vascular_tree(rng, shape):
    """Full 3D vascular tree with Murray's law branching + contrast agent.

    Inspired by CBCTLiTS (2024) portal vasculature — depth-6 branching."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.44, H*.40, W*.42))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["soft"], 0.85)

    # Organs as background
    for _ in range(rng.integers(4, 8)):
        oc = tuple(c[i] + rng.uniform(-0.2, 0.2)*shape[i] for i in range(3))
        orr = tuple(rng.uniform(D*0.05, D*0.15) for _ in range(3))
        o_sdf = sdf_ellipsoid(shape, oc, orr)
        vol = composite(vol, sdf_to_mask(o_sdf), rng.uniform(MU["liver"], MU["muscle"]), 0.7)

    # Main arterial tree (deep branching)
    aorta_segs = branching_tree(
        shape, (D*0.1, c[1], c[2]), (1, 0, 0),
        length=D*0.7, radius=7, depth=6, branch_angle=0.45, decay=0.68, rng=rng,
    )
    # Contrast agent → higher attenuation than normal blood
    contrast_val = MU["blood"] + 0.25  # iodine contrast
    vol = render_tree_segments(vol, aorta_segs, contrast_val, 0.9)

    # Venous tree (separate, slightly different trajectory)
    vein_segs = branching_tree(
        shape, (D*0.15, c[1]+H*0.08, c[2]+W*0.05), (1, -0.1, 0.1),
        length=D*0.6, radius=8, depth=5, branch_angle=0.5, decay=0.7, rng=rng,
    )
    vol = render_tree_segments(vol, vein_segs, MU["blood"]+0.15, 0.85)

    # Portal venous system
    portal_segs = branching_tree(
        shape, (c[0], c[1]-H*0.05, c[2]+W*0.1), (0.2, -0.5, 0.3),
        length=D*0.25, radius=5, depth=5, branch_angle=0.55, decay=0.65, rng=rng,
    )
    vol = render_tree_segments(vol, portal_segs, MU["blood"]+0.18, 0.85)

    # Capillary bed (very fine vessels via noise)
    capillary = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=0.6, rng=rng)
    cap_mask = (capillary > 0.6).astype(np.float32) * sdf_to_mask(body_sdf)
    vol = np.maximum(vol, cap_mask * (MU["blood"]+0.10) * 0.5)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=22.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_lung_parenchyma(rng, shape):
    """Bronchial tree + alveolar texture + nodules + ground-glass opacity.

    Inspired by LIDC-IDRI / ICASSP 2024 3D CBCT Challenge — 1010 lung CTs."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Chest wall
    chest_sdf = sdf_ellipsoid(shape, c, (D*.44, H*.38, W*.44))
    chest_inner = sdf_ellipsoid(shape, c, (D*.40, H*.34, W*.40))
    wall_mask = sdf_to_mask(chest_sdf) * (1 - sdf_to_mask(chest_inner))
    vol = composite(vol, wall_mask, MU["muscle"], 0.85)

    lung_mask = sdf_to_mask(chest_inner)

    # Lung parenchyma base (air + very fine texture)
    parenchyma = fbm_noise_3d(shape, octaves=8, persistence=0.65, base_sigma=0.6, rng=rng)
    vol = composite(vol, lung_mask, MU["lung"] + 0.04*parenchyma, 0.9)

    # Bronchial tree
    for side in [-1, 1]:
        root = (c[0]-D*0.15, c[1]-H*0.15, c[2]+side*W*0.02)
        direction = (0.8, 0.3, side*0.4)
        bronchi = branching_tree(shape, root, direction, length=D*0.25, radius=5,
                                  depth=6, branch_angle=0.5, decay=0.65, rng=rng)
        vol = render_tree_segments(vol, bronchi, MU["air"]+0.02, 0.8)
        # Bronchial walls
        for p0, p1, r in bronchi:
            vol = render_tube(vol, p0, p1, r+1.5, MU["soft"], 0.3)

    # Pulmonary vessels
    for side in [-1, 1]:
        root = (c[0]-D*0.12, c[1]-H*0.08, c[2]+side*W*0.04)
        pv_segs = branching_tree(shape, root, (0.7, 0.4, side*0.3),
                                  length=D*0.2, radius=4, depth=5, decay=0.6, rng=rng)
        vol = render_tree_segments(vol, pv_segs, MU["blood"], 0.75)

    # Alveolar-level Worley texture (very fine cellular)
    alveolar = worley_noise_3d(shape, n_pts=300, mode="F1", rng=rng)
    vol = vol + 0.03 * alveolar * lung_mask

    # Nodules (small round densities)
    n_nodules = rng.integers(3, 10)
    for _ in range(n_nodules):
        nc = tuple(c[i] + rng.uniform(-0.25, 0.25)*shape[i] for i in range(3))
        nr = rng.uniform(3, 10)
        n_sdf = sdf_ellipsoid(shape, nc, (nr, nr, nr))
        vol = composite(vol, sdf_to_mask(n_sdf)*lung_mask, MU["soft"], 0.9)

    # Ground-glass opacity region
    gg_c = tuple(c[i] + rng.uniform(-0.15, 0.15)*shape[i] for i in range(3))
    gg_sdf = sdf_ellipsoid(shape, gg_c, (D*0.08, H*0.06, W*0.07))
    gg_tex = fbm_noise_3d(shape, octaves=6, persistence=0.55, base_sigma=1.5, rng=rng)
    vol = composite(vol, sdf_to_mask(gg_sdf, edge_width=5.0)*lung_mask,
                    MU["lung"]+0.08+0.04*gg_tex, 0.6)

    # Mediastinum
    med_sdf = sdf_ellipsoid(shape, (c[0], c[1]-H*0.05, c[2]), (D*0.20, H*0.10, W*0.08))
    vol = composite(vol, sdf_to_mask(med_sdf), MU["soft"], 0.8)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_fractal_membrane(rng, shape):
    """Fractal-boundary organs with ultra-thin curved membranes.

    Inspired by CBCTLiTS (2024) organ boundary complexity."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    body_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.38, W*.40))
    vol = composite(vol, sdf_to_mask(body_sdf), MU["soft"]*0.8, 0.7)

    # Multiple organs with fractal (noise-deformed) boundaries
    n_organs = rng.integers(6, 12)
    for _ in range(n_organs):
        oc = tuple(c[i] + rng.uniform(-0.25, 0.25)*shape[i] for i in range(3))
        orr = tuple(rng.uniform(D*0.05, D*0.18) for _ in range(3))
        o_sdf = sdf_ellipsoid(shape, oc, orr)

        # Fractal boundary perturbation
        boundary_noise = fbm_noise_3d(shape, octaves=8, persistence=0.7,
                                       base_sigma=0.8, rng=rng)
        perturbed_sdf = o_sdf + (boundary_noise - 0.5) * min(orr) * 0.3
        organ_mask = sdf_to_mask(perturbed_sdf, edge_width=1.0)

        val = rng.uniform(MU["water"], MU["muscle"])
        vol = composite(vol, organ_mask, val, 0.8)

        # Thin membrane (capsule) around organ
        expanded_sdf = perturbed_sdf - 1.5
        membrane = sdf_to_mask(expanded_sdf) * (1 - organ_mask)
        vol = composite(vol, membrane, MU["soft"]+0.05, 0.85)

    # Peritoneal folds (thin sheets)
    n_folds = rng.integers(5, 12)
    zz, yy, xx = _coords(shape)
    for _ in range(n_folds):
        angle_y = rng.uniform(0, np.pi)
        angle_x = rng.uniform(0, np.pi)
        offset = rng.uniform(-0.2, 0.2) * max(shape)
        coord = (np.cos(angle_y)*(yy-c[1]) + np.sin(angle_y)*np.cos(angle_x)*(xx-c[2]) +
                 np.sin(angle_y)*np.sin(angle_x)*(zz-c[0]) + offset)
        fold_noise = fbm_noise_3d(shape, octaves=6, persistence=0.6,
                                    base_sigma=1.5, rng=rng)
        coord += (fold_noise - 0.5) * 8
        membrane = np.exp(-coord**2 / 2.0).astype(np.float32)
        membrane *= sdf_to_mask(body_sdf)
        vol = composite(vol, membrane, MU["soft"]+0.03, 0.5)

    # Ascitic fluid (gravitational layering)
    fluid_level = c[0] + rng.uniform(D*0.05, D*0.20)
    fluid_mask = (zz > fluid_level).astype(np.float32) * sdf_to_mask(body_sdf)
    fluid_mask *= (vol < MU["soft"]).astype(np.float32)
    vol = composite(vol, fluid_mask, MU["water"], 0.5)

    vol = elastic_deform(vol, rng, strength=3.0, sigma=18.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_gyroid_scaffold(rng, shape):
    """Gyroid + Schwarz-P metamaterial scaffolds (additive manufacturing test).

    Inspired by 2DeteCT (2023) industrial parts / DM4CT (2025) micro-structure."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.zeros(shape, dtype=np.float32)

    # Multi-scale TPMS structures
    for scale_idx in range(3):
        period = rng.uniform(10, 35) * (1.5 ** scale_idx)
        thickness = rng.uniform(0.8, 2.5)
        if rng.random() < 0.5:
            layer = gyroid_surface(shape, period, thickness)
        else:
            layer = schwarz_p_surface(shape, period, thickness)
        val = rng.uniform(MU["cancellous"], MU["cortical"])
        vol = np.maximum(vol, layer * val)

    # Confine to an outer shape
    outer_sdf = sdf_superquadric(shape, c, (D*.40, H*.38, W*.38),
                                   rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5))
    outer_mask = sdf_to_mask(outer_sdf)
    vol *= outer_mask

    # Dense core region
    core_sdf = sdf_ellipsoid(shape, c, (D*0.12, H*0.12, W*0.12))
    vol = composite(vol, sdf_to_mask(core_sdf), MU["cortical"], 0.85)

    # Fine noise overlay
    tex = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=0.8, rng=rng)
    vol = vol + 0.04 * tex * outer_mask

    # Cellular inclusions via Worley
    cells = worley_noise_3d(shape, n_pts=120, mode="F2-F1", rng=rng)
    vol = vol + 0.06 * cells * outer_mask

    vol = elastic_deform(vol, rng, strength=1.0, sigma=15.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_dental_metal(rng, shape):
    """Full dental arch with metal crowns + amalgam fillings + root canals.

    Inspired by MMDental (2025) / CTooth+ extreme metal artifact scenarios."""
    D, H, W = shape
    c = (D/2, H/2, W/2)

    # Start with dental_arch base and add metals
    vol = _recipe_dental_arch(rng, shape)

    # Add metal crowns on random teeth
    n_crowns = rng.integers(2, 6)
    for i in range(n_crowns):
        theta = np.pi * 0.15 + rng.uniform(0, 1) * np.pi * 0.7
        tz = c[0] + rng.uniform(-3, 3)
        ty = c[1] - H*0.05 - 25 * np.cos(theta)
        tx = c[2] + W*0.25 * np.sin(theta - np.pi/2)
        crown_r = rng.uniform(5, 8)
        crown_h = rng.uniform(8, 12)
        cr_sdf = sdf_cylinder(shape, (tz-crown_h/2, ty, tx), crown_r, crown_h, axis=0)
        vol = composite(vol, sdf_to_mask(cr_sdf), MU["metal"], 0.95)

    # Amalgam fillings
    n_fillings = rng.integers(2, 5)
    for _ in range(n_fillings):
        theta = np.pi * 0.15 + rng.uniform(0, 1) * np.pi * 0.7
        tz = c[0] - rng.uniform(2, 8)
        ty = c[1] - H*0.05 - 25 * np.cos(theta)
        tx = c[2] + W*0.25 * np.sin(theta - np.pi/2)
        f_sdf = sdf_ellipsoid(shape, (tz, ty, tx),
                                (rng.uniform(2, 5), rng.uniform(2, 5), rng.uniform(2, 5)))
        vol = composite(vol, sdf_to_mask(f_sdf), MU["metal"], 1.0)

    # Orthodontic wire (thin metal arc)
    wire_sdf = sdf_torus(shape, (c[0]-D*0.04, c[1]-H*0.05, c[2]),
                          W*0.24, 0.8, axis=0)
    zz = _coords(shape)[0]
    anterior = (zz < c[0]+5).astype(np.float32)
    vol = composite(vol, sdf_to_mask(wire_sdf, edge_width=0.5)*anterior, MU["metal"], 0.95)

    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_cardiac_chambers(rng, shape):
    """Heart chambers + wall motion + coronary vessels + valves.

    Inspired by AAPM Low-Dose CT (2017) cardiac anatomy."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Pericardium + chest wall
    chest_sdf = sdf_ellipsoid(shape, c, (D*.44, H*.40, W*.44))
    vol = composite(vol, sdf_to_mask(chest_sdf), MU["fat"], 0.7)
    vol = composite(vol, sdf_to_mask(sdf_ellipsoid(shape, c, (D*.40, H*.36, W*.40))),
                    MU["muscle"], 0.6)

    # Left ventricle (thick-walled chamber, slightly left and inferior)
    lv_c = (c[0]+D*0.05, c[1]-H*0.02, c[2]-W*0.04)
    lv_outer = sdf_superquadric(shape, lv_c, (D*0.14, H*0.10, W*0.10), 0.8, 0.8)
    lv_inner = sdf_superquadric(shape, lv_c, (D*0.10, H*0.06, W*0.06), 0.9, 0.9)
    lv_wall = sdf_to_mask(lv_outer) * (1 - sdf_to_mask(lv_inner))
    muscle_tex = anisotropic_noise_3d(shape, (1, 0.5, 0), stretch=4.0, octaves=6, rng=rng)
    vol = composite(vol, lv_wall, MU["muscle"]+0.03*muscle_tex, 0.9)
    vol = composite(vol, sdf_to_mask(lv_inner), MU["blood"]+0.15, 0.9)  # contrast

    # Right ventricle (thinner wall, wraps around LV)
    rv_c = (c[0]+D*0.03, c[1]-H*0.02, c[2]+W*0.06)
    rv_outer = sdf_superquadric(shape, rv_c, (D*0.12, H*0.10, W*0.08), 0.7, 0.9)
    rv_inner = sdf_superquadric(shape, rv_c, (D*0.10, H*0.08, W*0.06), 0.8, 0.9)
    rv_wall = sdf_to_mask(rv_outer) * (1 - sdf_to_mask(rv_inner))
    vol = composite(vol, rv_wall, MU["muscle"]+0.01, 0.85)
    vol = composite(vol, sdf_to_mask(rv_inner), MU["blood"]+0.12, 0.85)

    # Atria (thin-walled upper chambers)
    for side, offset in [(-1, -W*0.05), (1, W*0.05)]:
        ac = (c[0]-D*0.10, c[1]-H*0.03, c[2]+offset)
        a_outer = sdf_ellipsoid(shape, ac, (D*0.08, H*0.08, W*0.06))
        a_inner = sdf_ellipsoid(shape, ac, (D*0.06, H*0.06, W*0.04))
        a_wall = sdf_to_mask(a_outer) * (1 - sdf_to_mask(a_inner))
        vol = composite(vol, a_wall, MU["muscle"], 0.8)
        vol = composite(vol, sdf_to_mask(a_inner), MU["blood"]+0.10, 0.8)

    # Interventricular septum texture
    sept_tex = fbm_noise_3d(shape, octaves=7, persistence=0.6, base_sigma=0.8, rng=rng)
    vol = vol + 0.015 * sept_tex * sdf_to_mask(lv_outer)

    # Coronary arteries (surface vessels)
    for k in range(3):
        angle = k * 2 * np.pi / 3
        root = (lv_c[0]-D*0.08, lv_c[1]+30*np.cos(angle), lv_c[2]+30*np.sin(angle))
        direction = (0.6, np.cos(angle+0.5), np.sin(angle+0.5))
        segs = branching_tree(shape, root, direction, length=D*0.15, radius=2.5,
                               depth=4, branch_angle=0.5, decay=0.65, rng=rng)
        vol = render_tree_segments(vol, segs, MU["blood"]+0.20, 0.85)

    # Aortic root
    vol = render_tube(vol, (c[0]-D*0.15, c[1]-H*0.08, c[2]-W*0.02),
                      (c[0]-D*0.30, c[1]-H*0.08, c[2]-W*0.02), 8, MU["blood"]+0.18, 0.9)

    # Valve calcifications (small bright spots)
    n_calc = rng.integers(2, 6)
    for _ in range(n_calc):
        vc = (c[0]-D*0.08+rng.uniform(-8, 8), c[1]+rng.uniform(-8, 8), c[2]+rng.uniform(-8, 8))
        v_sdf = sdf_ellipsoid(shape, vc, (rng.uniform(1, 3), rng.uniform(1, 3), rng.uniform(1, 3)))
        vol = composite(vol, sdf_to_mask(v_sdf), MU["cortical"], 0.9)

    vol = elastic_deform(vol, rng, strength=2.5, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_multi_contrast(rng, shape):
    """Extreme dynamic range: air + fat + tissue + bone + metal in one phantom.

    Inspired by 2DeteCT (2023) multi-beam mode extreme contrast range."""
    D, H, W = shape
    c = (D/2, H/2, W/2)
    vol = np.full(shape, MU["air"], dtype=np.float32)

    # Nested regions of every material class
    outer_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.40, W*.40))
    vol = composite(vol, sdf_to_mask(outer_sdf), MU["fat"], 0.9)

    # Fat lobules
    fat_worley = worley_noise_3d(shape, n_pts=80, mode="F1", rng=rng)
    fat_mask = sdf_to_mask(outer_sdf) * (fat_worley < 0.3).astype(np.float32)
    vol = composite(vol, fat_mask, MU["fat"]+0.02, 0.6)

    # Muscle compartments with fiber texture
    for k in range(6):
        angle = k * np.pi / 3 + rng.uniform(-0.2, 0.2)
        mc = (c[0], c[1]+H*0.15*np.cos(angle), c[2]+W*0.15*np.sin(angle))
        m_sdf = sdf_cylinder(shape, mc, rng.uniform(15, 25), D*0.35, axis=0)
        fiber = anisotropic_noise_3d(shape, (1, 0, 0), stretch=5.0, octaves=6, rng=rng)
        vol = composite(vol, sdf_to_mask(m_sdf), MU["muscle"]+0.02*fiber, 0.7)

    # Central bone with cortex + cancellous + marrow
    bone_sdf = sdf_cylinder(shape, c, 18, D*0.35, axis=0)
    cortex_tex = anisotropic_noise_3d(shape, (1, 0, 0), stretch=6.0, octaves=5, rng=rng)
    vol = composite(vol, sdf_to_mask(bone_sdf), MU["cortical"]+0.02*cortex_tex, 0.95)
    marrow_sdf = sdf_cylinder(shape, c, 10, D*0.30, axis=0)
    trab = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=0.6, rng=rng)
    trab_mask = (trab > 0.45).astype(np.float32)
    marrow_region = sdf_to_mask(marrow_sdf)
    vol = composite(vol, marrow_region * trab_mask, MU["cancellous"], 0.85)
    vol = composite(vol, marrow_region * (1-trab_mask), MU["fat"]+0.03, 0.7)

    # Air cavities
    for _ in range(rng.integers(2, 5)):
        ac = tuple(c[i] + rng.uniform(-0.2, 0.2)*shape[i] for i in range(3))
        ar = tuple(rng.uniform(5, 18) for _ in range(3))
        a_sdf = sdf_ellipsoid(shape, ac, ar)
        vol = composite(vol, sdf_to_mask(a_sdf), MU["air"], 0.95)

    # Water/fluid
    for _ in range(rng.integers(1, 4)):
        wc = tuple(c[i] + rng.uniform(-0.2, 0.2)*shape[i] for i in range(3))
        wr = tuple(rng.uniform(8, 20) for _ in range(3))
        w_sdf = sdf_ellipsoid(shape, wc, wr)
        vol = composite(vol, sdf_to_mask(w_sdf), MU["water"], 0.85)

    # Metal inserts
    for _ in range(rng.integers(2, 5)):
        mc_pos = tuple(c[i] + rng.uniform(-0.25, 0.25)*shape[i] for i in range(3))
        mr = rng.uniform(2, 6)
        m_sdf = sdf_ellipsoid(shape, mc_pos, (mr, mr, mr))
        vol = composite(vol, sdf_to_mask(m_sdf), MU["metal"], 1.0)

    # Vessels
    segs = branching_tree(shape, (D*0.1, c[1], c[2]), (1, 0, 0),
                          D*0.5, 5, depth=4, rng=rng)
    vol = render_tree_segments(vol, segs, MU["blood"]+0.15, 0.8)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=20.0)
    return np.clip(vol, 0, 1).astype(np.float32)


def _recipe_reaction_diffusion(rng, shape):
    """Turing pattern 3D structures — adversarial aliasing + organic complexity.

    Inspired by DM4CT (2025) complex rock micro-structure patterns."""
    D, H, W = shape
    c = (D/2, H/2, W/2)

    # Reaction-diffusion pattern (computationally expensive)
    rd = reaction_diffusion_3d(shape, n_steps=400, rng=rng)

    # Normalize and scale to bone-like attenuation
    rd_mask = (rd > 0.3).astype(np.float32)
    vol = rd_mask * rng.uniform(MU["cancellous"], MU["cortical"])

    # Add a second RD pattern at different scale for multi-scale complexity
    rd2 = reaction_diffusion_3d(shape, n_steps=250, rng=rng)
    rd2_mask = (rd2 > 0.35).astype(np.float32)
    vol = np.maximum(vol, rd2_mask * rng.uniform(0.35, 0.55))

    # Confine to outer shape
    outer_sdf = sdf_ellipsoid(shape, c, (D*.42, H*.40, W*.40))
    vol *= sdf_to_mask(outer_sdf)

    # High-contrast periodic overlay
    zz, yy, xx = _coords(shape)
    freq = rng.integers(8, 20)
    grid = 0.5 + 0.5 * np.sin(2*np.pi*xx/(W/freq)) * np.cos(2*np.pi*yy/(H/freq))
    vol = vol + 0.08 * grid.astype(np.float32) * sdf_to_mask(outer_sdf)

    # Worley cell boundaries
    cells = worley_noise_3d(shape, n_pts=100, mode="F2-F1", rng=rng)
    vol = vol + 0.06 * cells * sdf_to_mask(outer_sdf)

    # fBm texture
    tex = fbm_noise_3d(shape, octaves=8, persistence=0.6, base_sigma=1.0, rng=rng)
    vol = vol + 0.04 * tex * sdf_to_mask(outer_sdf)

    # Metal inclusions for extreme dynamic range
    for _ in range(rng.integers(1, 4)):
        mc = tuple(c[i] + rng.uniform(-0.2, 0.2)*shape[i] for i in range(3))
        mr = rng.uniform(2, 5)
        m_sdf = sdf_ellipsoid(shape, mc, (mr, mr, mr))
        vol = composite(vol, sdf_to_mask(m_sdf), MU["metal"], 1.0)

    vol = elastic_deform(vol, rng, strength=2.0, sigma=18.0)
    return np.clip(vol, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — Recipe dispatch & public API
# ═══════════════════════════════════════════════════════════════════════════════

RECIPE_NAMES = [
    # Dev (0-9)
    "head_cranial", "torso_thorax", "abdomen_organs", "extremity_bone",
    "dental_arch", "pelvis_hip", "shoulder_complex", "knee_joint",
    "spine_segment", "hand_wrist",
    # Hidden (10-19)
    "trabecular_micro", "multi_metal", "vascular_tree", "lung_parenchyma",
    "fractal_membrane", "gyroid_scaffold", "dental_metal", "cardiac_chambers",
    "multi_contrast", "reaction_diffusion",
]

_DEV_DISPATCH = [
    _recipe_head_cranial, _recipe_torso_thorax, _recipe_abdomen_organs,
    _recipe_extremity_bone, _recipe_dental_arch, _recipe_pelvis_hip,
    _recipe_shoulder_complex, _recipe_knee_joint, _recipe_spine_segment,
    _recipe_hand_wrist,
]

_HIDDEN_DISPATCH = [
    _recipe_trabecular_micro, _recipe_multi_metal, _recipe_vascular_tree,
    _recipe_lung_parenchyma, _recipe_fractal_membrane, _recipe_gyroid_scaffold,
    _recipe_dental_metal, _recipe_cardiac_chambers, _recipe_multi_contrast,
    _recipe_reaction_diffusion,
]

# Explicit per-sample recipe assignments (20 samples each)
DEV_RECIPE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
HIDDEN_RECIPE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# N_views per sample
DEV_NVIEWS = [256, 256, 256, 512, 256, 256, 512, 256, 512, 256,
              512, 512, 128, 128, 512, 128, 256, 128, 256, 512]
HIDDEN_NVIEWS = [128, 128, 128, 128, 256, 128, 256, 128, 256, 128,
                 256, 256, 512, 256, 512, 256, 128, 512, 128, 256]


def generate_cbct_phantom(
    seed: int,
    mode: str = "dev",
    shape: tuple[int, int, int] = (256, 256, 256),
) -> tuple[np.ndarray, str]:
    """Generate a procedural CBCT phantom volume.

    Args:
        seed: Random seed for reproducibility.
        mode: "dev" (anatomy-inspired) or "hidden" (adversarial stress-test).
        shape: Output volume shape (D, H, W).

    Returns:
        (mu, recipe_name) where mu is float32 in [0, 1].
    """
    if mode not in ("dev", "hidden"):
        raise ValueError(f"mode must be 'dev' or 'hidden', got {mode!r}")

    rng = np.random.default_rng(seed)
    dispatch = _DEV_DISPATCH if mode == "dev" else _HIDDEN_DISPATCH
    recipe_ids = DEV_RECIPE_IDS if mode == "dev" else HIDDEN_RECIPE_IDS

    # Pick recipe based on seed index
    idx = seed % len(recipe_ids)
    recipe_id = recipe_ids[idx]
    func = dispatch[recipe_id]
    recipe_name = func.__name__.replace("_recipe_", "")

    mu = func(rng, shape)
    return np.clip(mu, 0, 1).astype(np.float32), recipe_name


def generate_batch(
    n: int,
    mode: str = "dev",
    base_seed: int = 8000,
    shape: tuple[int, int, int] = (256, 256, 256),
) -> list[tuple[str, np.ndarray, str]]:
    """Generate a batch of procedural phantoms."""
    phantoms = []
    for i in range(n):
        seed = base_seed + i
        mu, recipe = generate_cbct_phantom(seed, mode=mode, shape=shape)
        name = f"proc_{mode}_{i:02d}"
        phantoms.append((name, mu, recipe))
    return phantoms


# ═══════════════════════════════════════════════════════════════════════════════
# CLI demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    demo_shape = (64, 64, 64)
    print(f"Generating {n} {mode} phantoms (shape={demo_shape})...")

    for name, mu, recipe in generate_batch(n, mode=mode, shape=demo_shape):
        print(f"  {name}: shape={mu.shape} range=[{mu.min():.3f}, {mu.max():.3f}] recipe={recipe}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        phantoms = generate_batch(n, mode=mode, shape=demo_shape)
        fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
        if n == 1:
            axes = axes[:, np.newaxis]
        for col, (name, mu, recipe) in enumerate(phantoms):
            D, H, W = mu.shape
            axes[0, col].imshow(mu[D//2], cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"{name}\n{recipe}\naxial", fontsize=8)
            axes[0, col].axis("off")
            axes[1, col].imshow(mu[:, H//2, :], cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_title("coronal", fontsize=8)
            axes[1, col].axis("off")
            axes[2, col].imshow(mu[:, :, W//2], cmap="gray", vmin=0, vmax=1)
            axes[2, col].set_title("sagittal", fontsize=8)
            axes[2, col].axis("off")
        fig.tight_layout()
        fig.savefig("_phantom_preview.png", dpi=100)
        print(f"\n  Preview saved to _phantom_preview.png")
    except Exception as e:
        print(f"  (No preview: {e})")
