"""Procedural brain axial T2w MRI phantom generator for PWM MRI benchmark.

Generates synthetic 2D T2-weighted brain MRI slices (320×320, float32 [0,1])
matching the anatomy of real multi-coil axial T2 brain acquisitions.

T2w signal intensity reference (normalised):
  CSF / fluid          ~0.92  (very bright — long T2)
  Scalp fat            ~0.82  (bright — short T1)
  Gray matter (cortex) ~0.64  (intermediate — folded ribbon)
  Basal ganglia / thal ~0.55  (slightly above WM)
  White matter         ~0.40  (darker — long T1, moderate T2)
  Cortical bone        ~0.03  (very dark)
  Background / air      0.00

Layering strategy
-----------------
Uses alpha compositing (painter's algorithm):
  img = img*(1-alpha) + signal*alpha     [_lerp helper]

Tissue radii (normalised elliptical distance from brain centre):
  1.000 + SCALP_T  → outer scalp surface
  1.000            → outer skull (R_SKULL_OUTER)
  1.000 - SKULL_T  → inner skull / outer SAS  (R_SKULL_INNER)
  …   - SAS_T      → inner SAS / outer cortex  (R_SAS_INNER)
  …   - CORTEX_T   → inner cortex / outer WM   (R_CORTEX_INNER)

The gyral folding field G(θ) ∈ N(0,1) modulates R_SAS_INNER angularly:
  R_CORTEX_OUTER_EFF(θ) = R_SAS_INNER + G(θ) * GYRAL_AMP
Gyri  → R_CORTEX_OUTER_EFF > R_SAS_INNER (cortex protrudes into SAS)
Sulci → R_CORTEX_OUTER_EFF < R_SAS_INNER (CSF fills the retracted gap)

Recipes
-------
Dev (mild mismatch):
  brain_t2_normal      55%  Standard mid-brain: cortex, WM, lateral ventricles
  brain_t2_csf_rich    30%  Enlarged ventricles (hydrocephalus-like)
  brain_t2_posterior   15%  Posterior fossa: cerebellum + brainstem

Hidden (adversarial, severe mismatch):
  brain_t2_wm_lesions    35%  Focal WM hyperintensities (MS-like plaques)
  brain_t2_atrophy       30%  Cortical atrophy — widened sulci and ventricles
  brain_t2_high_contrast 20%  Extreme CSF / WM intensity ratio
  brain_t2_fine_gyri     15%  Very fine cortical folding (resolution stress)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (320, 320)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_mri_gt(seed: int, mode: str = "dev", shape: tuple = SHAPE) -> tuple:
    """Return (x: float32 (H,W) in [0,1], recipe_name: str)."""
    rng = np.random.default_rng(seed)
    if mode == "dev":
        recipes = [("brain_t2_normal",    0.55),
                   ("brain_t2_csf_rich",  0.30),
                   ("brain_t2_posterior", 0.15)]
    else:
        recipes = [("brain_t2_wm_lesions",    0.35),
                   ("brain_t2_atrophy",       0.30),
                   ("brain_t2_high_contrast", 0.20),
                   ("brain_t2_fine_gyri",     0.15)]
    names, probs = zip(*recipes)
    recipe = str(rng.choice(names, p=list(probs)))
    return _build_scene(recipe, shape, rng).astype(np.float32), recipe


# ── Scene dispatcher ──────────────────────────────────────────────────────────

def _build_scene(recipe, shape, rng):
    if recipe == "brain_t2_normal":
        return _brain_t2(shape, rng)
    if recipe == "brain_t2_csf_rich":
        return _brain_t2(shape, rng, enlarged_ventricles=True)
    if recipe == "brain_t2_posterior":
        return _brain_t2_posterior(shape, rng)
    if recipe == "brain_t2_wm_lesions":
        return _brain_t2(shape, rng, wm_lesions=True)
    if recipe == "brain_t2_atrophy":
        return _brain_t2(shape, rng, atrophy=True, enlarged_ventricles=True)
    if recipe == "brain_t2_high_contrast":
        return _brain_t2(shape, rng, high_contrast=True)
    if recipe == "brain_t2_fine_gyri":
        return _brain_t2(shape, rng, fine_gyri=True)
    return _brain_t2(shape, rng)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _ellipse(H, W, cy, cx, ry, rx, angle_deg=0.0):
    """Boolean ellipse mask (H, W)."""
    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    if angle_deg:
        ang = np.radians(angle_deg)
        yr =  yy * np.cos(ang) + xx * np.sin(ang)
        xr = -yy * np.sin(ang) + xx * np.cos(ang)
    else:
        yr, xr = yy, xx
    return ((yr / ry) ** 2 + (xr / rx) ** 2) <= 1.0


def _soft(mask, sigma=1.0):
    """Gaussian-smooth a binary mask → soft alpha [0, 1]."""
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def _lerp(img, signal, alpha):
    """Alpha-composite: replace img with signal proportional to alpha."""
    return img * (1.0 - alpha) + float(signal) * alpha


def _gyral_field(theta, rng, n_min=5, n_max=13):
    """Random angular sinusoidal field, normalised to unit std."""
    field = np.zeros_like(theta, dtype=np.float64)
    for n in range(n_min, n_max + 1):
        field += rng.uniform(0.5, 1.5) * np.sin(n * theta + rng.uniform(0, 2 * np.pi))
    return (field / (float(np.std(field)) + 1e-6)).astype(np.float32)


# ── Main brain T2 axial phantom ───────────────────────────────────────────────

def _brain_t2(shape, rng, *, enlarged_ventricles=False, atrophy=False,
               wm_lesions=False, high_contrast=False, fine_gyri=False):
    """Axial T2w brain phantom — layered alpha compositing.

    Radial structure (normalised elliptical distance):
      background → scalp → calvarium → SAS (CSF) → cortex (gyral) → WM
      then subcortical structures and ventricles overlaid.
    """
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)  # background = air (0)

    # ── Geometry ──────────────────────────────────────────────────────────────
    cy = H / 2.0 + rng.uniform(-8.0,  8.0)
    cx = W / 2.0 + rng.uniform(-6.0,  6.0)
    skull_ry = H * rng.uniform(0.420, 0.448) * rng.uniform(0.97, 1.03)
    skull_rx = W * rng.uniform(0.385, 0.415) * rng.uniform(0.97, 1.03)

    yy    = np.arange(H, dtype=np.float64)[:, None] - cy
    xx    = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    # ── Tissue signals ────────────────────────────────────────────────────────
    SCALP_SIG = float(rng.uniform(0.76, 0.84))
    SKULL_SIG = float(rng.uniform(0.02, 0.05))
    WM_SIG    = float(rng.uniform(0.22, 0.30) if high_contrast else rng.uniform(0.35, 0.45))
    GM_SIG    = float(rng.uniform(0.58, 0.70))
    CSF_SIG   = float(rng.uniform(0.90, 0.96) if high_contrast else rng.uniform(0.88, 0.94))
    BG_SIG    = float(rng.uniform(0.50, 0.62))

    # ── Structural radii (normalised) ─────────────────────────────────────────
    SCALP_T   = 0.060   # scalp fat thickness
    SKULL_T   = 0.048   # calvarium bone thickness  (≈ 7 px at skull_ry=141)
    SAS_T     = 0.028   # subarachnoid space CSF   (≈ 4 px)
    CORTEX_T  = 0.096   # gray matter cortex       (≈ 13 px — clearly visible)
    GYRAL_AMP = 0.025   # gyral perturbation ±     (≈ ±3.5 px)

    R_SKULL_OUTER  = 1.000
    R_SKULL_INNER  = R_SKULL_OUTER  - SKULL_T          # ≈ 0.952
    R_SAS_INNER    = R_SKULL_INNER  - SAS_T             # ≈ 0.924
    R_CORTEX_INNER = R_SAS_INNER    - CORTEX_T          # ≈ 0.828

    # Atrophy: shrink brain contents (wider sulci, bigger ventricles handled separately)
    if atrophy:
        R_SAS_INNER    -= 0.022
        R_CORTEX_INNER -= 0.022

    # ── Gyral folding field ───────────────────────────────────────────────────
    n_min = 7 if fine_gyri else 5
    n_max = 17 if fine_gyri else 13
    gyral = _gyral_field(theta, rng, n_min=n_min, n_max=n_max)

    # Perturbed outer cortex boundary (varies angularly)
    R_CORTEX_OUTER_EFF = (R_SAS_INNER + gyral * GYRAL_AMP).astype(np.float32)

    # ── Layer 1: scalp fat ────────────────────────────────────────────────────
    img = _lerp(img, SCALP_SIG,
                _soft((r_norm < R_SKULL_OUTER + SCALP_T).astype(np.float32), 1.5))

    # ── Layer 2: calvarium (dark bone ring) ───────────────────────────────────
    skull_ring = (r_norm < R_SKULL_OUTER) & (r_norm >= R_SKULL_INNER)
    img = _lerp(img, SKULL_SIG, _soft(skull_ring.astype(np.float32), 1.2))

    # ── Layer 3: subarachnoid space (base CSF just inside skull) ─────────────
    # Paint the full SAS + a little extra so gyral perturbation is always covered
    sas_wide = (r_norm < R_SKULL_INNER) & (r_norm >= R_SAS_INNER - GYRAL_AMP * 3.0)
    img = _lerp(img, CSF_SIG, _soft(sas_wide.astype(np.float32), 1.2))

    # ── Layer 4: white matter (fills brain interior up to cortex) ────────────
    wm_mask = r_norm < R_CORTEX_INNER
    img = _lerp(img, WM_SIG, _soft(wm_mask.astype(np.float32), 2.0))

    # ── Layer 5: cortex (GM ribbon with gyral outer boundary) ────────────────
    cortex_mask = (r_norm < R_CORTEX_OUTER_EFF) & (r_norm >= R_CORTEX_INNER)
    img = _lerp(img, GM_SIG, _soft(cortex_mask.astype(np.float32), 1.0))

    # ── Layer 6: sulcal CSF (where cortex retracted, SAS fills deeper) ────────
    sulcal_mask = (r_norm >= R_CORTEX_OUTER_EFF) & (r_norm < R_SKULL_INNER)
    img = _lerp(img, CSF_SIG, _soft(sulcal_mask.astype(np.float32), 1.0))

    # ── Layer 7: lateral ventricles ───────────────────────────────────────────
    v_scale = float(rng.uniform(1.3, 1.9) if enlarged_ventricles else rng.uniform(0.75, 1.20))
    vent_y  = cy + H * rng.uniform(-0.05,  0.01)
    vent_dx = W  * rng.uniform( 0.088, 0.130)

    for side in [-1, 1]:
        vx  = cx + side * vent_dx
        fh  = _ellipse(H, W, vent_y - H * 0.050, vx - side * W * 0.016,
                       H * 0.065 * v_scale, W * 0.044 * v_scale, angle_deg=side * 14)
        bdy = _ellipse(H, W, vent_y + H * 0.010, vx,
                       H * 0.055 * v_scale, W * 0.055 * v_scale)
        oh  = _ellipse(H, W, vent_y + H * 0.062, vx + side * W * 0.014,
                       H * 0.058 * v_scale, W * 0.040 * v_scale, angle_deg=-side * 12)
        vent = (fh | bdy | oh).astype(np.float32)
        img = _lerp(img, CSF_SIG, _soft(vent, 2.0))

    # Third ventricle (midline slit)
    third = _ellipse(H, W, vent_y + H * 0.030, cx, H * 0.055, W * 0.013)
    img = _lerp(img, CSF_SIG, _soft(third.astype(np.float32), 1.5))

    # Cerebral aqueduct (present on ~40% of mid-brain slices)
    if rng.random() < 0.40:
        aq = _ellipse(H, W, vent_y + H * 0.075, cx, H * 0.016, W * 0.013)
        img = _lerp(img, CSF_SIG, _soft(aq.astype(np.float32), 1.2))

    # ── Layer 8: basal ganglia / thalami ──────────────────────────────────────
    bg_y  = vent_y + H * rng.uniform(0.01, 0.03)
    bg_dx = W      * rng.uniform(0.070, 0.100)

    for side in [-1, 1]:
        bgx  = cx + side * bg_dx
        bg   = _ellipse(H, W, bg_y, bgx, H * 0.068, W * 0.055)
        thal = _ellipse(H, W, bg_y + H * 0.028, cx + side * W * 0.044,
                        H * 0.058, W * 0.052)
        img = _lerp(img, BG_SIG,        _soft(bg.astype(np.float32),   2.5))
        img = _lerp(img, BG_SIG - 0.06, _soft(thal.astype(np.float32), 2.5))
        # Internal capsule (thin WM lane)
        ic = _ellipse(H, W, bg_y + H * 0.012, cx + side * W * 0.062,
                      H * 0.048, W * 0.012)
        img = _lerp(img, WM_SIG * 0.88, _soft(ic.astype(np.float32), 1.5))

    # ── Layer 9: corpus callosum (slightly darker WM) ─────────────────────────
    cc_y = vent_y - H * 0.008
    cc_g = _ellipse(H, W, cc_y - H * 0.028, cx, H * 0.028, W * 0.090)
    cc_b = _ellipse(H, W, cc_y + H * 0.005, cx, H * 0.016, W * 0.155)
    img  = _lerp(img, WM_SIG * 0.88, _soft((cc_g | cc_b).astype(np.float32), 1.5))

    # ── Layer 10: focal WM lesions (hidden tier) ──────────────────────────────
    if wm_lesions:
        for _ in range(int(rng.integers(3, 9))):
            ldir  = float(rng.uniform(0, 2 * np.pi))
            ldist = H * float(rng.uniform(0.06, 0.22))
            ly    = vent_y + ldist * np.cos(ldir)
            lx    = cx     + ldist * np.sin(ldir)
            lry   = H * float(rng.uniform(0.011, 0.028))
            lrx   = W * float(rng.uniform(0.011, 0.028)) * float(rng.uniform(0.6, 1.8))
            lesion = _ellipse(H, W, ly, lx, lry, lrx,
                              angle_deg=float(rng.uniform(0, 180)))
            img = _lerp(img, float(rng.uniform(0.55, 0.76)),
                        _soft(lesion.astype(np.float32), 1.2))

    # ── MRI-realistic field effects ───────────────────────────────────────────
    brain_mask = (r_norm < R_SKULL_INNER).astype(np.float32)

    # B1+ inhomogeneity (3T: brighter centre)
    b1_sig = float(rng.uniform(0.50, 0.70))
    b1_amp = float(rng.uniform(0.04, 0.10))
    b1 = 1.0 + b1_amp * np.exp(-((yy / (H * b1_sig)) ** 2
                                  + (xx / (W * b1_sig)) ** 2)).astype(np.float32)
    img *= b1

    # Receive coil roll-off (mild linear gradient)
    ga   = float(rng.uniform(0, 2 * np.pi))
    gamp = float(rng.uniform(0.02, 0.06))
    ramp = 1.0 + gamp * (np.cos(ga) * yy / H + np.sin(ga) * xx / W).astype(np.float32)
    img *= ramp.clip(0.85, 1.15)

    # Rician-like noise
    sn = float(rng.uniform(0.010, 0.022))
    n1 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    n2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    img += sn * (np.sqrt(n1 ** 2 + n2 ** 2) - float(np.sqrt(np.pi / 2))) * brain_mask

    return img.clip(0.0, 1.0)


# ── Posterior fossa phantom ───────────────────────────────────────────────────

def _brain_t2_posterior(shape, rng):
    """Posterior fossa axial slice: bilateral cerebellum + brainstem.

    Uses the same layered strategy as _brain_t2 but with different anatomy:
    - Smaller oval cross-section (posterior fossa is narrower)
    - Bilateral cerebellum with fine transverse foliation (folia)
    - Central brainstem with 4th ventricle posteriorly
    - Large cisterns with CSF signal
    """
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    cy = H / 2.0 + rng.uniform(-8.0, 8.0)
    cx = W / 2.0 + rng.uniform(-6.0, 6.0)
    skull_ry = H * rng.uniform(0.355, 0.398) * rng.uniform(0.96, 1.04)
    skull_rx = W * rng.uniform(0.345, 0.388) * rng.uniform(0.96, 1.04)

    yy    = np.arange(H, dtype=np.float64)[:, None] - cy
    xx    = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    SCALP_SIG = float(rng.uniform(0.76, 0.84))
    SKULL_SIG = float(rng.uniform(0.02, 0.05))
    WM_SIG    = float(rng.uniform(0.36, 0.46))
    GM_SIG    = float(rng.uniform(0.58, 0.70))
    CSF_SIG   = float(rng.uniform(0.88, 0.95))

    # Skull + scalp
    img = _lerp(img, SCALP_SIG, _soft((r_norm < 1.060).astype(np.float32), 1.5))
    img = _lerp(img, SKULL_SIG,
                _soft(((r_norm < 1.000) & (r_norm >= 0.952)).astype(np.float32), 1.2))
    # SAS and WM background
    img = _lerp(img, CSF_SIG,  _soft((r_norm < 0.952).astype(np.float32), 1.2))
    img = _lerp(img, WM_SIG,   _soft((r_norm < 0.820).astype(np.float32), 2.0))

    # Cerebellar cortex with fine foliation (using the same gyral-modulation approach)
    CEREBELLAR_CORTEX_T  = 0.090
    CEREBELLAR_GYRAL_AMP = 0.022
    cer_gyral = _gyral_field(theta, rng, n_min=9, n_max=20)
    cer_cortex_inner = 0.820 - CEREBELLAR_CORTEX_T   # = 0.730
    R_CER_CORTEX_OUTER = 0.820 + cer_gyral * CEREBELLAR_GYRAL_AMP
    cer_cortex = (r_norm < R_CER_CORTEX_OUTER) & (r_norm >= cer_cortex_inner)
    cer_sulcus = (r_norm >= R_CER_CORTEX_OUTER) & (r_norm < 0.952)
    img = _lerp(img, GM_SIG,  _soft(cer_cortex.astype(np.float32), 1.0))
    img = _lerp(img, CSF_SIG, _soft(cer_sulcus.astype(np.float32), 1.0))

    # ── Brainstem (central oval) ──────────────────────────────────────────────
    bs_cy = cy + H * float(rng.uniform(-0.02, 0.04))
    bs_cx = cx + W * float(rng.uniform(-0.02, 0.02))
    bs_ry = H * float(rng.uniform(0.105, 0.130))
    bs_rx = W * float(rng.uniform(0.090, 0.115))

    bs = _ellipse(H, W, bs_cy, bs_cx, bs_ry, bs_rx)
    img = _lerp(img, WM_SIG, _soft(bs.astype(np.float32), 2.5))

    # 4th ventricle (tent-shaped CSF cavity posterior to brainstem)
    v4 = _ellipse(H, W, bs_cy + bs_ry * 0.80, bs_cx,
                  H * float(rng.uniform(0.035, 0.055)),
                  W * float(rng.uniform(0.055, 0.080)))
    img = _lerp(img, CSF_SIG, _soft(v4.astype(np.float32), 1.8))

    # Cerebral aqueduct
    aq = _ellipse(H, W, bs_cy - bs_ry * 0.60, bs_cx, H * 0.015, W * 0.013)
    img = _lerp(img, CSF_SIG, _soft(aq.astype(np.float32), 1.2))

    # Basilar artery flow void (dark dot anterior to brainstem)
    ba = _ellipse(H, W, bs_cy - bs_ry * 0.90, bs_cx, H * 0.012, W * 0.012)
    img = _lerp(img, 0.01, _soft(ba.astype(np.float32), 1.0))

    # ── Prepontine cistern + CPA cisterns (large CSF) ─────────────────────────
    cist = _ellipse(H, W, bs_cy - bs_ry * 1.30, bs_cx,
                    H * 0.040, W * float(rng.uniform(0.16, 0.24)))
    img = _lerp(img, CSF_SIG, _soft(cist.astype(np.float32), 3.0))

    for side in [-1, 1]:
        cpa_cx = cx + side * W * float(rng.uniform(0.13, 0.18))
        cpa    = _ellipse(H, W, bs_cy + H * 0.02, cpa_cx,
                          H * float(rng.uniform(0.03, 0.05)),
                          W * float(rng.uniform(0.04, 0.07)))
        img = _lerp(img, CSF_SIG, _soft(cpa.astype(np.float32), 2.0))

    # ── Field effects + noise ─────────────────────────────────────────────────
    b1 = 1.0 + float(rng.uniform(0.03, 0.08)) * np.exp(
        -((yy / (H * 0.60)) ** 2 + (xx / (W * 0.60)) ** 2)).astype(np.float32)
    img *= b1

    brain_mask = (r_norm < 0.95).astype(np.float32)
    sn = float(rng.uniform(0.010, 0.020))
    n1 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    n2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    img += sn * (np.sqrt(n1 ** 2 + n2 ** 2) - float(np.sqrt(np.pi / 2))) * brain_mask

    return img.clip(0.0, 1.0)
