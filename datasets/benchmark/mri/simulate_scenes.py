"""Procedural brain axial T2w MRI phantom generator for PWM MRI benchmark.

Generates synthetic 2D T2-weighted brain MRI slices (320x320, float32 [0,1])
matching the anatomy of real multi-coil axial T2 brain acquisitions.

T2w signal intensity reference (normalised, TE~80ms):
  CSF / fluid          ~0.92  (very bright - long T2)
  Scalp fat            ~0.80  (bright - short T1)
  Gray matter (cortex) ~0.64  (intermediate - folded ribbon)
  Basal ganglia        ~0.55  (slightly above WM)
  Thalamus             ~0.52  (slightly above WM)
  White matter         ~0.40  (darker - long T1, moderate T2)
  Cortical bone        ~0.03  (very dark)
  Vascular flow void   ~0.01  (near-black)
  Background / air      0.00

Layering strategy
-----------------
Uses alpha compositing (painter's algorithm):
  img = img*(1-alpha) + signal*alpha     [_lerp helper]

Tissue radii (normalised elliptical distance from brain centre):
  1.000 + SCALP_T  -> outer scalp surface
  1.000            -> outer skull (R_SKULL_OUTER)
  1.000 - SKULL_T  -> inner skull / outer SAS  (R_SKULL_INNER)
  ...   - SAS_T    -> inner SAS / outer cortex  (R_SAS_INNER)
  ...   - CORTEX_T -> inner cortex / outer WM   (R_CORTEX_INNER)

Sulcal system uses polar-coordinate radial CSF wedges for realism.

Dev recipes (20 samples):
  brain_t2_supratentorial  25%  Mid-brain with full subcortical atlas
  brain_t2_temporal_slice  15%  Temporal lobe level (hippocampus, amygdala)
  brain_t2_frontal_slice   15%  Frontal lobe level (frontal horns)
  brain_t2_posterior_fossa 15%  Cerebellum + brainstem
  brain_t2_csf_rich        15%  Enlarged ventricles
  brain_t2_elderly         15%  Mild atrophy, prominent sulci

Hidden recipes (20 samples):
  brain_t2_glioma          20%  GBM-like: edema + core + necrosis
  brain_t2_ms_lesions      20%  MS plaques (periventricular + JC)
  brain_t2_stroke          15%  Ischemic stroke territory
  brain_t2_severe_atrophy  15%  Extreme cortical + subcortical atrophy
  brain_t2_hydrocephalus   10%  Massive ventricles + transependymal edema
  brain_t2_microbleeds     10%  Hemosiderin deposits (very dark spots)
  brain_t2_meningioma      10%  Extra-axial mass + brain compression
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (320, 320)

DEV_RECIPES = [
    ("brain_t2_supratentorial",  0.25),
    ("brain_t2_temporal_slice",  0.15),
    ("brain_t2_frontal_slice",   0.15),
    ("brain_t2_posterior_fossa", 0.15),
    ("brain_t2_csf_rich",        0.15),
    ("brain_t2_elderly",         0.15),
]

HIDDEN_RECIPES = [
    ("brain_t2_glioma",          0.20),
    ("brain_t2_ms_lesions",      0.20),
    ("brain_t2_stroke",          0.15),
    ("brain_t2_severe_atrophy",  0.15),
    ("brain_t2_hydrocephalus",   0.10),
    ("brain_t2_microbleeds",     0.10),
    ("brain_t2_meningioma",      0.10),
]


# ===========================================================================
# Public API
# ===========================================================================

def generate_mri_gt(seed: int, mode: str = "dev", shape: tuple = SHAPE) -> tuple:
    """Return (x: float32 (H,W) in [0,1], recipe_name: str)."""
    rng = np.random.default_rng(seed)
    recipes = DEV_RECIPES if mode == "dev" else HIDDEN_RECIPES
    names, probs = zip(*recipes)
    recipe = str(rng.choice(names, p=list(probs)))
    return _build_scene(recipe, shape, rng).astype(np.float32), recipe


# ===========================================================================
# Scene dispatcher
# ===========================================================================

def _build_scene(recipe: str, shape: tuple, rng: np.random.Generator) -> np.ndarray:
    if recipe == "brain_t2_supratentorial":
        return _brain_t2_base(shape, rng, slice_type="supratentorial")
    if recipe == "brain_t2_temporal_slice":
        return _brain_t2_temporal_slice(shape, rng)
    if recipe == "brain_t2_frontal_slice":
        return _brain_t2_frontal_slice(shape, rng)
    if recipe == "brain_t2_posterior_fossa":
        return _brain_t2_posterior_fossa(shape, rng)
    if recipe == "brain_t2_csf_rich":
        return _brain_t2_base(shape, rng, enlarged_ventricles=True, slice_type="supratentorial")
    if recipe == "brain_t2_elderly":
        return _brain_t2_elderly(shape, rng)
    if recipe == "brain_t2_glioma":
        return _brain_t2_glioma(shape, rng)
    if recipe == "brain_t2_ms_lesions":
        return _brain_t2_ms_lesions(shape, rng)
    if recipe == "brain_t2_stroke":
        return _brain_t2_stroke(shape, rng)
    if recipe == "brain_t2_severe_atrophy":
        return _brain_t2_base(shape, rng, atrophy=True, enlarged_ventricles=True,
                               slice_type="supratentorial")
    if recipe == "brain_t2_hydrocephalus":
        return _brain_t2_hydrocephalus(shape, rng)
    if recipe == "brain_t2_microbleeds":
        return _brain_t2_microbleeds(shape, rng)
    if recipe == "brain_t2_meningioma":
        return _brain_t2_meningioma(shape, rng)
    return _brain_t2_base(shape, rng, slice_type="supratentorial")


# ===========================================================================
# Low-level geometry + compositing helpers
# ===========================================================================

def _ellipse(H, W, cy, cx, ry, rx, angle_deg=0.0) -> np.ndarray:
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


def _soft(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-smooth a binary mask to a soft alpha [0, 1]."""
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def _lerp(img: np.ndarray, signal: float, alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite: blend img toward signal proportional to alpha."""
    return img * (1.0 - alpha) + float(signal) * alpha


def _gyral_field(theta: np.ndarray, rng: np.random.Generator,
                 n_min: int = 5, n_max: int = 13) -> np.ndarray:
    """Random angular sinusoidal field, normalised to unit std."""
    field = np.zeros_like(theta, dtype=np.float64)
    for n in range(n_min, n_max + 1):
        field += rng.uniform(0.5, 1.5) * np.sin(n * theta + rng.uniform(0, 2 * np.pi))
    return (field / (float(np.std(field)) + 1e-6)).astype(np.float32)


# ===========================================================================
# Sulcal channel system (polar radial CSF wedges)
# ===========================================================================

def _sulcal_channels(r_norm: np.ndarray, theta: np.ndarray,
                     sulci: list, R_SAS: float, sigma: float = 0.8) -> np.ndarray:
    """Compute a soft alpha map where sulcal CSF is present.

    sulci = list of (angle, depth, surface_width_rad, curve) tuples.
    Returns float32 array in [0,1].
    """
    result = np.zeros_like(r_norm, dtype=np.float32)
    for angle, depth, w0, curve in sulci:
        r_outer = R_SAS + 0.012
        r_inner = R_SAS - depth
        in_band = (r_norm < r_outer) & (r_norm > r_inner)
        r_frac = np.where(
            in_band,
            (r_norm - r_inner) / (r_outer - r_inner + 1e-6),
            0.0
        ).clip(0.0, 1.0)
        depth_frac = 1.0 - r_frac
        eff_angle = angle + curve * depth_frac
        d_theta = np.abs(((theta - eff_angle + np.pi) % (2.0 * np.pi)) - np.pi)
        w_r = w0 * (0.20 + 0.80 * r_frac)
        in_sulcus = in_band & (d_theta < w_r / 2.0)
        result = np.maximum(result,
                            gaussian_filter(in_sulcus.astype(np.float32), sigma=sigma))
    return result.clip(0.0, 1.0)


def _make_sulci(rng: np.random.Generator, R_SAS: float,
                has_sylvian: bool = True, atrophy: bool = False) -> list:
    """Generate a list of (angle, depth, surface_width_rad, curve) tuples."""
    sulci = []
    depth_scale = 1.35 if atrophy else 1.0
    width_scale = 1.30 if atrophy else 1.0

    # Interhemispheric fissure at top and bottom midline
    for angle in [-np.pi / 2.0, np.pi / 2.0]:
        sulci.append((angle, 0.060 * depth_scale, 0.020 * width_scale, 0.0))

    # Sylvian fissure bilateral - wide, at ~0 and pi from horizontal
    if has_sylvian:
        for base in [0.0, np.pi]:
            a = base + float(rng.uniform(-0.15, 0.15))
            sulci.append((
                a,
                float(rng.uniform(0.095, 0.145)) * depth_scale,
                float(rng.uniform(0.062, 0.090)) * width_scale,
                float(rng.uniform(-0.012, 0.012)),
            ))

    # Primary sulci (8-12, both hemispheres)
    n_primary = int(rng.integers(8, 13))
    angles_used = [s[0] for s in sulci]
    for _ in range(n_primary):
        a = float(rng.uniform(-np.pi, np.pi))
        if any(abs(((a - b + np.pi) % (2 * np.pi)) - np.pi) < 0.16
               for b in angles_used):
            continue
        angles_used.append(a)
        sulci.append((
            a,
            float(rng.uniform(0.060, 0.115)) * depth_scale,
            float(rng.uniform(0.030, 0.048)) * width_scale,
            float(rng.uniform(-0.010, 0.010)),
        ))

    # Secondary sulci (6-10, shallower)
    n_secondary = int(rng.integers(6, 11))
    for _ in range(n_secondary):
        a = float(rng.uniform(-np.pi, np.pi))
        sulci.append((
            a,
            float(rng.uniform(0.028, 0.058)) * depth_scale,
            float(rng.uniform(0.016, 0.026)) * width_scale,
            float(rng.uniform(-0.006, 0.006)),
        ))

    return sulci


# ===========================================================================
# WM fiber texture
# ===========================================================================

def _wm_fiber_texture(H: int, W: int, wm_mask: np.ndarray,
                      rng: np.random.Generator, amplitude: float = 0.005) -> np.ndarray:
    """Subtle slow-varying WM intensity modulation (multi-scale smooth blobs).

    Uses superimposed smoothed noise at several spatial scales to produce the
    low-frequency heterogeneity seen in real WM — NOT visible as parallel bands.
    """
    texture = np.zeros((H, W), dtype=np.float32)
    for sigma, amp in [(50, 1.0), (25, 0.6), (12, 0.3)]:
        t = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                            sigma=sigma)
        t -= t.mean()
        t /= (np.abs(t).max() + 1e-6)
        texture += amp * t
    texture /= (np.abs(texture).max() + 1e-6)
    return texture * wm_mask.astype(np.float32) * amplitude


# ===========================================================================
# Vascular flow voids (Circle of Willis)
# ===========================================================================

def _add_flow_voids(img: np.ndarray, cy: float, cx: float,
                    H: int, W: int, rng: np.random.Generator,
                    flow_sig: float, vent_y: float) -> np.ndarray:
    """Paint Circle of Willis flow voids as near-black small circles."""
    # Basilar artery - midline, posterior
    ba_y = vent_y + H * float(rng.uniform(0.06, 0.10))
    r_ba = int(rng.integers(2, 4))
    ba = _ellipse(H, W, ba_y, cx, r_ba, r_ba)
    img = _lerp(img, flow_sig, _soft(ba.astype(np.float32), 0.7))

    # ICA bilateral (lateral to hypothalamus)
    ica_dx = W * float(rng.uniform(0.055, 0.075))
    ica_y = vent_y + H * float(rng.uniform(0.03, 0.06))
    for side in [-1, 1]:
        r_ica = int(rng.integers(2, 4))
        ica = _ellipse(H, W, ica_y, cx + side * ica_dx, r_ica, r_ica)
        img = _lerp(img, flow_sig, _soft(ica.astype(np.float32), 0.7))

    # MCA bilateral (Sylvian fissure area)
    mca_dx = W * float(rng.uniform(0.10, 0.14))
    mca_y = cy + H * float(rng.uniform(-0.04, 0.02))
    for side in [-1, 1]:
        r_mca = int(rng.integers(2, 5))
        mca = _ellipse(H, W, mca_y, cx + side * mca_dx, r_mca, r_mca)
        img = _lerp(img, flow_sig, _soft(mca.astype(np.float32), 0.7))

    return img


# ===========================================================================
# MRI field effects (shared)
# ===========================================================================

def _apply_field_effects(img: np.ndarray, yy: np.ndarray, xx: np.ndarray,
                         H: int, W: int, r_norm: np.ndarray,
                         rng: np.random.Generator,
                         r_skull_inner: float = 0.952) -> np.ndarray:
    """Apply B1+ inhomogeneity, coil roll-off, and Rician noise."""
    # B1+ inhomogeneity (3T: brighter centre)
    b1_sig = float(rng.uniform(0.50, 0.70))
    b1_amp = float(rng.uniform(0.04, 0.10))
    b1 = 1.0 + b1_amp * np.exp(
        -((yy / (H * b1_sig)) ** 2 + (xx / (W * b1_sig)) ** 2)
    ).astype(np.float32)
    img = img * b1

    # Receive coil roll-off (mild linear gradient)
    ga = float(rng.uniform(0, 2 * np.pi))
    gamp = float(rng.uniform(0.02, 0.06))
    ramp = (1.0 + gamp * (np.cos(ga) * yy / H + np.sin(ga) * xx / W)).astype(np.float32)
    img = img * ramp.clip(0.85, 1.15)

    # Rician-like noise (only inside skull)
    brain_mask = (r_norm < r_skull_inner).astype(np.float32)
    sn = float(rng.uniform(0.010, 0.022))
    n1 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    n2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32),
                         sigma=float(rng.uniform(0.8, 1.5)))
    img = img + sn * (np.sqrt(n1 ** 2 + n2 ** 2) - float(np.sqrt(np.pi / 2))) * brain_mask

    return img.clip(0.0, 1.0)


# ===========================================================================
# Main supratentorial brain builder
# ===========================================================================

def _brain_t2_base(shape: tuple, rng: np.random.Generator, *,
                   enlarged_ventricles: bool = False,
                   atrophy: bool = False,
                   high_contrast: bool = False,
                   slice_type: str = "supratentorial") -> np.ndarray:
    """Full-featured axial T2w supratentorial brain phantom.

    Layering order:
      1. Scalp fat
      2. Calvarium (skull bone)
      3. Subarachnoid space base CSF
      4. White matter (fills brain interior)
      5. WM fiber texture
      6. Cortex (GM ribbon with gentle gyral perturbation)
      7. Sulcal CSF channels (deep radial wedges)
      8. Subcortical structures (BG, thalami, IC, claustrum)
      9. Lateral + 3rd ventricles with choroid plexus
     10. Falx cerebri + interhemispheric CSF
     11. Vascular flow voids (Circle of Willis)
     12. MRI field effects
    """
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    # --- Geometry -----------------------------------------------------------
    cy = H / 2.0 + rng.uniform(-8.0, 8.0)
    cx = W / 2.0 + rng.uniform(-6.0, 6.0)
    skull_ry = H * rng.uniform(0.420, 0.448) * rng.uniform(0.97, 1.03)
    skull_rx = W * rng.uniform(0.385, 0.415) * rng.uniform(0.97, 1.03)

    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    # --- Signal intensities (T2w model) -------------------------------------
    CSF_SIG    = float(rng.uniform(0.90, 0.96))
    SCALP_SIG  = float(rng.uniform(0.74, 0.86))
    SKULL_SIG  = float(rng.uniform(0.02, 0.05))
    GM_SIG     = float(rng.uniform(0.58, 0.70))
    WM_SIG     = float(rng.uniform(0.22, 0.32) if high_contrast else rng.uniform(0.32, 0.44))
    PUTAMEN_SIG = float(rng.uniform(0.50, 0.60))
    CAUDATE_SIG = float(rng.uniform(0.52, 0.62))
    GP_SIG      = float(rng.uniform(0.38, 0.50))
    THAL_SIG    = float(rng.uniform(0.46, 0.58))
    IC_SIG      = WM_SIG * float(rng.uniform(0.88, 0.96))
    FALX_SIG    = float(rng.uniform(0.08, 0.18))
    FLOW_SIG    = float(rng.uniform(0.00, 0.03))
    CHPLX_SIG   = float(rng.uniform(0.50, 0.62))

    # --- Structural radii (normalised) --------------------------------------
    SKULL_T   = 0.048
    SAS_T     = 0.028
    CORTEX_T  = 0.092
    GYRAL_AMP = 0.018

    if atrophy:
        SAS_T     += 0.020
        CORTEX_T  *= 0.80

    R_SKULL_INNER  = 1.000 - SKULL_T
    R_SAS_INNER    = R_SKULL_INNER - SAS_T
    R_CORTEX_INNER = R_SAS_INNER   - CORTEX_T

    # --- Gyral folding field ------------------------------------------------
    gyral = _gyral_field(theta, rng, n_min=5, n_max=13)
    R_CORTEX_OUTER_EFF = (R_SAS_INNER + gyral * GYRAL_AMP).astype(np.float32)

    # --- Layer 1: scalp fat -------------------------------------------------
    img = _lerp(img, SCALP_SIG,
                _soft((r_norm < 1.000 + 0.060).astype(np.float32), 1.5))

    # --- Layer 2: calvarium -------------------------------------------------
    skull_ring = (r_norm < 1.000) & (r_norm >= R_SKULL_INNER)
    img = _lerp(img, SKULL_SIG, _soft(skull_ring.astype(np.float32), 1.2))

    # --- Layer 3: SAS base CSF ----------------------------------------------
    sas_wide = (r_norm < R_SKULL_INNER) & (r_norm >= R_SAS_INNER - GYRAL_AMP * 3.0)
    img = _lerp(img, CSF_SIG, _soft(sas_wide.astype(np.float32), 1.2))

    # --- Layer 4: white matter ----------------------------------------------
    wm_mask = (r_norm < R_CORTEX_INNER).astype(np.float32)
    img = _lerp(img, WM_SIG, _soft(wm_mask, 2.0))

    # --- Layer 4b: WM fiber texture -----------------------------------------
    texture = _wm_fiber_texture(H, W, r_norm < R_CORTEX_INNER, rng, amplitude=0.018)
    img = img + texture  # additive texture on WM only

    # --- Layer 5: cortex (GM ribbon) ----------------------------------------
    cortex_mask = (r_norm < R_CORTEX_OUTER_EFF) & (r_norm >= R_CORTEX_INNER)
    img = _lerp(img, GM_SIG, _soft(cortex_mask.astype(np.float32), 0.9))

    # --- Layer 6: sulcal CSF channels ---------------------------------------
    sulci = _make_sulci(rng, R_SAS_INNER, has_sylvian=True, atrophy=atrophy)
    sulcal_alpha = _sulcal_channels(r_norm, theta, sulci, R_SAS_INNER, sigma=0.8)
    img = _lerp(img, CSF_SIG, sulcal_alpha)

    # --- Layer 7: subcortical structures ------------------------------------
    vent_y = cy + H * rng.uniform(-0.05, 0.01)
    bg_dx  = W  * rng.uniform(0.078, 0.108)

    for side in [-1, 1]:
        bgx = cx + side * bg_dx

        # Putamen (lateral to IC)
        put_cy = vent_y + H * rng.uniform(0.005, 0.025)
        put = _ellipse(H, W, put_cy, bgx,
                       H * 0.068, W * 0.052, angle_deg=side * float(rng.uniform(5, 15)))
        img = _lerp(img, PUTAMEN_SIG, _soft(put.astype(np.float32), 2.5))

        # Caudate head (medial to IC, adjacent to frontal horn)
        caud_x = cx + side * bg_dx * 0.52
        caud_cy = vent_y + H * rng.uniform(-0.05, -0.02)
        caud = _ellipse(H, W, caud_cy, caud_x, H * 0.038, W * 0.030)
        img = _lerp(img, CAUDATE_SIG, _soft(caud.astype(np.float32), 2.0))

        # Globus pallidus (medial to putamen, iron-rich => darker)
        gp_x = cx + side * bg_dx * 0.72
        gp = _ellipse(H, W, put_cy + H * 0.012, gp_x, H * 0.040, W * 0.028)
        img = _lerp(img, GP_SIG, _soft(gp.astype(np.float32), 2.0))

        # Internal capsule (thin WM lane between BG and thalamus)
        ic_x = cx + side * bg_dx * 0.88
        ic = _ellipse(H, W, put_cy + H * 0.018, ic_x,
                      H * 0.058, W * 0.014, angle_deg=side * float(rng.uniform(8, 18)))
        img = _lerp(img, IC_SIG, _soft(ic.astype(np.float32), 1.5))

        # Thalamus (medial to IC)
        thal_x = cx + side * W * 0.044
        thal = _ellipse(H, W, put_cy + H * 0.032, thal_x, H * 0.062, W * 0.054)
        img = _lerp(img, THAL_SIG, _soft(thal.astype(np.float32), 2.5))

        # Claustrum (thin GM strip lateral to putamen)
        claus_x = cx + side * bg_dx * 1.18
        claus = _ellipse(H, W, put_cy, claus_x,
                         H * 0.050, W * 0.010, angle_deg=side * float(rng.uniform(5, 15)))
        img = _lerp(img, GM_SIG * 1.02, _soft(claus.astype(np.float32), 1.2))

    # --- Layer 8: lateral ventricles ----------------------------------------
    v_scale = float(rng.uniform(1.5, 2.2) if enlarged_ventricles else rng.uniform(0.75, 1.25))
    vent_dx = W * rng.uniform(0.088, 0.130)

    for side in [-1, 1]:
        vx = cx + side * vent_dx
        # Frontal horn
        fh = _ellipse(H, W, vent_y - H * 0.050, vx - side * W * 0.016,
                      H * 0.065 * v_scale, W * 0.044 * v_scale, angle_deg=side * 14)
        # Body
        bdy = _ellipse(H, W, vent_y + H * 0.010, vx,
                       H * 0.055 * v_scale, W * 0.055 * v_scale)
        # Occipital horn
        oh = _ellipse(H, W, vent_y + H * 0.062, vx + side * W * 0.014,
                      H * 0.058 * v_scale, W * 0.040 * v_scale, angle_deg=-side * 12)
        vent = (fh | bdy | oh).astype(np.float32)
        img = _lerp(img, CSF_SIG, _soft(vent, 2.5))

        # Choroid plexus (bright dots inside lateral ventricle body)
        chp_x = vx + side * W * float(rng.uniform(-0.01, 0.01))
        chp_y = vent_y + H * float(rng.uniform(0.01, 0.03))
        chp = _ellipse(H, W, chp_y, chp_x,
                       H * 0.016 * v_scale, W * 0.022 * v_scale)
        img = _lerp(img, CHPLX_SIG, _soft(chp.astype(np.float32), 1.5))

    # Third ventricle (midline slit)
    third = _ellipse(H, W, vent_y + H * 0.030, cx, H * 0.055, W * 0.013)
    img = _lerp(img, CSF_SIG, _soft(third.astype(np.float32), 1.8))

    # Cerebral aqueduct (~40% of slices)
    if rng.random() < 0.40:
        aq = _ellipse(H, W, vent_y + H * 0.075, cx, H * 0.016, W * 0.013)
        img = _lerp(img, CSF_SIG, _soft(aq.astype(np.float32), 1.2))

    # --- Layer 9: falx cerebri + interhemispheric CSF -----------------------
    # Interhemispheric fissure: soft CSF band along top midline (above brain)
    # and bottom midline, rendered as a narrow vertical strip in the SAS region
    for sign in [-1.0, 1.0]:   # sign=-1 → top (yy<0), sign=+1 → bottom (yy>0)
        fiss = np.where(
            (np.abs(xx) < 5.0) & (sign * yy > 0)
            & (r_norm > R_CORTEX_INNER) & (r_norm < R_SAS_INNER + 0.010),
            1.0, 0.0
        ).astype(np.float32)
        fiss = gaussian_filter(fiss, sigma=1.5)
        img = _lerp(img, CSF_SIG, fiss.clip(0.0, 1.0) * 0.70)

    # Falx cerebri: thin dark dura band along full midline inside SAS
    falx = np.where(
        (np.abs(xx) < 2.0) & (r_norm < R_SAS_INNER + 0.008) & (r_norm > R_CORTEX_INNER),
        1.0, 0.0
    ).astype(np.float32)
    falx = gaussian_filter(falx, sigma=1.2)
    img = _lerp(img, FALX_SIG, falx.clip(0.0, 1.0) * 0.55)

    # --- Layer 10: Circle of Willis flow voids ------------------------------
    img = _add_flow_voids(img, cy, cx, H, W, rng, FLOW_SIG, vent_y)

    # --- Layer 11: MRI field effects ----------------------------------------
    img = _apply_field_effects(img, yy, xx, H, W, r_norm, rng,
                               r_skull_inner=R_SKULL_INNER)
    return img


# ===========================================================================
# Temporal slice
# ===========================================================================

def _brain_t2_temporal_slice(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Temporal lobe level slice with hippocampus, amygdala, temporal horn."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    cy = H / 2.0 + rng.uniform(-6.0, 6.0)
    cx = W / 2.0 + rng.uniform(-6.0, 6.0)
    # Temporal slices are slightly more triangular / oval
    skull_ry = H * rng.uniform(0.395, 0.430)
    skull_rx = W * rng.uniform(0.400, 0.435)

    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    CSF_SIG   = float(rng.uniform(0.90, 0.96))
    SCALP_SIG = float(rng.uniform(0.74, 0.86))
    SKULL_SIG = float(rng.uniform(0.02, 0.05))
    GM_SIG    = float(rng.uniform(0.58, 0.70))
    WM_SIG    = float(rng.uniform(0.32, 0.44))
    HIPPO_SIG = float(rng.uniform(0.60, 0.68))
    AMYG_SIG  = float(rng.uniform(0.58, 0.65))

    SKULL_T, SAS_T, CORTEX_T, GYRAL_AMP = 0.048, 0.028, 0.092, 0.018
    R_SKULL_INNER  = 1.000 - SKULL_T
    R_SAS_INNER    = R_SKULL_INNER - SAS_T
    R_CORTEX_INNER = R_SAS_INNER   - CORTEX_T

    gyral = _gyral_field(theta, rng, n_min=5, n_max=13)
    R_CORTEX_OUTER_EFF = (R_SAS_INNER + gyral * GYRAL_AMP).astype(np.float32)

    img = _lerp(img, SCALP_SIG, _soft((r_norm < 1.060).astype(np.float32), 1.5))
    skull_ring = (r_norm < 1.000) & (r_norm >= R_SKULL_INNER)
    img = _lerp(img, SKULL_SIG, _soft(skull_ring.astype(np.float32), 1.2))
    sas_wide = (r_norm < R_SKULL_INNER) & (r_norm >= R_SAS_INNER - GYRAL_AMP * 3.0)
    img = _lerp(img, CSF_SIG, _soft(sas_wide.astype(np.float32), 1.2))
    wm_mask = r_norm < R_CORTEX_INNER
    img = _lerp(img, WM_SIG, _soft(wm_mask.astype(np.float32), 2.0))
    texture = _wm_fiber_texture(H, W, wm_mask, rng)
    img = img + texture
    cortex_mask = (r_norm < R_CORTEX_OUTER_EFF) & (r_norm >= R_CORTEX_INNER)
    img = _lerp(img, GM_SIG, _soft(cortex_mask.astype(np.float32), 0.9))

    # Temporal sulci (no Sylvian fissure at this level - instead more temporal sulci)
    sulci = _make_sulci(rng, R_SAS_INNER, has_sylvian=False, atrophy=False)
    # Add temporal sulci manually
    for _ in range(int(rng.integers(3, 6))):
        a = float(rng.uniform(-0.6, 0.6))  # lateral temporal
        for side_sign in [-1.0, 1.0]:
            sulci.append((side_sign * (np.pi / 2.0 + a),
                          float(rng.uniform(0.06, 0.10)),
                          float(rng.uniform(0.025, 0.042)),
                          float(rng.uniform(-0.008, 0.008))))
    sulcal_alpha = _sulcal_channels(r_norm, theta, sulci, R_SAS_INNER)
    img = _lerp(img, CSF_SIG, sulcal_alpha)

    # Temporal horn of lateral ventricle (narrow CSF curve lateral+inferior)
    vent_y = cy + H * rng.uniform(0.02, 0.05)
    for side in [-1, 1]:
        th_cx = cx + side * W * float(rng.uniform(0.12, 0.16))
        th_cy = vent_y + H * float(rng.uniform(0.04, 0.08))
        th_ry = H * float(rng.uniform(0.020, 0.035))
        th_rx = W * float(rng.uniform(0.030, 0.050))
        th = _ellipse(H, W, th_cy, th_cx, th_ry, th_rx,
                      angle_deg=side * float(rng.uniform(20, 40)))
        img = _lerp(img, CSF_SIG, _soft(th.astype(np.float32), 1.8))

        # Hippocampus (curved GM medial temporal, adjacent to temporal horn)
        hipp_cx = cx + side * W * float(rng.uniform(0.085, 0.115))
        hipp_cy = th_cy + H * float(rng.uniform(0.005, 0.018))
        hipp = _ellipse(H, W, hipp_cy, hipp_cx,
                        H * float(rng.uniform(0.025, 0.038)),
                        W * float(rng.uniform(0.038, 0.055)),
                        angle_deg=side * float(rng.uniform(15, 30)))
        img = _lerp(img, HIPPO_SIG, _soft(hipp.astype(np.float32), 1.8))

        # Amygdala (anterior to hippocampus)
        amyg_cy = hipp_cy - H * float(rng.uniform(0.010, 0.020))
        amyg_cx = hipp_cx + side * W * float(rng.uniform(-0.010, 0.010))
        amyg = _ellipse(H, W, amyg_cy, amyg_cx,
                        H * float(rng.uniform(0.020, 0.030)),
                        W * float(rng.uniform(0.022, 0.032)))
        img = _lerp(img, AMYG_SIG, _soft(amyg.astype(np.float32), 1.5))

    # Temporal CSF spaces (more prominent at this level)
    for side in [-1, 1]:
        temp_csf_cx = cx + side * W * float(rng.uniform(0.15, 0.20))
        temp_csf_cy = cy + H * float(rng.uniform(0.05, 0.12))
        tc = _ellipse(H, W, temp_csf_cy, temp_csf_cx,
                      H * float(rng.uniform(0.025, 0.045)),
                      W * float(rng.uniform(0.020, 0.035)))
        img = _lerp(img, CSF_SIG, _soft(tc.astype(np.float32), 2.0))

    img = _apply_field_effects(img, yy, xx, H, W, r_norm, rng,
                               r_skull_inner=R_SKULL_INNER)
    return img


# ===========================================================================
# Frontal slice
# ===========================================================================

def _brain_t2_frontal_slice(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Frontal lobe level slice with frontal horns and caudate heads."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    cy = H / 2.0 + rng.uniform(-6.0, 6.0)
    cx = W / 2.0 + rng.uniform(-6.0, 6.0)
    # Frontal slices: rounder/oval
    skull_ry = H * rng.uniform(0.430, 0.458)
    skull_rx = W * rng.uniform(0.390, 0.420)

    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    CSF_SIG   = float(rng.uniform(0.90, 0.96))
    SCALP_SIG = float(rng.uniform(0.74, 0.86))
    SKULL_SIG = float(rng.uniform(0.02, 0.05))
    GM_SIG    = float(rng.uniform(0.58, 0.70))
    WM_SIG    = float(rng.uniform(0.32, 0.44))
    CAUDATE_SIG = float(rng.uniform(0.52, 0.62))

    SKULL_T, SAS_T, CORTEX_T, GYRAL_AMP = 0.048, 0.028, 0.092, 0.018
    R_SKULL_INNER  = 1.000 - SKULL_T
    R_SAS_INNER    = R_SKULL_INNER - SAS_T
    R_CORTEX_INNER = R_SAS_INNER   - CORTEX_T

    gyral = _gyral_field(theta, rng, n_min=5, n_max=13)
    R_CORTEX_OUTER_EFF = (R_SAS_INNER + gyral * GYRAL_AMP).astype(np.float32)

    img = _lerp(img, SCALP_SIG, _soft((r_norm < 1.060).astype(np.float32), 1.5))
    skull_ring = (r_norm < 1.000) & (r_norm >= R_SKULL_INNER)
    img = _lerp(img, SKULL_SIG, _soft(skull_ring.astype(np.float32), 1.2))
    sas_wide = (r_norm < R_SKULL_INNER) & (r_norm >= R_SAS_INNER - GYRAL_AMP * 3.0)
    img = _lerp(img, CSF_SIG, _soft(sas_wide.astype(np.float32), 1.2))
    wm_mask = r_norm < R_CORTEX_INNER
    img = _lerp(img, WM_SIG, _soft(wm_mask.astype(np.float32), 2.0))
    texture = _wm_fiber_texture(H, W, wm_mask, rng)
    img = img + texture
    cortex_mask = (r_norm < R_CORTEX_OUTER_EFF) & (r_norm >= R_CORTEX_INNER)
    img = _lerp(img, GM_SIG, _soft(cortex_mask.astype(np.float32), 0.9))

    # Frontal sulci - no Sylvian at this level, prominent interhemispheric fissure
    sulci = _make_sulci(rng, R_SAS_INNER, has_sylvian=False, atrophy=False)
    sulcal_alpha = _sulcal_channels(r_norm, theta, sulci, R_SAS_INNER)
    img = _lerp(img, CSF_SIG, sulcal_alpha)

    # Frontal horns of lateral ventricles
    vent_y = cy + H * rng.uniform(-0.08, -0.02)
    v_scale = float(rng.uniform(0.80, 1.20))
    vent_dx = W * rng.uniform(0.075, 0.105)
    for side in [-1, 1]:
        vx = cx + side * vent_dx
        fh = _ellipse(H, W, vent_y, vx,
                      H * 0.058 * v_scale, W * 0.040 * v_scale,
                      angle_deg=side * float(rng.uniform(10, 20)))
        img = _lerp(img, CSF_SIG, _soft(fh.astype(np.float32), 2.0))

        # Caudate head adjacent to frontal horn
        caud_x = cx + side * vent_dx * 1.35
        caud = _ellipse(H, W, vent_y, caud_x, H * 0.038, W * 0.030)
        img = _lerp(img, CAUDATE_SIG, _soft(caud.astype(np.float32), 2.0))

    # Prominent interhemispheric fissure (wide at frontal level)
    fiss_alpha = np.where(
        (np.abs(xx) < 5.0) & (r_norm < R_SAS_INNER + 0.020),
        1.0, 0.0
    ).astype(np.float32)
    fiss_alpha = gaussian_filter(fiss_alpha, sigma=1.0)
    img = _lerp(img, CSF_SIG, fiss_alpha.clip(0.0, 1.0))

    # Falx
    falx_alpha = np.where(
        (np.abs(xx) < 1.2) & (r_norm < R_CORTEX_INNER + 0.08),
        1.0, 0.0
    ).astype(np.float32)
    falx_alpha = gaussian_filter(falx_alpha, sigma=0.5)
    img = _lerp(img, float(rng.uniform(0.08, 0.18)), falx_alpha.clip(0.0, 1.0))

    img = _apply_field_effects(img, yy, xx, H, W, r_norm, rng,
                               r_skull_inner=R_SKULL_INNER)
    return img


# ===========================================================================
# Posterior fossa
# ===========================================================================

def _brain_t2_posterior_fossa(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Posterior fossa slice: bilateral cerebellum + brainstem + cisterns."""
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)

    cy = H / 2.0 + rng.uniform(-8.0, 8.0)
    cx = W / 2.0 + rng.uniform(-6.0, 6.0)
    skull_ry = H * rng.uniform(0.355, 0.398)
    skull_rx = W * rng.uniform(0.345, 0.388)

    yy = np.arange(H, dtype=np.float64)[:, None] - cy
    xx = np.arange(W, dtype=np.float64)[None, :] - cx
    theta = np.arctan2(yy, xx).astype(np.float32)
    r_norm = np.sqrt((yy / skull_ry) ** 2 + (xx / skull_rx) ** 2).astype(np.float32)

    CSF_SIG   = float(rng.uniform(0.90, 0.95))
    SCALP_SIG = float(rng.uniform(0.74, 0.86))
    SKULL_SIG = float(rng.uniform(0.02, 0.05))
    GM_SIG    = float(rng.uniform(0.58, 0.68))
    WM_SIG    = float(rng.uniform(0.34, 0.44))
    DENT_SIG  = float(rng.uniform(0.55, 0.62))  # dentate nucleus

    R_SKULL_INNER  = 0.952
    R_SAS_INNER    = 0.924
    R_CORTEX_INNER = 0.820

    # Scalp and skull
    img = _lerp(img, SCALP_SIG, _soft((r_norm < 1.060).astype(np.float32), 1.5))
    skull_ring = (r_norm < 1.000) & (r_norm >= R_SKULL_INNER)
    img = _lerp(img, SKULL_SIG, _soft(skull_ring.astype(np.float32), 1.2))
    img = _lerp(img, CSF_SIG,  _soft((r_norm < R_SKULL_INNER).astype(np.float32), 1.2))
    img = _lerp(img, WM_SIG,   _soft((r_norm < R_CORTEX_INNER).astype(np.float32), 2.0))

    # Cerebellar cortex with fine foliation (folia: n_min=12, n_max=22)
    cer_gyral = _gyral_field(theta, rng, n_min=12, n_max=22)
    CEREBELLAR_GYRAL_AMP = 0.022
    R_CER_CORTEX_OUTER = (R_CORTEX_INNER + cer_gyral * CEREBELLAR_GYRAL_AMP).astype(np.float32)
    cer_cortex_inner = R_CORTEX_INNER - 0.090
    cer_cortex = (r_norm < R_CER_CORTEX_OUTER) & (r_norm >= cer_cortex_inner)
    cer_sulcus = (r_norm >= R_CER_CORTEX_OUTER) & (r_norm < R_SAS_INNER)
    img = _lerp(img, GM_SIG,  _soft(cer_cortex.astype(np.float32), 0.9))
    img = _lerp(img, CSF_SIG, _soft(cer_sulcus.astype(np.float32), 0.9))

    # Brainstem (central oval)
    bs_cy = cy + H * float(rng.uniform(-0.02, 0.04))
    bs_cx = cx + W * float(rng.uniform(-0.02, 0.02))
    bs_ry = H * float(rng.uniform(0.105, 0.130))
    bs_rx = W * float(rng.uniform(0.090, 0.115))
    bs = _ellipse(H, W, bs_cy, bs_cx, bs_ry, bs_rx)
    img = _lerp(img, WM_SIG, _soft(bs.astype(np.float32), 2.5))

    # 4th ventricle
    v4_cy = bs_cy + bs_ry * 0.80
    v4 = _ellipse(H, W, v4_cy, bs_cx,
                  H * float(rng.uniform(0.035, 0.055)),
                  W * float(rng.uniform(0.055, 0.080)))
    img = _lerp(img, CSF_SIG, _soft(v4.astype(np.float32), 1.8))

    # Cerebral aqueduct
    aq = _ellipse(H, W, bs_cy - bs_ry * 0.60, bs_cx, H * 0.015, W * 0.013)
    img = _lerp(img, CSF_SIG, _soft(aq.astype(np.float32), 1.2))

    # Dentate nuclei (bilateral, slightly darker GM in cerebellar WM)
    for side in [-1, 1]:
        dent_cx = bs_cx + side * W * float(rng.uniform(0.05, 0.08))
        dent_cy = bs_cy + H * float(rng.uniform(0.02, 0.05))
        dent = _ellipse(H, W, dent_cy, dent_cx,
                        H * float(rng.uniform(0.018, 0.028)),
                        W * float(rng.uniform(0.022, 0.032)))
        img = _lerp(img, DENT_SIG, _soft(dent.astype(np.float32), 1.5))

    # Prepontine cistern
    cist = _ellipse(H, W, bs_cy - bs_ry * 1.30, bs_cx,
                    H * 0.040, W * float(rng.uniform(0.16, 0.24)))
    img = _lerp(img, CSF_SIG, _soft(cist.astype(np.float32), 3.0))

    # CPA cisterns
    for side in [-1, 1]:
        cpa_cx = cx + side * W * float(rng.uniform(0.13, 0.18))
        cpa = _ellipse(H, W, bs_cy + H * 0.02, cpa_cx,
                       H * float(rng.uniform(0.03, 0.05)),
                       W * float(rng.uniform(0.04, 0.07)))
        img = _lerp(img, CSF_SIG, _soft(cpa.astype(np.float32), 2.0))

    # Basilar artery flow void
    ba = _ellipse(H, W, bs_cy - bs_ry * 0.90, bs_cx, H * 0.012, W * 0.012)
    img = _lerp(img, float(rng.uniform(0.00, 0.03)), _soft(ba.astype(np.float32), 0.8))

    img = _apply_field_effects(img, yy, xx, H, W, r_norm, rng,
                               r_skull_inner=R_SKULL_INNER)
    return img


# ===========================================================================
# Elderly (atrophy)
# ===========================================================================

def _brain_t2_elderly(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Elderly brain: atrophy, prominent sulci, leukoaraiosis, small WM lesions."""
    H, W = shape
    # Build base with atrophy + enlarged ventricles
    img = _brain_t2_base(shape, rng, atrophy=True, enlarged_ventricles=True,
                         slice_type="supratentorial")

    # Add periventricular leukoaraiosis (mild T2 signal increase around ventricles)
    cy = H / 2.0
    cx = W / 2.0
    vent_y = cy + rng.uniform(-0.05, 0.01) * H
    vent_dx = W * rng.uniform(0.088, 0.130)
    v_scale = rng.uniform(1.4, 2.0)
    leukoaraiosis_sig = float(rng.uniform(0.48, 0.58))

    for side in [-1, 1]:
        vx = cx + side * vent_dx
        pv_halo = _ellipse(H, W, vent_y, vx,
                           H * 0.12 * v_scale, W * 0.10 * v_scale)
        img = _lerp(img, leukoaraiosis_sig,
                    _soft(pv_halo.astype(np.float32), 4.0) * 0.35)

    # Small non-specific WM hyperintensities (2-5 lesions)
    n_wml = int(rng.integers(2, 6))
    for _ in range(n_wml):
        ldir  = float(rng.uniform(0, 2 * np.pi))
        ldist = H * float(rng.uniform(0.05, 0.18))
        ly    = vent_y + ldist * np.cos(ldir)
        lx    = cx     + ldist * np.sin(ldir)
        lr    = H * float(rng.uniform(0.008, 0.018))
        lesion = _ellipse(H, W, ly, lx, lr, lr * float(rng.uniform(0.8, 1.5)),
                          angle_deg=float(rng.uniform(0, 180)))
        img = _lerp(img, float(rng.uniform(0.50, 0.62)),
                    _soft(lesion.astype(np.float32), 1.2))

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: Glioma (GBM-like)
# ===========================================================================

def _brain_t2_glioma(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """GBM-like: irregular mass with necrosis, tumor core, vasogenic edema."""
    H, W = shape
    img = _brain_t2_base(shape, rng, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0

    # Place tumor off-center in one hemisphere
    side = rng.choice([-1, 1])
    t_cy = cy + H * float(rng.uniform(-0.10, 0.12))
    t_cx = cx + side * W * float(rng.uniform(0.06, 0.16))
    t_ry = float(rng.uniform(25, 55))
    t_rx = t_ry * float(rng.uniform(0.70, 1.30))

    EDEMA_SIG  = float(rng.uniform(0.78, 0.92))
    CORE_SIG   = float(rng.uniform(0.70, 0.82))
    NECRO_SIG  = float(rng.uniform(0.88, 0.96))

    # Vasogenic edema halo (largest zone, finger-like extensions along WM)
    edema_mask = _ellipse(H, W, t_cy, t_cx, t_ry * 1.55, t_rx * 1.55)
    # Add finger-like extensions
    n_fingers = int(rng.integers(3, 7))
    for _ in range(n_fingers):
        fa = float(rng.uniform(-np.pi, np.pi))
        fd = float(rng.uniform(0.5, 1.2)) * t_ry
        f_cy = t_cy + fd * np.cos(fa)
        f_cx = t_cx + fd * np.sin(fa)
        f_ry = float(rng.uniform(6, 14))
        f_rx = f_ry * float(rng.uniform(0.4, 0.9))
        edema_mask = edema_mask | _ellipse(H, W, f_cy, f_cx, f_ry, f_rx,
                                           angle_deg=float(rng.uniform(0, 180)))
    img = _lerp(img, EDEMA_SIG, _soft(edema_mask.astype(np.float32), 3.5))

    # Tumor core
    core_mask = _ellipse(H, W, t_cy, t_cx, t_ry, t_rx)
    img = _lerp(img, CORE_SIG, _soft(core_mask.astype(np.float32), 2.5))

    # Necrotic center with heterogeneous signal
    necro_ry = t_ry * float(rng.uniform(0.40, 0.60))
    necro_rx = t_rx * float(rng.uniform(0.40, 0.60))
    necro_mask = _ellipse(H, W, t_cy + H * float(rng.uniform(-0.02, 0.02)),
                          t_cx + W * float(rng.uniform(-0.02, 0.02)),
                          necro_ry, necro_rx)
    # Heterogeneous necrosis signal (random blobs)
    necro_base = np.full((H, W), NECRO_SIG, dtype=np.float32)
    for _ in range(int(rng.integers(4, 10))):
        ba = float(rng.uniform(-np.pi, np.pi))
        bd = float(rng.uniform(0, 0.7)) * necro_ry
        b_cy = t_cy + bd * np.cos(ba)
        b_cx = t_cx + bd * np.sin(ba)
        b_r  = float(rng.uniform(3, 10))
        b_mask = _ellipse(H, W, b_cy, b_cx, b_r, b_r)
        blob_sig = float(rng.uniform(0.55, 0.90))
        necro_base = np.where(b_mask, blob_sig, necro_base)
    necro_alpha = _soft(necro_mask.astype(np.float32), 2.0)
    img = img * (1.0 - necro_alpha) + necro_base * necro_alpha

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: MS lesions
# ===========================================================================

def _brain_t2_ms_lesions(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Multiple sclerosis: periventricular + juxtacortical + subcortical lesions."""
    H, W = shape
    img = _brain_t2_base(shape, rng, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0
    vent_y = cy + rng.uniform(-0.05, 0.01) * H
    vent_dx = W * rng.uniform(0.088, 0.130)
    v_scale = rng.uniform(0.8, 1.2)

    n_total = int(rng.integers(4, 11))
    n_perivent = int(rng.integers(2, min(6, n_total - 1)))
    n_juxta    = min(int(rng.integers(1, 4)), n_total - n_perivent)
    n_subcort  = n_total - n_perivent - n_juxta

    MS_SIG = float(rng.uniform(0.65, 0.80))

    # Periventricular lesions (oval, perpendicular to ventricle wall)
    for _ in range(n_perivent):
        side = rng.choice([-1, 1])
        vx = cx + side * vent_dx
        pv_angle = float(rng.uniform(0, 2 * np.pi))
        pv_dist = H * float(rng.uniform(0.05, 0.12)) * v_scale
        ly = vent_y + pv_dist * np.cos(pv_angle)
        lx = vx     + pv_dist * np.sin(pv_angle)
        lr   = H * float(rng.uniform(0.011, 0.028))
        lrx  = lr * float(rng.uniform(0.5, 1.0))
        ang  = float(np.degrees(pv_angle) + rng.uniform(-30, 30))
        les  = _ellipse(H, W, ly, lx, lr, lrx, angle_deg=ang)
        img  = _lerp(img, float(rng.uniform(MS_SIG - 0.05, MS_SIG + 0.05)),
                     _soft(les.astype(np.float32), 1.2))
        # Optional central vein (1px dark dot)
        if rng.random() < 0.40:
            cv = _ellipse(H, W, ly, lx, 1.5, 1.5)
            img = _lerp(img, float(rng.uniform(0.10, 0.25)),
                        _soft(cv.astype(np.float32), 0.5))

    # Juxtacortical lesions (GM-WM junction, slightly peripheral)
    skull_ry = H * 0.435
    skull_rx = W * 0.400
    R_WM = 0.832  # approximate cortex inner
    for _ in range(n_juxta):
        ja = float(rng.uniform(-np.pi, np.pi))
        jd = skull_ry * float(rng.uniform(R_WM * 0.85, R_WM * 1.05))
        ly = cy + jd * np.sin(ja) * (skull_ry / skull_rx)
        lx = cx + jd * np.cos(ja)
        lr = H * float(rng.uniform(0.009, 0.022))
        les = _ellipse(H, W, ly, lx, lr, lr * float(rng.uniform(0.6, 1.6)),
                       angle_deg=float(rng.uniform(0, 180)))
        img = _lerp(img, float(rng.uniform(MS_SIG - 0.05, MS_SIG + 0.05)),
                    _soft(les.astype(np.float32), 1.0))

    # Subcortical WM lesions
    for _ in range(n_subcort):
        la = float(rng.uniform(-np.pi, np.pi))
        ld = H * float(rng.uniform(0.04, 0.20))
        ly = vent_y + ld * np.cos(la)
        lx = cx     + ld * np.sin(la)
        lr = H * float(rng.uniform(0.008, 0.020))
        les = _ellipse(H, W, ly, lx, lr, lr * float(rng.uniform(0.7, 1.5)),
                       angle_deg=float(rng.uniform(0, 180)))
        img = _lerp(img, float(rng.uniform(MS_SIG - 0.05, MS_SIG + 0.05)),
                    _soft(les.astype(np.float32), 1.0))

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: Ischemic stroke
# ===========================================================================

def _brain_t2_stroke(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Ischemic stroke: wedge-shaped vascular territory T2 hyperintensity."""
    H, W = shape
    img = _brain_t2_base(shape, rng, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0
    skull_ry = H * 0.435

    # Choose vascular territory: MCA (most common), ACA, PCA
    territory = rng.choice(["MCA", "ACA", "PCA"], p=[0.60, 0.20, 0.20])
    side = rng.choice([-1, 1])
    STROKE_SIG = float(rng.uniform(0.70, 0.85))

    if territory == "MCA":
        # Large lateral territory (frontal + parietal + temporal)
        t_cy = cy + H * float(rng.uniform(-0.02, 0.06))
        t_cx = cx + side * W * float(rng.uniform(0.06, 0.14))
        t_ry = skull_ry * float(rng.uniform(0.55, 0.80))
        t_rx = t_ry * float(rng.uniform(0.70, 1.10))
        stroke_mask = _ellipse(H, W, t_cy, t_cx, t_ry, t_rx)
        # Exclude medial structures (MCA spares midline)
        exclude = _ellipse(H, W, cy, cx, skull_ry * 0.50, skull_ry * 0.40)
        stroke_mask = stroke_mask & ~exclude
    elif territory == "ACA":
        # Medial frontal territory
        t_cy = cy + H * float(rng.uniform(-0.12, -0.04))
        t_cx = cx + side * W * float(rng.uniform(0.01, 0.06))
        t_ry = skull_ry * float(rng.uniform(0.30, 0.50))
        t_rx = t_ry * float(rng.uniform(0.60, 0.90))
        stroke_mask = _ellipse(H, W, t_cy, t_cx, t_ry, t_rx)
    else:  # PCA
        # Posterior territory (occipital + posterior temporal)
        t_cy = cy + H * float(rng.uniform(0.04, 0.12))
        t_cx = cx + side * W * float(rng.uniform(0.04, 0.12))
        t_ry = skull_ry * float(rng.uniform(0.28, 0.45))
        t_rx = t_ry * float(rng.uniform(0.70, 1.20))
        stroke_mask = _ellipse(H, W, t_cy, t_cx, t_ry, t_rx)

    img = _lerp(img, STROKE_SIG, _soft(stroke_mask.astype(np.float32), 3.5))

    # Gyral swelling: slight cortical thickening (simulate as additional bright ring)
    gyral_swelling = np.zeros((H, W), dtype=np.float32)
    gyral_swelling[stroke_mask] = 1.0
    gyral_swelling = gaussian_filter(gyral_swelling, sigma=4.0)
    img = _lerp(img, STROKE_SIG + 0.05, gyral_swelling * 0.4)

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: Hydrocephalus
# ===========================================================================

def _brain_t2_hydrocephalus(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Communicating hydrocephalus: massive ventricles + transependymal edema."""
    H, W = shape

    # Build base with very enlarged ventricles
    img = _brain_t2_base(shape, rng, enlarged_ventricles=True, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0
    vent_y = cy + rng.uniform(-0.05, 0.01) * H
    vent_dx = W * rng.uniform(0.088, 0.130)
    v_scale = float(rng.uniform(2.0, 3.5))  # massive

    CSF_SIG   = float(rng.uniform(0.90, 0.96))
    EDEMA_SIG = float(rng.uniform(0.48, 0.58))  # periventricular T2 increase

    for side in [-1, 1]:
        vx = cx + side * vent_dx
        # Massively enlarged frontal horn
        fh = _ellipse(H, W, vent_y - H * 0.050, vx,
                      H * 0.085 * v_scale, W * 0.065 * v_scale)
        bdy = _ellipse(H, W, vent_y + H * 0.010, vx,
                       H * 0.075 * v_scale, W * 0.075 * v_scale)
        vent = (fh | bdy).astype(np.float32)
        img = _lerp(img, CSF_SIG, _soft(vent, 3.0))

        # Transependymal CSF seepage (bright halo around ventricles)
        halo = _ellipse(H, W, vent_y, vx,
                        H * 0.10 * v_scale, W * 0.09 * v_scale)
        img = _lerp(img, EDEMA_SIG, _soft(halo.astype(np.float32), 5.0) * 0.45)

    # Enlarged 3rd ventricle
    third = _ellipse(H, W, vent_y + H * 0.030, cx, H * 0.075, W * 0.020)
    img = _lerp(img, CSF_SIG, _soft(third.astype(np.float32), 2.0))

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: Microbleeds
# ===========================================================================

def _brain_t2_microbleeds(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Cerebral microbleeds: tiny very dark hemosiderin foci throughout WM."""
    H, W = shape
    img = _brain_t2_base(shape, rng, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0
    vent_y = cy + rng.uniform(-0.05, 0.01) * H
    skull_ry = H * 0.435

    n_bleeds = int(rng.integers(8, 21))
    for _ in range(n_bleeds):
        # Distributed throughout WM and GM-WM junction
        ba = float(rng.uniform(-np.pi, np.pi))
        bd = skull_ry * float(rng.uniform(0.15, 0.80))
        b_cy = vent_y + bd * np.cos(ba)
        b_cx = cx     + bd * np.sin(ba)
        b_r  = float(rng.uniform(1, 3))
        bleed = _ellipse(H, W, b_cy, b_cx, b_r, b_r)
        img = _lerp(img, float(rng.uniform(0.01, 0.04)),
                    _soft(bleed.astype(np.float32), 0.5))

    return img.clip(0.0, 1.0)


# ===========================================================================
# Pathology: Meningioma
# ===========================================================================

def _brain_t2_meningioma(shape: tuple, rng: np.random.Generator) -> np.ndarray:
    """Extra-axial meningioma: well-defined mass + brain edema + compression."""
    H, W = shape
    img = _brain_t2_base(shape, rng, slice_type="supratentorial")

    cy = H / 2.0
    cx = W / 2.0
    skull_ry = H * rng.uniform(0.420, 0.448)

    MENING_SIG = float(rng.uniform(0.45, 0.65))
    EDEMA_SIG  = float(rng.uniform(0.60, 0.76))

    # Extra-axial location: placed at brain surface, outside SAS
    ma = float(rng.uniform(-np.pi, np.pi))
    m_dist = skull_ry * float(rng.uniform(0.88, 0.96))  # near inner skull surface
    m_cy = cy + m_dist * np.sin(ma)
    m_cx = cx + m_dist * np.cos(ma)
    m_ry = H * float(rng.uniform(0.035, 0.065))
    m_rx = W * float(rng.uniform(0.050, 0.095))

    # Mass itself (iso/hypointense, well-defined)
    mening_mask = _ellipse(H, W, m_cy, m_cx, m_ry, m_rx,
                           angle_deg=float(np.degrees(ma)))
    img = _lerp(img, MENING_SIG, _soft(mening_mask.astype(np.float32), 1.5))

    # Surrounding brain edema (bright halo extending into WM beneath mass)
    edema_ry = m_ry * float(rng.uniform(1.8, 2.8))
    edema_rx = m_rx * float(rng.uniform(1.8, 2.8))
    edema_mask = _ellipse(H, W, m_cy, m_cx, edema_ry, edema_rx,
                          angle_deg=float(np.degrees(ma)))
    edema_only = edema_mask & ~mening_mask
    img = _lerp(img, EDEMA_SIG, _soft(edema_only.astype(np.float32), 3.5))

    # Local brain compression: shift nearby sulci (bright border effect)
    compress_alpha = np.zeros((H, W), dtype=np.float32)
    compress_alpha[mening_mask] = 1.0
    compress_alpha = gaussian_filter(compress_alpha, sigma=6.0)
    # Darkening effect at compressed brain surface
    img = img * (1.0 - compress_alpha * 0.15)

    return img.clip(0.0, 1.0)
