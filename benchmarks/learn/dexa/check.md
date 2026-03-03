# Comprehensive 6-Point Check — DEXA

**URL:** https://pwm.platformai.org/benchmark/dexa
**Check Date:** 2026-03-03
**Status:** FIXED (previously CRITICAL — wrong forward model)

---

## 1. Physics & Forward Model

**Modality:** Dual-Energy X-ray Absorptiometry (DEXA/DXA)

**Physical principle:** DEXA measures bone mineral density (BMD) by acquiring two X-ray projection images at different energies (typically ~40 keV and ~70 keV). The differential attenuation of bone and soft tissue at two energies allows material decomposition — separating bone from surrounding tissue in a 2D projection geometry.

**Forward model (dual-energy projection):**
```
y_low(i,j)  = μ_bone(E_low)  · t_bone(i,j) + μ_tissue(E_low)  · t_tissue(i,j)
y_high(i,j) = μ_bone(E_high) · t_bone(i,j) + μ_tissue(E_high) · t_tissue(i,j)
```
Where:
- `t_bone(i,j)` = bone thickness/density map (quantity of interest)
- `t_tissue(i,j)` = soft tissue thickness map
- `μ_bone(E)`, `μ_tissue(E)` = energy-dependent mass attenuation coefficients
- `y` = log-attenuation images at each energy

**Inverse problem:** Material decomposition — recover `t_bone` and `t_tissue` from the two measured projections. This is a 2×2 linear system per pixel, but noise amplification makes the inverse ill-conditioned.

**Key difference from CT:** DEXA is NOT tomographic. It produces 2D projections, not sinograms. There is no Radon transform or angular sampling involved.

**Previous issue (FIXED):** The dataset was incorrectly using the `radon` runner (CT sinograms with 180 angles), producing `y: (180, 182)` sinogram measurements from 3D volumes `x_true: (128, 128, 64)`. This has been corrected to use the `dual_energy` runner producing `y: (256, 256, 2)` dual-energy projections from `x_true: (256, 256, 2)` bone+tissue maps.

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** `Λ(E₁,E₂) → Π(proj) → D(g, η₁)`

The mismatch parameters for DEXA involve:
- Energy calibration errors (E₁, E₂ shifted from ideal)
- Detector gain/offset drift
- Beam hardening effects
- Scatter fraction variations
- Patient positioning/motion

**Benchmark structure (3 tiers):**
- **Public:** Ground truth `x_true` included; for algorithm development
- **Dev:** No `x_true`; for validation (server-side scoring)
- **Hidden:** Blocked from download; for final leaderboard ranking

**Data format:**
- `x_true: (256, 256, 2)` — bone density (ch0) + soft tissue thickness (ch1)
- `y: (256, 256, 2)` — log-attenuation at low energy (ch0) and high energy (ch1)
- `H_ideal: (2, 2)` — attenuation coefficient matrix [[μ_bone_low, μ_tissue_low], [μ_bone_high, μ_tissue_high]]

## 3. Reconstruction Methods & Leaderboard

**Current algorithms (4 entries):**

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Dual-Energy Subtraction (DES) | Classical | Lehmann et al., Med. Phys. 1981 | ✓ Foundational DEXA decomposition |
| PnP-ADMM | PnP | Venkatakrishnan et al., 2013 | ✓ General-purpose with material decomposition prior |
| Butterfly-Net | Deep Learning | Li et al., SIAM J. Sci. Comput. 2020 | ✓ Physics-informed dual-energy decomposition |
| DECT-MULTRA | Deep Unrolling | Gong et al., IEEE TMI 2020 | ✓ Model-based deep learning for multi-material decomposition |

**Assessment:** All 4 algorithms are appropriate for the dual-energy decomposition inverse problem. The classical DES provides a strong baseline; Butterfly-Net and DECT-MULTRA are specifically designed for dual-energy material decomposition.

## 4. Literature & State of the Art (2024–2025)

Recent advances in DEXA and dual-energy decomposition:

1. **E2E-DEcomp** (2024): End-to-end model-based deep learning for DECT material decomposition. Incorporates spectral model knowledge into training loss. (arXiv 2406.00479)
2. **Unsupervised DE decomposition** (2024): Combines iterative decomposition with GAN-based image prior. Reduced noise SD by 97% vs direct inversion. (PMC 11489026)
3. **Updated DXA Practice Guideline** (2024): New ISCD guidelines for DXA technology addressing technical and clinical advances.
4. **Opportunistic osteoporosis screening** (2025): Deep learning for BMD estimation from routine CT, demonstrating cross-modality transfer.

The current algorithm set covers the key approaches well. Future additions could include E2E-DEcomp or unsupervised decomposition methods.

## 5. Local Dataset & GCS Status

**GCS datasets (verified):**
- `dexa_challenge_public.h5` — 2,342 KB ✓ (3 samples, dual-energy format)
- `dexa_challenge_dev.h5` — 1,357 KB ✓ (3 samples, x_true stripped)
- `dexa_challenge_hidden.h5` — 1,372 KB ✓ (3 samples)

**Ground truth source:** Simulated DEXA phantoms with anatomically-inspired bone structures (vertebral bodies, pelvis, femoral heads) and soft tissue background. Each tier uses different random seeds to prevent data leakage.

**Data verification:**
- Public: `x_true (256,256,2)`, `y (256,256,2)`, `H_ideal (2,2)` ✓
- Dev: No `x_true` ✓ (stripped via strip_dev_ground_truth.py)
- Hidden: Blocked from download via GCS proxy ✓

## 6. Comprehensive Assessment & Recommendations

**Status:** PASS (after fixes)

**Fixes applied in this check:**
1. Added `dual_energy` runner type to challenge dataset generator
2. Added `_forward_dual_energy()` forward model (Beer-Lambert dual-energy projection)
3. Added `_make_dexa_phantom()` with anatomical bone/tissue structure generation
4. Added `_VARIANT_TO_RUNNER` override: `dexa → dual_energy`
5. Added `_VARIANT_SIGNAL_SHAPE` override: `dexa → [256, 256, 2]`
6. Regenerated all 3 tiers of challenge datasets on GCS
7. Stripped x_true from dev tier

**Remaining opportunities:**
- Public dataset could benefit from real clinical DEXA scans (currently simulated phantoms)
- Consider adding E2E-DEcomp as a 5th algorithm when available
- 3D DEXA (volumetric BMD from fan-beam DEXA) is an emerging technique

---
*Comprehensive 6-point check by deep-check pipeline v3*
