# Comprehensive 6-Point Check — Muon Tomography

**URL:** https://pwm.platformai.org/benchmark/muon_tomo
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Muon Tomography (Cosmic-Ray Muon Scattering Tomography)

**Physical principle:** Cosmic-ray muons are high-energy charged particles (mean energy ~3 GeV at sea level) that penetrate dense matter. When passing through material, muons undergo multiple Coulomb scattering, with the RMS scattering angle depending on the material's radiation length X₀: θ_rms ∝ (L/X₀)^{1/2} / p, where L is path length and p is momentum. High-Z materials (lead, uranium) have short radiation lengths and cause large scattering angles, enabling discrimination of nuclear materials in containers or voids in volcanoes/pyramids.

**Forward model:**
```
θ_scat ~ Normal(0, σ_θ²)

σ_θ² = (13.6 MeV / (β·c·p))² · (L / X₀) · [1 + 0.038·ln(L/X₀)]²

Radon-transform-like accumulation:
P(x) = integral of θ_scat contributions along projected path

For PoCA (Point of Closest Approach):
scattering point estimate: r_PoCA = midpoint of closest approach
                             between incoming and outgoing track segments
```

**Inverse problem:** Reconstruct the 3D density/radiation-length distribution X₀(r) of the target object from a set of N muon track pairs (incoming direction, outgoing direction, momentum estimate).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(cosmic muon flux) → F(target object density distribution) → D(muon tracking detectors)

**Key mismatch parameters:**
- `detector_resolution_mrad`: angular resolution of tracking detectors; nominal 1 mrad, perturbed 3–5 mrad
- `muon_flux_rate`: muons per cm²/min at detector; nominal 1.0, perturbed 0.3–0.5 (exposure time)
- `momentum_uncertainty`: fractional momentum measurement error; nominal 0.10, perturbed 0.25–0.40
- `object_density_gcm3`: mean density of target (affects scattering rate); nominal 2.5 g/cm³, perturbed 7–11 g/cm³

**Dataset format:**
- `x_true: (256, 256)` — 2D slice of material density / radiation-length map
- `y: (N_muons, 6)` — muon track parameters: (x_in, y_in, θ_in, φ_in, θ_out, φ_out) per muon

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| PoCA (Point of Closest Approach) | Classical | Borozdin et al. (2003) *Nature* 422:277 | Foundational muon tomography reconstruction; estimates scattering points via track intersection |
| Maximum Likelihood / MLEM (ML-EM scattering) | Classical/Iterative | Schultz et al. (2007) *IEEE Trans. Image Processing* 16:1985–1993 | Maximum-likelihood EM reconstruction incorporating full scattering statistics |
| Filtered Back-Projection adapted for muons | Variational | Pesente et al. (2009) *Nucl. Instr. Meth. A* 604:738–746 | Analytical FBP-style inversion adapted for muon scattering geometry |
| Deep Muon Tomography (CNN/GNN) | Deep Learning | Gonçalves et al. (2022) *Front. Phys.* 10:875571 | Graph neural network processing individual muon trajectories for direct density estimation |

---

## 4. Literature & State of the Art (2024–2025)

1. **Saracino et al. (2024)** "Muographic imaging of volcanic conduit structure at Stromboli," *Nature Communications* — used cosmic-ray muon tomography with 30-day exposure to image the shallow conduit system of an active volcano at ~10 m resolution.
2. **Bonomi et al. (2024)** "Deep learning reconstruction for nuclear waste container inspection by muon scattering tomography," *Ann. Nucl. Energy* — CNN replacing PoCA with >40% improvement in material discrimination accuracy for simulated spent fuel containers.
3. **Bonechi et al. (2025)** "Machine-learning-enhanced muon tomography for cultural heritage 3D imaging," *J. Cultural Heritage* — applied ML-corrected muon flux reconstruction to image hidden cavities in the Egyptian pyramids.
4. **Thomay et al. (2024)** "Momentum-resolved muon scattering tomography with improved material discrimination," *JINST* — demonstrated that including muon momentum measurements reduces false-alarm rate for high-Z material detection by 60%.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/muon_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/muon_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/muon_tomo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/muon_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Muon tomography is correctly formulated as a statistical inverse problem where the observable (scattering angle distribution) depends on the integrated radiation length along each muon track, and the challenge is density reconstruction from sparse, noisy track data. The algorithm routing from PoCA through MLEM to deep GNN reconstruction appropriately spans the state of the art. The mismatch parameters (detector resolution, muon flux, momentum uncertainty, object density) capture the dominant experimental limitations in real muon tomography deployments.

---
*Comprehensive 6-point check by deep-check pipeline v3*
