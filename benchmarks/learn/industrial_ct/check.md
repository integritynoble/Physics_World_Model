# Comprehensive 6-Point Check — Industrial CT (X-ray Computed Tomography for NDT)

**URL:** https://pwm.platformai.org/benchmark/industrial_ct
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Industrial X-ray Computed Tomography (NDT/NDE)

**Physical principle:** Industrial CT uses polychromatic X-rays to non-destructively inspect manufactured components (metal castings, composites, electronics, additive-manufactured parts) for internal defects (porosity, cracks, inclusions). The Beer-Lambert law governs X-ray attenuation: the line integral of the linear attenuation coefficient μ(x) along each ray equals the log of the intensity ratio I/I₀. A detector array captures projections at multiple rotation angles, and tomographic reconstruction (FBP or iterative) recovers the 3D attenuation map μ(x,y,z) from which defects are segmented. Industrial CT operates at higher energies (100 keV – 15 MeV) and doses than medical CT, with sub-micron to millimeter voxel sizes depending on object size.

**Forward model:**
```
p(s, θ) = −ln[I(s,θ)/I₀] = ∫_ray(s,θ) μ(x) dl  +  η_log

Polychromatic (beam hardening):
  I(s,θ) = ∫ w(E) · I₀(E) · exp[−∫ μ(x,E) dl] dE

where:
  p(s, θ)      — log-attenuation projection (sinogram) at detector position s, angle θ
  μ(x)         — linear attenuation coefficient [cm⁻¹] (object to recover)
  I₀, I        — incident and transmitted X-ray intensities
  w(E)         — normalized source energy spectrum (polychromatic)
  η_log        — log-domain quantum noise (approximately Gaussian after log)
```

**Inverse problem:** Recover the 3D attenuation volume μ(x,y,z) from the complete or limited-angle sinogram, then detect and characterize defects (porosity, cracks) within the reconstruction.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(polychromatic X-ray tube/linac) → F(metal/composite workpiece) → D(flat-panel detector)

**Key mismatch parameters:**
- `beam_hardening`: polychromaticity degree; nominal 10% cupping artifact, perturbed 40% (heavy steel component)
- `scatter_fraction`: scattered X-ray contamination; nominal 5%, perturbed 20% (large object, poor scatter rejection)
- `angular_coverage`: total rotation angle; nominal 360°, perturbed 180° (limited-angle CT, missing data)
- `voxel_snr`: signal-to-noise per voxel in reconstruction; nominal 30 dB, perturbed 15 dB (low-dose scan)

**Dataset format:**
- `x_true: (H, W)` — ground-truth 2D slice of attenuation map μ(x,y) or binary defect mask
- `y: (N_angles, N_det)` — sinogram (N_angles projection views, N_det detector pixels per view)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| FBP (Filtered Back-Projection) | Classical | Kak & Slaney, "Principles of Computerized Tomographic Imaging," IEEE 1988 | Fast analytic reconstruction; standard baseline for industrial CT |
| SART (Simultaneous Algebraic Reconstruction) | Classical iterative | Andersen & Kak, Ultrason. Imaging 6:81 (1984) | Algebraic iterative method suitable for limited-angle/sparse-view CT |
| TV-minimization (TVAL3) | PnP / compressed sensing | Li et al., SIAM J. Sci. Comput. 32:2832 (2010) | Total variation minimization for sparse-view CT; excellent for defect detection |
| FBPConvNet | Deep Learning | Jin et al., IEEE Trans. Image Process. 26:4509 (2017) | Post-processing CNN on FBP output; widely adopted for CT denoising/artifact removal |
| SwinIR-CT / Transformer | Transformer | Hu et al., IEEE Trans. Instrum. Meas. 72:1 (2023) | Swin transformer for limited-angle industrial CT reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Leuschner et al. (2024)** "Quantitative comparison of deep learning-based image reconstruction methods for low-dose industrial CT," *NDT E Int.* — systematic benchmark of FBPConvNet, U-Net, and diffusion models on real industrial castings.
2. **Tan et al. (2024)** "Score-based diffusion models for sparse-view industrial CT reconstruction," *IEEE Trans. Instrum. Meas.* — diffusion model prior outperforming TV and CNN methods at <30 projection views.
3. **Winkler et al. (2023)** "Deep learning CT image reconstruction for sub-voxel defect detection in additive manufacturing," *Addit. Manuf.* — U-Net architecture detecting 50 µm porosity defects at reduced X-ray dose.
4. **Dong et al. (2024)** "Physics-informed deep learning for beam hardening correction in polychromatic CT," *Med. Phys.* — PINN enforcing Beer-Lambert polychromatic model reduces cupping artifacts by 60%.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/industrial_ct_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/industrial_ct_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/industrial_ct_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/industrial_ct/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Industrial CT is correctly formulated as a Beer-Lambert line-integral inverse problem with polychromatic beam-hardening corrections, distinguishing it from medical CT through higher energy, larger contrast variations, and stricter defect detection requirements. The algorithm routing appropriately spans FBP, SART, TV-minimization, FBPConvNet, and transformer-based methods reflecting the realistic progression from analytic to learned reconstruction in industrial NDT. The four mismatch parameters — beam hardening, scatter fraction, angular coverage, and voxel SNR — capture the dominant degradation modes in real industrial inspection scenarios from aluminum castings to fiber-reinforced composites.

---
*Comprehensive 6-point check by deep-check pipeline v3*
