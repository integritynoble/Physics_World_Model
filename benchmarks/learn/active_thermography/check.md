# Comprehensive 6-Point Check — Active Thermography (IR)

**URL:** https://pwm.platformai.org/benchmark/active_thermography
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Active Thermography (IR NDT)

**Physical principle:** Active thermography is a non-destructive testing technique in which a controlled heat stimulus (pulsed flash lamp, lock-in heating, or step heating) is applied to a test specimen. The resulting surface temperature evolution is captured by an infrared camera. Subsurface defects (delaminations, voids, inclusions) impede heat diffusion and produce local thermal anomalies in the temporal temperature profile. Recovery of defect maps requires inverting the heat diffusion equation with respect to anomalous regions.

**Forward model:**
```
T(x,y,t) = T_0 + Q/(ρ c_p) * G_D(x,y,t) * (1 + defect_perturbation(x,y,d))

G_D(x,y,t) = (4πDt)^{-1} exp(-(x²+y²)/4Dt)    [surface diffusion kernel]
D = k / (ρ c_p)                                   [thermal diffusivity, m²/s]

Discrete form:
  y = A(D) x + n
  y ∈ R^{H × W × T}   — IR image time sequence
  x ∈ R^{H × W}       — defect contrast/depth map (ground truth)
  A(D)                 — heat diffusion forward operator
  n                    — detector noise (NETD ~ 20 mK)
```

**Inverse problem:** Recover the subsurface defect map x (depth, size, contrast) from the IR thermal image sequence y by inverting or analysing the spatiotemporal heat diffusion response.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(IR heat source) → D(IR camera)

**Key mismatch parameters:**
- `emissivity_error` (e_e): surface emissivity deviation; nominal 0.95, perturbed 0.96
- `heat_source_power_drift` (h_s): flash lamp energy variation; nominal 1.0×, perturbed 1.02×
- `background_temperature` (b_t): ambient temperature shift; nominal 25.0°C, perturbed 26.0°C
- `integration_time_offset` (i_t): IR camera gate delay; nominal 0.0 s, perturbed 0.02 s

**Dataset format:**
- `x_true: (H, W)` — 2D defect depth/contrast map (ground truth)
- `y: (H, W, T)` — IR image time sequence; H×W spatial pixels, T time frames post-flash
- `H_ideal: (H*W*T, H*W)` — linearised heat diffusion operator (vectorised form)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| TSR | Classical | Shepard et al., SPIE Thermosense 2003 | Thermographic Signal Reconstruction; canonical pulsed thermography method |
| Thermography-FT | Classical | Maldague & Marinetti, J. Appl. Phys. 1996 | Lock-in Fourier transform phase/amplitude analysis |
| PnP-ADMM | Plug-and-Play | Venkatakrishnan et al., IEEE GlobalSIP 2013 | Regularised iterative inversion with learned denoising prior |
| DefectNet | Deep Learning | Liu et al., NDT&E Int. 2021 | U-Net for pulsed thermography defect detection and localisation |
| LSTM-NDT | Recurrent DL | Fang et al., Measurement 2022 | LSTM for temporal thermography sequence analysis |
| Inspection-ViT | Transformer | — | Vision Transformer for NDT inspection image classification and regression |

---

## 4. Literature & State of the Art (2024–2025)

1. **Deep CNN for pulsed thermography defect sizing** (2024): Multi-scale CNN architecture trained on synthetic + experimental CFRP data; outperforms TSR for shallow delaminations below 0.5 mm depth.
2. **Physics-informed LSTM for lock-in thermography** (2024): Embeds the heat equation as a recurrent prior; improves depth resolution for CFRP panels at multiple excitation frequencies.
3. **Transfer learning for composite NDT** (Sanchez-Lengeling et al., 2023 applied 2024): Domain adaptation from synthetic pulsed thermography data to experimental measurements; reduces training data requirements by 80%.
4. **Diffusion model for thermal defect reconstruction** (2025): Score-based diffusion posterior sampling for thermography; captures complex defect geometry distributions in aerospace panels.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/active_thermography_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/active_thermography_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/active_thermography_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/active_thermography/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses the `industrial_inspection` category pool (10 methods: TSR, Thermography-FT, PnP-ADMM, PnP-TV, DefectNet, U-Net-Thermal, LSTM-NDT, Inspection-ViT, DiffusionNDT, ScoreNDT). TSR (Shepard 2003) is the canonical pulsed thermography method, confirming domain correctness. The four mismatch parameters cover the key IR NDT calibration uncertainties: emissivity, heat source power, background temperature, and integration timing offset. No code changes are required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
