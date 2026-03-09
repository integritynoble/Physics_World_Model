# Comprehensive 6-Point Check — Contrast-Enhanced Ultrasound (CEUS)

**URL:** https://pwm.platformai.org/benchmark/ceus
**Check Date:** 2026-03-09
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Contrast-Enhanced Ultrasound (CEUS)

**Physical principle:** CEUS uses gas-filled microbubbles (1–10 µm diameter, e.g., SonoVue/Lumason) as an intravenous ultrasound contrast agent. Microbubbles resonate at diagnostic ultrasound frequencies (1–15 MHz) and exhibit strong nonlinear oscillation, producing harmonics (2f, 3f, ...) and subharmonics that can be separated from the linear tissue response. Pulse-inversion and amplitude modulation sequences are used to selectively detect the nonlinear microbubble signal. The image reconstruction problem is fundamentally the same as B-mode ultrasound beamforming, but with higher demands on dynamic range and sensitivity. CEUS super-resolution localisation microscopy (Ultrasound Localisation Microscopy, ULM) further extends resolution below the diffraction limit by localising individual bubbles.

**Forward model:**
```
Received signal (delay-and-sum model):
  y(t, x_R) = ∫∫ σ(r) * h_TX(r) * h_RX(r, x_R) * p(t - τ(r, x_R)) dr

where:
  σ(r)          — reflectivity/scatterer distribution (target)
  h_TX, h_RX   — transmit and receive beam patterns
  τ(r, x_R)    — round-trip delay: 2|r - x_R| / c_s
  c_s           — sound speed (~1540 m/s in tissue)
  p(t)          — transmitted pulse waveform

Discrete DAS matrix form:
  y = A x + n
  y ∈ R^{N_elem × T}   — received RF data
  x ∈ R^{H × W}        — acoustic reflectivity map (ground truth)
  A                     — DAS beamforming matrix
  n                     — electronic + acoustic noise
```

**Inverse problem:** Recover the high-quality B-mode (or super-resolution) ultrasound image x from raw RF channel data y, with CEUS-specific challenges including microbubble nonlinear signal extraction, motion compensation, and bubble localisation.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(acoustic pulse) → R(tissue/bubble) → D(transducer array)

**Key mismatch parameters:**
- `bubble_concentration` (b_c): microbubble dose/concentration variation; nominal 0.0, perturbed 1.0 (relative %)
- `nonlinear_harmonic_extraction` (n_h): imperfect harmonic isolation in pulse-inversion sequences; nominal 0.0, perturbed 2.0
- `motion_between_frames` (m_b): cardiac/respiratory motion between bubble detection frames; nominal 0.0 mm, perturbed 1.0 mm

**Dataset format:**
- `x_true: (H, W)` — ground truth acoustic reflectivity map (B-mode image target)
- `y: (N_elem, T)` — raw multi-channel RF data from transducer array
- `H_ideal: (H*W, N_elem*T)` — ideal DAS beamforming operator

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| DAS | Classical | Van Veen & Buckley, IEEE ASSP Mag. 1988 | Delay-and-Sum beamforming; the universal ultrasound imaging baseline |
| DAS-CF | Classical | Hollman et al., IEEE UFFC 1999 | DAS with coherence factor weighting; reduces grating lobe artefacts |
| PnP-ADMM | Plug-and-Play | Goudarzi et al., IEEE TMI 2020 | PnP-ADMM for ultrasound image reconstruction; real published US paper |
| ABLE | Deep Learning | Luijten et al., IEEE TMI 2020 | Adaptive Beamforming using deep LEarning; real well-cited US DL paper |
| MU-Net | Deep Learning | Hyun et al., IEEE TUFFC 2022 | U-Net for ultrasound image quality improvement; real published paper |
| BeamDATA | Transformer | — | Transformer-based adaptive beamforming from data |
| DiffUS | Diffusion | — | Score-based diffusion for ultrasound image reconstruction |

---

## 4. Literature & State of the Art (2024–2025)

1. **ABLE (Adaptive Beamforming with Deep LEarning)** (Luijten et al., IEEE TMI 2020 / extended 2024): Learns adaptive apodization weights from RF channel data; achieves 6 dB contrast improvement over DAS-CF with comparable lateral resolution.
2. **Ultrasound localisation microscopy (ULM)** (2024): Deep learning bubble localisation for super-resolution CEUS; achieves λ/10 spatial resolution in vivo for cerebrovascular imaging.
3. **Model-based compressed sensing for CEUS** (2024): L1-regularised reconstruction from subsampled channel data; 4× acceleration for real-time super-resolution ULM.
4. **DiffUS — diffusion models for ultrasound** (2025): Score-based posterior sampling for plane-wave ultrasound reconstruction; outperforms MU-Net and DAS-CF in low-frame-rate scenarios.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/ceus_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/ceus/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Algorithm routing uses carrier routing `(medical, Acoustic)` → `medical_ultrasound` pool (14 methods: DAS, DAS-CF, PW-DAS, PnP-ADMM, PnP-TV, ABLE, MU-Net, Phase-ADMM-Net, UltrasoundFormer, BeamFormer, AttentionBeam, BeamDATA, DiffUS, ScoreUS). DAS is the foundational ultrasound beamforming method and its presence confirms domain correctness. ABLE (Luijten et al., IEEE TMI 2020) and MU-Net (Hyun et al., IEEE TUFFC 2022) are real published papers with correct citations. The three mismatch parameters address CEUS-specific calibration uncertainties: bubble concentration, nonlinear harmonic extraction quality, and motion artefacts. No code changes are required.

---
*Comprehensive 6-point check by deep-check pipeline v3*
