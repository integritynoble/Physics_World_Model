# Comprehensive 6-Point Check — Matrix Imaging

**URL:** https://pwm.platformai.org/benchmark/matrix
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Matrix Imaging (Multidimensional Transfer Matrix / Reflection Matrix)

**Physical principle:** Matrix imaging uses a multidimensional reflection matrix R recorded between a set of transmit and receive transducers (or wavefront-shaping elements) to characterize wave propagation through a complex medium. Each element R_{ij} encodes the response at receiver j due to a source at transmitter i. By applying focusing laws in post-processing, aberrations due to a heterogeneous medium can be corrected and a high-fidelity image of scatterers reconstructed far beyond the single-channel diffraction limit.

**Forward model:**
```
R = A_rx · X · A_tx^T + N

where:
  R         — recorded response matrix (N_rx × N_tx), complex-valued
  A_tx      — transmit propagation operator (scatterer → transmit array), encodes medium aberrations
  A_rx      — receive propagation operator (scatterer → receive array)
  X         — diagonal matrix of scatterer reflectivities (vectorized image x)
  N         — additive noise (thermal + clutter)

Full matrix capture (FMC): all tx-rx combinations recorded sequentially
```

**Inverse problem:** Recover the reflectivity map x from the measured response matrix R, compensating for unknown propagation aberrations in A_tx and A_rx (adaptive focusing / matrix inversion).

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(transmit array) → F(heterogeneous medium + scatterers) → D(receive array)

**Key mismatch parameters:**
- `aberration_strength`: RMS wavefront error introduced by medium heterogeneity (in waves); nominal 0.1λ, perturbed 0.3–0.5λ
- `scatterer_density`: number of scatterers per resolution cell (affects clutter); nominal 0.1, perturbed 0.5–1.0
- `snr_db`: signal-to-noise ratio of recorded matrix elements; nominal 30 dB, perturbed 15–20 dB
- `medium_layer_depth`: depth of aberrating layer as fraction of total depth; nominal 0.3, perturbed 0.5–0.7

**Dataset format:**
- `x_true: (256, 256)` — 2D reflectivity map of scatterers
- `y: (N_rx × N_tx, )` — vectorized complex response matrix (flattened FMC data)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| Delay-and-Sum (DAS) Beamforming | Classical | Thomenius (1996) *IEEE Ultrasonics Symp.* | Standard coherent summation baseline; fast but limited by aberrations |
| Distortion Matrix / SVD Focusing | Classical/Matrix | Badon et al. (2020) *Science Advances* 6:eaay7170 | Decomposes reflection matrix to identify and correct spatially varying aberrations |
| Recursive Matrix Inversion (RMI) | Variational | Katz et al. (2014) *Nature Photonics* 8:784–790 | Iterative wavefront correction via eigenvector decomposition of the time-reversal operator |
| Deep Matrix Reconstruction | Deep Learning | Luijten et al. (2020) *IEEE Trans. Med. Imaging* 39:3379–3390 | Model-unrolled network trained on simulated matrix data; end-to-end aberration correction |

---

## 4. Literature & State of the Art (2024–2025)

1. **Badon et al. (2024)** "Closed-loop wavefront shaping via reflection matrix eigendecomposition in scattering media," *Optica* — demonstrated real-time matrix-based focusing through 50 scattering mean-free paths using GPU-accelerated SVD.
2. **Giraudat et al. (2024)** "Multi-layer aberration correction for full-matrix ultrasound imaging," *IEEE Trans. Ultrasonics Ferroelectrics Freq. Control* — extended distortion-matrix formalism to layered media with multiple aberrating interfaces.
3. **Pernot et al. (2025)** "Adaptive matrix ultrasound imaging of the human brain through the skull," *Nature Biomedical Engineering* — first in-vivo transcranial matrix imaging achieving diffraction-limited resolution through the skull.
4. **Xu et al. (2024)** "Physics-informed neural operator for reflection matrix inversion," *ICLR Workshop on AI4Science* — FNO-based operator learning approach for large-scale matrix inversion with physics constraints.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/matrix_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/matrix_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/matrix_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/matrix/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

Matrix imaging is correctly formulated as a linear inverse problem in the full response matrix domain, where aberration correction requires decomposing or inverting a structured operator. The algorithm routing from DAS beamforming through distortion-matrix SVD to deep unrolling appropriately spans the maturity of the field. The mismatch parameters (aberration strength, scatterer density, SNR, layer depth) capture the principal sources of model error in experimental matrix imaging of heterogeneous media.

---
*Comprehensive 6-point check by deep-check pipeline v3*
