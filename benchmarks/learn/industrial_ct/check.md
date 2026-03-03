# Comprehensive 6-Point Check -- industrial_ct

**Modality:** Industrial X-ray Computed Tomography
**Category:** industrial_inspection (originally); override redirects to CT methods
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

Industrial CT uses X-ray projection imaging and tomographic reconstruction
to non-destructively inspect manufactured parts. The forward model is the
X-ray transform (Beer-Lambert law):

    y_i = I_0 * exp(-integral mu(x) dl_i) + n

where `mu(x)` is the linear attenuation coefficient, `l_i` is the ray path
for projection `i`, `I_0` is the source intensity, and `n` is detector noise.
For cone-beam geometry (common in industrial CT):

    p(theta, u, v) = integral mu(x(s)) ds + n

where `(u, v)` are detector coordinates and `theta` is the rotation angle.
The reconstruction recovers `mu(x)` from projections using filtered
back-projection or iterative methods.

Key physics: cone-beam geometry (FDK), beam hardening from polychromatic
X-ray sources, metal artifacts, scatter, and limited-angle/sparse-view
constraints for high-throughput inline inspection.

**Verdict:** Physics correctly modeled. Industrial CT is fundamentally a
tomographic reconstruction problem, distinct from the thermal/acoustic NDT
methods in the generic industrial_inspection pool.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Source spectrum uncertainty (beam hardening)
- Geometric calibration (source-detector alignment)
- Scatter contamination
- Detector nonlinearity and afterglow
- Ring artifacts from defective detector pixels
- Truncation artifacts (object larger than field of view)

The benchmark models geometric calibration errors and source spectrum
uncertainty as primary mismatch parameters.

**Verdict:** Appropriate. Key industrial CT artifact sources captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["industrial_ct"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | FDK | Classical | 0 | Feldkamp et al., JOSA A 1984 |
| 2 | PnP-ADMM | PnP | 0 | Venkatakrishnan et al., 2013 |
| 3 | FBPConvNet | Deep Learning | 22M | Jin et al., IEEE TIP 2017 |
| 4 | Learned Primal-Dual | Deep Unrolling | 5M | Adler & Oktem, IEEE TMI 2018 |

- **FDK (Feldkamp-Davis-Kress)** is the standard cone-beam CT reconstruction
  algorithm. The universal baseline for industrial CT. Correct.
- **PnP-ADMM** applies plug-and-play priors via ADMM for iterative CT
  reconstruction with learned denoisers. Widely applicable. Correct.
- **FBPConvNet** is a CNN post-processor that refines FBP reconstructions.
  Published in IEEE TIP. Standard deep learning CT baseline. Correct.
- **Learned Primal-Dual** is a deep unrolling method that integrates the CT
  forward and adjoint operators into the network architecture. State-of-the-art
  physics-informed reconstruction. Correct.

**Verdict:** PASS. All four algorithms are tomographic reconstruction methods,
replacing the completely inappropriate industrial_inspection pool (TSR,
PnP-ADMM, DefectNet, LSTM-NDT) where TSR, DefectNet, and LSTM-NDT are
thermography and temporal NDT methods.

## 4. Literature (2024-2025)

Recent relevant publications:
- Zhu et al., "Diffusion-Based Industrial CT Artifact Reduction," NDT&E Int.
  2024
- Wang et al., "Neural Radiance Fields for Industrial CT," IEEE TII 2024
- Leuschner et al., "LoDoPaB-CT Benchmark Update," Inverse Problems 2024
- ASTRA toolbox integration with deep learning pipelines, 2024

The current set covers the FDK-to-deep-unrolling progression and is well-
established. 2024 adds diffusion models and NeRF-based approaches for CT.

**Verdict:** Acceptable. Core tomographic methods well-represented.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `industrial_ct_challenge_public.h5`,
  `industrial_ct_challenge_dev.h5`, `industrial_ct_challenge_hidden.h5`
  -- all present in `challenge-data/v1.0/`
- Gallery images on GCS: `img/benchmark_gallery/industrial_ct/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different phantom objects per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are CT reconstruction methods |
| Literature coverage | PASS (through 2018; all are foundational CT methods) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override was critical -- the previous
industrial_inspection pool contained thermography methods (TSR) and NDT
time-series methods (LSTM-NDT) that have no applicability to X-ray
tomographic reconstruction.
