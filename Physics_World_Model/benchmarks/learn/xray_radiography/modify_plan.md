# Modify Plan: xray_radiography

**Date:** 2026-03-06

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical
- **Algorithms assigned:**

| Name                | Type               | Source                               |
|---------------------|--------------------|--------------------------------------|
| FBP                 | Classical          | Analytical baseline                  |
| TV-ADMM             | Compressed Sensing | Rudin et al., Physica D 60, 259 (1992) + ADMM |
| FBPConvNet          | Deep Learning      | Jin et al., IEEE TIP 26, 4509 (2017) |
| RED-CNN             | Deep Learning      | Chen et al., IEEE TMI 36, 2524 (2017) |

## Assessment

**Acceptable — no code changes needed.**

The `medical` category pool is CT-centric, which is a reasonable fit for medical X-ray radiography.

1. **FBP** — While FBP is strictly a tomographic reconstruction algorithm, for radiography it represents the standard image processing baseline (Wiener filter deconvolution, ramp filter in Fourier domain). Acceptable as a classical baseline.
2. **TV-ADMM** — Total variation denoising is directly applicable to Poisson noise reduction in radiographs. This is actually a better fit for 2D radiography than for full 3D CT reconstruction. GOOD FIT.
3. **FBPConvNet** — A post-processing CNN operating on image-domain data. Applicable to radiograph enhancement. Jin et al. 2017 is a real, well-cited paper. GOOD FIT.
4. **RED-CNN** — Chen et al., IEEE TMI 2017 is an encoder-decoder CNN designed specifically for low-dose X-ray imaging. This is a more natural fit for 2D radiography denoising than 3D CT. EXCELLENT FIT.

### Note on RED-CNN vs. Learned Primal-Dual

Previous versions of this modality had Learned Primal-Dual (Adler & Oktem, 2018) instead of RED-CNN. RED-CNN is a better choice for radiography because:
- RED-CNN operates on 2D images (single frame denoising)
- Learned Primal-Dual is a projection-to-volume network designed for 3D CT
- Chen et al. 2017 explicitly addresses 2D X-ray denoising

The carrier routing does not reroute `("medical", "X-ray")`, so this modality correctly stays in the `medical` pool.

## Proposed Changes

No code changes needed. The current algorithm set is appropriate.

**Priority:** NONE — no changes needed.
