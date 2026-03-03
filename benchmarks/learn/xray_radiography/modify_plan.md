# Modify Plan: xray_radiography

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical
- **Algorithms assigned:**

| Name                | Type           | Source                           |
|---------------------|----------------|----------------------------------|
| FBP                 | Classical      | Analytical baseline              |
| PnP-ADMM           | PnP            | Venkatakrishnan et al., 2013     |
| FBPConvNet          | Deep Learning  | Jin et al., IEEE TIP 2017        |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018     |

## Assessment

**Acceptable -- no code changes needed.**

The `medical` category pool is CT-centric, which is a reasonable fit for medical X-ray radiography. The key considerations:

1. **FBP** -- While FBP is a tomographic reconstruction algorithm and radiography is a single-projection modality, "FBP" is commonly used as shorthand for backprojection-based analytical inversion. For a 2D radiograph the inverse problem is deconvolution/scatter correction rather than full tomographic reconstruction, but FBP-style filtering is still a legitimate classical baseline. Acceptable.
2. **PnP-ADMM** -- Generic and well-suited to any linear inverse problem including scatter estimation and noise reduction in radiographs. Good fit.
3. **FBPConvNet** -- A post-processing CNN that refines FBP output. Applicable to radiograph enhancement. Jin et al. 2017 is a real, well-cited paper. Good fit.
4. **Learned Primal-Dual** -- An unrolled optimization network. While originally designed for CT, the architecture generalizes to any linear forward model. Adler & Oktem 2018 is a real, well-cited paper. Acceptable fit.

The carrier routing does not reroute `("medical", "X-ray")`, so this modality correctly stays in the `medical` pool. The algorithms are all from the CT/X-ray reconstruction literature and are reasonable for medical radiography. The leaderboard PSNR/SSIM scores come from the `medical` pool which uses CT benchmarks -- this is a minor mismatch in absolute score values but not in algorithm selection.

## Proposed Changes

No code changes needed.

The algorithm pool is appropriate. If greater specificity were desired in the future, one could add a dedicated radiography pool with scatter-correction-specific algorithms (e.g., anti-scatter grid simulation, dual-energy decomposition), but the current assignment is not wrong.
