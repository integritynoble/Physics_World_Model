# Modify Plan -- coded_exposure

**Date:** 2026-03-03
**Category:** computational_photography | **Carrier:** Photon | **Score key:** computational_photography

## Current Algorithms (from catalog)

| # | Algorithm    | Type          | Source                          |
|---|--------------|---------------|---------------------------------|
| 1 | Wiener-Deconv| Classical     | Analytical baseline             |
| 2 | PnP-FFDNet   | PnP           | Zhang et al., 2017              |
| 3 | HDR-CNN      | Deep Learning | Eilertsen et al., ACM TOG 2017  |
| 4 | Uformer      | Transformer   | Wang et al., CVPR 2022          |

## Assessment

### Are algorithms domain-appropriate?
MOSTLY YES, with one notable mismatch. Coded exposure (flutter shutter) is a computational photography technique that modulates the camera's shutter timing during a single exposure to make the motion blur PSF more invertible.

- Wiener-Deconv (Wiener Deconvolution): EXCELLENT. This is THE classical method for coded exposure -- the entire point of flutter shutter (Raskar et al., SIGGRAPH 2006) is to create a broadband PSF that is well-conditioned for Wiener deconvolution.
- PnP-FFDNet: GOOD. PnP with FFDNet denoiser is a natural fit for image deblurring/deconvolution tasks. FFDNet handles the noise amplification inherent in deconvolution.
- HDR-CNN: POOR FIT. HDR-CNN (Eilertsen et al., ACM TOG 2017) is specifically for HDR image reconstruction from LDR inputs (inverse tone mapping). While coded exposure and HDR are both computational photography, they solve completely different problems. A motion deblurring network (e.g., DeblurGAN, MPRNet, Stripformer) would be more appropriate.
- Uformer: GOOD. Uformer (Wang et al., CVPR 2022) is a general image restoration Transformer that handles deblurring, denoising, and deraining. It is applicable to coded exposure deblurring.

### Are citations correct?
YES. All citations are real papers:
- Wiener-Deconv: "Analytical baseline" -- correct standard label
- PnP-FFDNet: Zhang et al., 2017 -- correct (FFDNet paper)
- HDR-CNN: Eilertsen et al., ACM TOG 2017 -- correct paper, but wrong domain (HDR, not deblurring)
- Uformer: Wang et al., CVPR 2022 -- correct

### Other issues
- check.md reports Restormer instead of Uformer, and PnP-DRUNet instead of PnP-FFDNet. The check.md is stale.
- The computational_photography pool is shared across HDR, coded exposure, light field, and related modalities. HDR-CNN makes sense for the pool as a whole but is not appropriate specifically for coded exposure.

## Plan

No code changes needed. The computational_photography pool provides 3 out of 4 appropriate algorithms for coded exposure. The HDR-CNN inclusion is a known limitation of sharing a single pool across all computational photography modalities. The Wiener-Deconv baseline is especially well-matched since Wiener deconvolution is the foundational method for flutter shutter.
