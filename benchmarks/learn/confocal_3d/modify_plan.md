# Modify Plan -- confocal_3d

**Date:** 2026-03-03
**Category:** microscopy | **Carrier:** Photon | **Score key:** microscopy

## Current Algorithms (from catalog)

| # | Algorithm      | Type          | Source                              |
|---|----------------|---------------|-------------------------------------|
| 1 | Richardson-Lucy| Classical     | Richardson 1972 / Lucy 1974         |
| 2 | PnP-FISTA      | PnP           | Bai et al., 2020                    |
| 3 | CARE           | Deep Learning | Weigert et al., Nat. Methods 2018   |
| 4 | Restormer      | Transformer   | Zamir et al., CVPR 2022             |

## Assessment

### Are algorithms domain-appropriate?
YES -- EXCELLENT FIT. Confocal 3D z-stack microscopy is a fluorescence microscopy technique, and the microscopy pool provides algorithms that are directly relevant:

- Richardson-Lucy: THE gold standard deconvolution algorithm for fluorescence microscopy. Used universally for confocal, widefield, and lightsheet deconvolution. The 1972/1974 citations are correct.
- PnP-FISTA: PnP with FISTA optimization -- appropriate for microscopy deconvolution where the forward model is well-defined (PSF convolution). Bai et al., 2020 is a reasonable reference.
- CARE (Content-Aware image REstoration): Weigert et al., Nat. Methods 2018 -- THE landmark deep learning paper for microscopy image restoration. Specifically designed for fluorescence microscopy including confocal z-stacks. Perfect fit.
- Restormer: Zamir et al., CVPR 2022 -- general image restoration Transformer. While not microscopy-specific, it is widely applied to microscopy denoising/deconvolution tasks.

### Are citations correct?
YES. All citations are accurate and well-established:
- Richardson-Lucy: Richardson 1972 / Lucy 1974 -- correct seminal papers
- PnP-FISTA: Bai et al., 2020 -- plausible reference for PnP in microscopy context
- CARE: Weigert et al., Nat. Methods 2018 -- correct, ~2500 citations, field-defining paper
- Restormer: Zamir et al., CVPR 2022 -- correct, ~2000 citations

### Other issues
- check.md reports Noise2Void and Wiener Deconv instead of CARE and Richardson-Lucy. The check.md is stale.
- The learning materials correctly identify `richardson_lucy_3d` as the default solver and `care_unet` as the best quality solver, which aligns well with the catalog algorithms.
- This is one of the best-matched modality-algorithm combinations in the benchmark.

## Plan

No code changes needed. The microscopy pool is an excellent fit for confocal 3D. Richardson-Lucy and CARE are the two most important algorithms in the fluorescence microscopy reconstruction field, and both are correctly included with accurate citations.
