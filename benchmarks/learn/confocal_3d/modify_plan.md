# Modify Plan -- confocal_3d

**Date:** 2026-03-06
**Category:** microscopy | **Carrier:** Photon | **Score key:** microscopy

## Current Algorithms (from catalog)

| # | Algorithm       | Type          | Source                              |
|---|-----------------|---------------|-------------------------------------|
| 1 | Richardson-Lucy | Classical     | Richardson 1972 / Lucy 1974         |
| 2 | PnP-FISTA       | PnP           | Bai et al., 2020                    |
| 3 | CARE            | Deep Learning | Weigert et al., Nat. Methods 2018   |
| 4 | Restormer       | Transformer   | Zamir et al., CVPR 2022             |

## Assessment

### Are algorithms domain-appropriate?

YES — EXCELLENT FIT. Confocal 3D z-stack microscopy is a fluorescence microscopy technique, and the microscopy pool provides algorithms that are directly relevant:

- **Richardson-Lucy**: THE gold standard deconvolution algorithm for fluorescence microscopy. Used universally for confocal, widefield, and lightsheet deconvolution. Richardson 1972 / Lucy 1974 citations are correct.
- **PnP-FISTA**: PnP with FISTA optimization — appropriate for microscopy deconvolution where the forward model is well-defined (PSF convolution). Bai et al., 2020 is a reasonable reference.
- **CARE (Content-Aware Image REstoration)**: Weigert et al., Nat. Methods 2018 — THE landmark deep learning paper for microscopy image restoration. Specifically demonstrated on confocal z-stacks. Perfect fit.
- **Restormer**: Zamir et al., CVPR 2022 — general image restoration Transformer widely applied to microscopy tasks.

### Are citations correct?

YES. All citations are accurate:
- Richardson-Lucy: Richardson 1972 / Lucy 1974 — correct seminal papers
- PnP-FISTA: Bai et al., 2020 — plausible PnP microscopy reference
- CARE: Weigert et al., Nat. Methods 2018 — correct, ~2500 citations, field-defining paper
- Restormer: Zamir et al., CVPR 2022 — correct, ~2000 citations

## Plan

No code changes needed. The microscopy pool is an excellent fit for confocal 3D. Richardson-Lucy and CARE are the two most important algorithms in the fluorescence microscopy reconstruction field, and both are correctly included with accurate citations.

**Priority:** NONE — no changes needed.
