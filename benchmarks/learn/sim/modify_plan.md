# Modify Plan -- sim

## Current State

- **Category:** microscopy
- **Carrier:** Photon
- **Routing:** No carrier routing for `("microscopy", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["microscopy"]`
- **Score key:** microscopy
- **Algorithms assigned:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

**Partially appropriate: Acceptable but misses SIM-specific algorithms.**

Structured Illumination Microscopy (SIM) achieves super-resolution by illuminating the sample with patterned light (typically sinusoidal gratings) at multiple orientations and phases. The reconstruction problem is fundamentally different from standard deconvolution: it involves frequency-space separation of mixed spectral components to extend the optical transfer function (OTF) beyond the diffraction limit.

- **Richardson-Lucy**: Standard deconvolution, applicable as a baseline but does not exploit the structured illumination. SIM's classical reconstruction is Wiener filtering in frequency space with component separation (Gustafsson, J. Microsc. 2000), not Richardson-Lucy.
- **PnP-FISTA**: Generic PnP optimization. Applicable but ignores the multi-pattern structure.
- **CARE**: Published for microscopy image restoration broadly. Can process SIM data but as a generic denoising step, not as a SIM reconstructor.
- **Restormer**: Generic transformer restoration. Same caveat.

Domain-specific SIM algorithms would include: Wiener-SIM (Gustafsson 2000), fairSIM (Muller et al., 2016), HiFi-SIM (Wen et al., 2023), and DL-SIM (Christensen et al., 2021). The check.md already notes that HiFi-SIM and DL-SIM from the registry are not found on the page.

However, the platform framework treats all modalities with the same generic forward model, and the benchmark measures reconstruction quality from the linear operator perspective. The microscopy pool provides a reasonable spread of algorithm families.

## Plan

No code changes needed. While SIM-specific algorithms (fairSIM, HiFi-SIM) would improve domain authenticity, the current microscopy pool is functional for the benchmark framework. The generic forward model does not distinguish SIM's multi-pattern acquisition from standard widefield imaging at the algorithm-selection level.
