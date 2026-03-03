# Modify Plan: X-ray Crystallography

**Created:** 2026-03-03
**Status:** Algorithms are generic but acceptable for the instrumentation pool

## Assessment

X-ray crystallography falls under `scientific_instrumentation` category with carrier `X-ray`. It receives:

- Deconv (Classical) -- analytical deconvolution baseline
- PnP-BM3D (PnP) -- plug-and-play with BM3D (Danielyan et al., 2012)
- ResNet-Calib (Deep Learning) -- ResNet for calibration
- CalibFormer (Transformer) -- transformer for calibration

### Analysis

X-ray crystallography determines atomic structure from diffraction pattern intensities. The classical reconstruction pipeline involves:

1. Indexing and integration of Bragg reflections
2. Phase determination (direct methods, molecular replacement, SAD/MAD phasing)
3. Electron density map calculation via inverse Fourier transform
4. Model building and refinement

The core inverse problem is the **phase problem** -- recovering phases lost in intensity measurements. Specialized algorithms include:
- Classical: Direct methods (Shake-and-Bake, SHELXD), Patterson methods
- Iterative: Charge flipping (Oszlanyi & Suto, 2004)
- Deep Learning: AlphaFold-based molecular replacement, PhAI (Terwilliger et al., 2023)
- Refinement: REFMAC, phenix.refine

The generic `scientific_instrumentation` algorithms (Deconv, PnP-BM3D) treat this as a generic signal recovery problem, which abstracts away the crystallographic specifics. This is a known limitation of the shared instrumentation pool.

### Decision

While the algorithms are not crystallography-specific, the `scientific_instrumentation` pool is designed as a catch-all for diverse instrumentation modalities (mass spectrometry, atom probe, diffraction, etc.). The PSNR/SSIM benchmark framework does not capture crystallographic quality metrics (R-factor, resolution) anyway, so the generic algorithms serve as reconstruction baselines.

## Deferred Items

1. **MEDIUM PRIORITY**: Could add `xray_crystallography` to `_VARIANT_OVERRIDES` with phase-retrieval-based algorithms:
   - Classical: Direct methods / HIO phase retrieval
   - Iterative: Charge flipping (Oszlanyi & Suto, Acta Cryst. 2004)
   - Deep Learning: CNN phase predictor
   - Transformer: PhAI-style transformer for phasing
2. **Shared with xfel_sfx**: Both X-ray crystallography and XFEL SFX solve the crystallographic phase problem. A shared `crystallography` sub-pool could serve both.

No code changes required at this time.
