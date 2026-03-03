# Modify Plan: XFEL Serial Femtosecond Crystallography (SFX)

**Created:** 2026-03-03
**Status:** Algorithms are a significant mismatch -- ultrafast imaging methods assigned to crystallography

## Assessment

XFEL SFX falls under `ultrafast` category with carrier `X-ray`. It receives:

- TwIST (Classical) -- compressed sensing solver (Bioucas-Dias & Figueiredo, IEEE TIP 2007)
- PnP-FFDNet (PnP) -- plug-and-play for compressed ultrafast photography (Yuan et al., 2020)
- CUP-Net (Deep Learning) -- compressed ultrafast photography network (Parker et al., 2021)
- AL-DL (Hybrid) -- alternating learning for ultrafast imaging (Yao et al., Photon. Res. 2021)

### Issue

The `ultrafast` category algorithms are designed for **compressed ultrafast photography (CUP)** and streak-camera-based temporal imaging. XFEL SFX is fundamentally different:

- SFX collects diffraction patterns from single crystals hit by femtosecond X-ray pulses
- The "ultrafast" aspect is the pulse duration, not the reconstruction problem
- The reconstruction task is **crystallographic phasing**: determining electron density from diffraction intensities

XFEL SFX reconstruction involves:
- Indexing and integration of diffraction patterns (hit finding)
- Merging partial-reflection intensities from many crystals
- Phase retrieval via molecular replacement or ab initio methods

Appropriate algorithms would be:
- Classical: CrystFEL indexing + Monte Carlo integration (White et al., J. Appl. Cryst. 2012)
- Iterative: Expand-Maximize-Compress (EMC) algorithm (Loh & Elser, PRE 2009)
- Deep Learning: DeepFreak (Ke et al., Acta Cryst. 2018) for pattern classification
- Phase retrieval: Difference map / HIO for structure factor phasing

TwIST and CUP-Net solve completely different inverse problems (video frame recovery from a single coded exposure).

### Decision

XFEL SFX was placed in `ultrafast` because of the femtosecond X-ray pulses, but the reconstruction problem is crystallographic, not temporal. It should either be moved to `scientific_instrumentation` or given a variant override.

## Deferred Items

1. **HIGH PRIORITY**: Add `xfel_sfx` to `_VARIANT_OVERRIDES` with crystallography-appropriate algorithms:
   - Classical: CrystFEL Monte Carlo integration (White et al., J. Appl. Cryst. 2012)
   - Iterative: EMC algorithm (Loh & Elser, PRE 2009)
   - Deep Learning: CNN-based hit finding / indexing
   - Phasing: Direct methods or iterative phasing
2. **Category reconsideration**: The `ultrafast` category label is misleading for SFX. The modality might better fit `scientific_instrumentation` or a dedicated `crystallography` sub-pool.
3. **Score key**: `ultrafast` scores (CUP-Net PSNR/SSIM) are meaningless for crystallographic reconstruction where the metric should be R-factor or resolution.

No code changes made in this pass, but this is one of the most significant mismatches reviewed.
