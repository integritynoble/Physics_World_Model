# Modify Plan -- ptychography

## Current State

- **Category:** coherent
- **Carrier:** Electron/Photon
- **Routing:** No carrier routing for `("coherent", "Electron/Photon")` -> falls to `_CATEGORY_ALGORITHMS["coherent"]`
- **Score key:** coherent
- **Algorithms assigned:**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 1982
  2. prDeep (PnP) -- Metzler et al., ICML 2018
  3. PhaseNet (Deep Learning) -- Rivenson et al., LSA 2018
  4. LRGS (Deep Unrolling) -- Choi et al., 2023

## Assessment

**Appropriate: YES.**

Ptychography is a scanning coherent diffraction imaging technique where overlapping diffraction patterns are used to recover both the object transmission function and the illumination probe. The core inverse problem is phase retrieval from intensity measurements, which places it squarely in the "coherent" category.

- **GS/HIO** (Gerchberg-Saxton / Hybrid Input-Output): The ePIE (extended Ptychographic Iterative Engine) algorithm that is standard for ptychography is a direct descendant of the GS/HIO family. Using GS/HIO as the classical baseline is appropriate since ePIE is essentially HIO adapted for overlapping scan positions.
- **prDeep**: PnP phase retrieval is directly applicable; the phase retrieval structure is the same.
- **PhaseNet**: Deep learning for coherent imaging / phase retrieval applies here.
- **LRGS**: Unrolled phase retrieval is a natural fit.

Domain-specific algorithms like ePIE (Rodenburg & Faulkner, 2004) or PtychoNN (Cherukara et al., Appl. Phys. Lett. 2020) would be even more precise, but the current coherent pool is a reasonable and defensible representation of the algorithm landscape.

## Plan

No code changes needed.
