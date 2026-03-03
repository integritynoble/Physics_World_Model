# Modify Plan: cryo_et

## Current State

- **Category:** electron_microscopy
- **Carrier:** Electron
- **Routing:** Category is `electron_microscopy`, variant is in `_CRYO_EM_VARIANTS`, so gets the electron_microscopy pool.
- **Score key:** electron_microscopy
- **Algorithms served:**
  1. RELION (Classical) -- Scheres, J. Struct. Biol. 2012
  2. cryoSPARC (PnP) -- Punjani et al., Nat. Methods 2017
  3. cryoDRGN (Deep Learning) -- Zhong et al., Nat. Methods 2021
  4. CryoTransformer (Transformer) -- Dhakal et al., Bioinf. 2024

## Assessment

The algorithms are appropriate for cryo-electron tomography:

- **RELION:** The gold-standard software for cryo-EM/ET reconstruction. Supports subtomogram averaging and tilt series processing. CORRECT.
- **cryoSPARC:** Industry-standard for cryo-EM, increasingly used for cryo-ET subtomogram averaging. CORRECT.
- **cryoDRGN:** Deep generative model for heterogeneous reconstruction. Applied to both SPA and tomography. CORRECT.
- **CryoTransformer:** Transformer-based approach for cryo-EM/ET reconstruction. CORRECT.

Minor note: cryo-ET-specific tools like IMOD (weighted back-projection for tilt series) or IsoNet (missing wedge compensation) would be even more domain-specific, but the current algorithms are all genuinely used in the cryo-ET community.

No code changes needed.
