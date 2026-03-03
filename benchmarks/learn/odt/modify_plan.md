# Modify Plan: odt (Optical Diffraction Tomography)

## Current State

- **Category:** coherent
- **Carrier:** Photon
- **Score key:** coherent (direct category match)
- **Algorithms served (4):**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 1982
  2. prDeep (PnP) -- Metzler et al., ICML 2018
  3. PhaseNet (Deep Learning) -- Rivenson et al., LSA 2018
  4. LRGS (Deep Unrolling) -- Choi et al., 2023

## Assessment

**Good.** ODT (Optical Diffraction Tomography) is a coherent imaging modality that
reconstructs 3D refractive index distributions from multiple-angle holographic
measurements. The coherent pool is appropriate:

- **GS/HIO** (Gerchberg-Saxton / Hybrid Input-Output) is the standard phase
  retrieval algorithm and is used as the first step in ODT reconstruction
  (recovering complex fields from intensity measurements).
- **prDeep** is a PnP phase retrieval method directly applicable to ODT.
- **PhaseNet** (Rivenson et al.) is a deep learning approach for phase recovery
  from intensity measurements.
- **LRGS** is a deep unrolling method for phase retrieval.

More ODT-specific algorithms would include:
- Rytov/Born inversion (classical ODT-specific reconstruction)
- Iterative multi-slice beam propagation
- ODT-specific learned methods (e.g., Learning-ODT, Lim et al., Optica 2023)

But the coherent pool accurately captures the core phase retrieval challenge that
underlies ODT, and all four algorithms have published applications to coherent
imaging problems closely related to ODT.

## Verdict

No code changes needed.
