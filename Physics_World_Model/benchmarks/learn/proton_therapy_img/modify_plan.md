# Modify Plan -- proton_therapy_img

## Current State

- **Category:** medical
- **Carrier:** Proton
- **Routing:** `("medical", "Proton")` -> `"medical"` (CT/X-ray pool)
- **Score key:** medical
- **Algorithms assigned:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

**Appropriate: YES, with caveats.**

Proton therapy imaging (proton CT / proton radiography) reconstructs relative stopping power (RSP) maps from proton energy loss measurements. The inverse problem is structurally very similar to X-ray CT: projections through the patient are acquired at multiple angles and tomographic reconstruction recovers a 2D/3D image. The carrier routing `("medical", "Proton") -> "medical"` correctly sends this to the CT-like algorithm pool.

- **FBP** is used in proton CT (distance-driven backprojection of proton energy loss).
- **PnP-ADMM** is a sensible regularized iterative approach for this modality.
- **FBPConvNet** and **Learned Primal-Dual** have been adapted to proton CT in published work (e.g., Kaser et al., Phys. Med. Biol. 2022).

The main difference from X-ray CT is that proton trajectories are curved (multiple Coulomb scattering), so the forward model is not a straight-line Radon transform. However, the reconstruction algorithms are the same family, just applied to a different forward operator. The current mapping is a reasonable and defensible choice.

## Plan

No code changes needed.

## 2026-03-06 Comprehensive Check Update

- Physics: WEPL sinogram via most-likely path (MLP) integral of RSP; Gaussian energy spread noise
- Key mismatch: MLP model (Highland formula), beam energy spread, detector hull geometry, prior RSP from CT
- GCS datasets: 3 tiers confirmed in challenge-data/v1.0/
- Algorithm pool: PASS — CT pool (FBP, TV-ADMM, Learned Primal-Dual, DiffusionCT) is correct since pCT shares the Radon-integral reconstruction structure with X-ray CT
- Note: Learned Primal-Dual is particularly well-suited for pCT since it can embed the MLP forward model as a differentiable layer
