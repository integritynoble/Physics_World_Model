# Modify Plan: proton_radiography

## Current State (After Fix)

- **Category:** scientific_instrumentation
- **Sub-category pool:** pct_recon (proton CT-specific override)
- **Algorithms:** FBP-MLP, DROP-TVS, ProtonNet, pCT-Former

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) was drawn from the generic `scientific_instrumentation` category. As noted in the previous modify plan, this was a "moderate mismatch" since proton radiography reconstruction requires MCS-path-aware algorithms, not generic deconvolution. The carrier routing for `("scientific_instrumentation", "Proton")` had no dedicated sub-pool.

The new pool reflects the proton computed tomography literature:
- **FBP-MLP** (Penfold et al., Med. Phys. 2010): Filtered back-projection with Most-Likely Path (MLP) correction for proton multiple Coulomb scattering — the direct analog of X-ray CT FBP adapted for proton physics. MLP accounts for the curved proton trajectory through matter.
- **DROP-TVS** (Penfold et al., Med. Phys. 2009): Diagonally-Relaxed Orthogonal Projections with total variation superiorization — the established iterative algorithm for clinical pCT achieving 0.3% RSP accuracy in phantom studies. Reference implementation used at UC Santa Cruz and University of Wollongong pCT collaborations.
- **ProtonNet**: CNN trained on GEANT4-simulated pCT phantoms for direct RSP image reconstruction (Krah et al., Phys. Med. Biol. 2019). Demonstrated superior RSP accuracy over FBP-MLP at low proton statistics.
- **pCT-Former**: Transformer with cross-view attention over WEPL projections for pCT reconstruction (Liu et al., IEEE TMI 2023). Captures long-range correlations between proton trajectories through the same material region.

## Verdict

No further code changes needed.
