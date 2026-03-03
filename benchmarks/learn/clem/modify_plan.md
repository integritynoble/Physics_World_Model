# Modify Plan -- clem

**Date:** 2026-03-03
**Category:** multi_modal_fusion | **Carrier:** Photon | **Score key:** multi_modal_fusion

## Current Algorithms (from catalog)

| # | Algorithm | Type          | Source                             |
|---|-----------| --------------|------------------------------------|
| 1 | MLAA      | Classical     | Rezaei et al., IEEE TMI 2012       |
| 2 | MR-Guided | PnP           | Ehrhardt et al., SIIS 2015         |
| 3 | FBSEM-Net | Deep Learning | Mehranian & Reader, IEEE TMI 2020  |
| 4 | PPMF-Net  | Transformer   | Li et al., 2024                    |

## Assessment

### Are algorithms domain-appropriate?
NO -- SIGNIFICANT MISMATCH. CLEM (Correlative Light-Electron Microscopy) is a multi-modal microscopy technique that combines fluorescence (light) microscopy with electron microscopy (SEM/TEM). The multi_modal_fusion pool provides PET-MR/PET-CT fusion algorithms, which are from a completely different domain:

- MLAA (Maximum Likelihood Activity and Attenuation): This is a PET-CT joint reconstruction algorithm. It has NO relevance to CLEM. CLEM does not involve activity estimation or attenuation correction.
- MR-Guided: MR-guided PET reconstruction (Ehrhardt et al., 2015). Again, PET-MR specific, not CLEM-related.
- FBSEM-Net: Forward-Backward Stochastic EM for PET (Mehranian & Reader, 2020). PET-specific, not CLEM.
- PPMF-Net: Multi-modal PET fusion network (Li et al., 2024). PET fusion, not CLEM.

CLEM-appropriate algorithms would include:
- Image registration methods (rigid/affine/deformable) to align LM and EM images at different scales
- Super-resolution methods to bridge resolution gap between LM (~200nm) and EM (~1nm)
- Overlay/fusion methods for combining fluorescence signal with ultrastructural EM data
- ec-CLEM (Lasagne et al., 2019) -- a widely-used CLEM registration tool

### Are citations correct?
YES, the citations are real papers, but they are for PET-CT/PET-MR fusion, not CLEM:
- MLAA: Rezaei et al., IEEE TMI 2012 -- real PET paper
- MR-Guided: Ehrhardt et al., SIIS 2015 -- real PET-MR paper
- FBSEM-Net: Mehranian & Reader, IEEE TMI 2020 -- real PET paper
- PPMF-Net: Li et al., 2024 -- real multi-modal PET paper

### Other issues
- The multi_modal_fusion pool is designed for nuclear medicine (PET-CT, PET-MR, SPECT-CT) but CLEM is a microscopy fusion technique with fundamentally different physics
- The carrier is "Photon" which does not trigger any carrier routing override for multi_modal_fusion, so it falls through to the base pool
- check.md shows PPMF-Net, FuseNet, MR-Guided, OSEM+AC -- slightly different from actual catalog but still PET-oriented

## Plan

No code changes needed. While the PET-oriented multi_modal_fusion algorithms are a poor domain match for CLEM, fixing this would require creating a new CLEM-specific or microscopy-fusion sub-category pool, which is beyond the scope of this review. The current algorithms serve as placeholder baselines in the cross-modality benchmark framework. A future improvement would be to add carrier-based routing for (multi_modal_fusion, Photon) -> a microscopy_fusion pool, or to add (multi_modal_fusion, Electron) routing since CLEM involves both photon and electron carriers.
