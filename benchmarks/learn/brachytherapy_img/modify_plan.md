# Modify Plan -- brachytherapy_img

## Algorithm Catalog Review

**Category:** medical | **Carrier:** Gamma/X-ray | **Score key:** medical

| Algorithm | Type | Source |
|-----------|------|--------|
| FBP | Classical | Analytical baseline |
| PnP-ADMM | PnP | Venkatakrishnan et al., 2013 |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 2017 |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 |

### Domain Appropriateness

**Acceptable but not ideal.** Brachytherapy imaging uses X-ray or gamma-ray based imaging (fluoroscopy, CT) to verify seed/source placement for radiation therapy. The algorithms fall through to the generic `medical` pool because the carrier "Gamma/X-ray" does not match any `_CARRIER_ROUTING` entry (the routing has "Gamma" and "X-ray" separately but not "Gamma/X-ray" as a combined string).

- **FBP** -- Valid for X-ray CT reconstruction used in brachytherapy verification imaging. Appropriate.
- **PnP-ADMM** -- Venkatakrishnan et al., 2013. Real citation. Appropriate.
- **FBPConvNet** -- Jin et al., IEEE TIP 2017. Real CT reconstruction paper. Appropriate.
- **Learned Primal-Dual** -- Adler & Oktem, IEEE TMI 2018. Real paper. Appropriate.

Since brachytherapy verification imaging is essentially CT or fluoroscopic X-ray imaging, these CT reconstruction algorithms are technically appropriate.

**Issues:**
1. **Carrier routing gap** -- "Gamma/X-ray" is a compound carrier that does not match either "Gamma" or "X-ray" routing. This is technically a routing bug, but the fallthrough to the `medical` CT pool happens to be correct for this modality. If the carrier were just "Gamma", it would route to `particle_imaging` (PET/SPECT), which would be wrong.
2. **No brachytherapy-specific methods** -- Seed localization, dose verification, and TG-43 protocol methods are absent, but these are more dosimetry than imaging reconstruction.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "FBP" and "DL-Recon" -- FBP matches, but "DL-Recon" is a generic placeholder that should be replaced with specific names.

## Proposed Changes

1. **`_algorithm_catalog.py`**: Add carrier routing for `("medical", "Gamma/X-ray")` pointing to `"medical"` explicitly, to document the intent and prevent future breakage if the fallthrough logic changes.
2. **`03_reconstruction_algorithms.md`**: Replace "DL-Recon" with FBPConvNet and add PnP-ADMM and Learned Primal-Dual entries.

No algorithm changes needed -- the CT-family algorithms are appropriate for brachytherapy verification imaging.

**Priority:** LOW -- algorithms are appropriate; only a carrier routing documentation fix and learning materials sync needed.
