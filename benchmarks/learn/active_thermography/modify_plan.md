# Modify Plan -- active_thermography

## Algorithm Catalog Review

**Category:** industrial_inspection | **Carrier:** IR | **Score key:** industrial_inspection

| Algorithm | Type | Source |
|-----------|------|--------|
| TSR | Classical | Shepard et al., 2003 |
| PnP-ADMM | PnP | ADMM + denoiser prior |
| DefectNet | Deep Learning | U-Net for NDT, 2021 |
| LSTM-NDT | Recurrent | Fang et al., 2022 |

### Domain Appropriateness

**Good fit.** Active thermography is squarely in the industrial NDT domain, and the carrier "IR" does not trigger any carrier routing override, so it stays in the `industrial_inspection` pool. This is correct behavior.

- **TSR (Shepard et al., 2003)** -- Thermographic Signal Reconstruction is one of the most widely cited methods for pulsed thermography data analysis. Correct and appropriate.
- **PnP-ADMM** -- Generic plug-and-play method. Applicable as a general-purpose regularized solver.
- **DefectNet** -- U-Net architecture for NDT defect detection. Reasonable.
- **LSTM-NDT** -- Recurrent approach for temporal thermography sequences. Reasonable.

**Issues:**
1. **PnP-ADMM source too vague** -- "ADMM + denoiser prior" is not a citable reference. Should be "Venkatakrishnan et al., IEEE GlobalSIP 2013" or a thermography-specific PnP reference.
2. **DefectNet source vague** -- "U-Net for NDT, 2021" needs a real author/venue/DOI.
3. **LSTM-NDT** -- "Fang et al., 2022" is plausible but needs full citation details.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists only "Adjoint" and "PnP-ADMM" as solvers, not matching the leaderboard page (TSR, PnP-ADMM, DefectNet, LSTM-NDT). Should be updated.

## Proposed Changes

1. **`_algorithm_catalog.py`**: Fix PnP-ADMM source to a proper citation (Venkatakrishnan et al., IEEE GlobalSIP 2013).
2. **`_algorithm_catalog.py`**: Fix DefectNet source to include real authors/venue.
3. **`03_reconstruction_algorithms.md`**: Update solver table to match the 4 leaderboard algorithms (TSR, PnP-ADMM, DefectNet, LSTM-NDT).

**Priority:** LOW -- algorithms are appropriate for the domain; only citation quality and learning material sync need fixing.
