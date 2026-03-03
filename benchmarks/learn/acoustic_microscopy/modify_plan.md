# Modify Plan -- acoustic_microscopy

## Algorithm Catalog Review

**Category:** industrial_inspection | **Carrier:** Acoustic | **Score key:** industrial_inspection

| Algorithm | Type | Source |
|-----------|------|--------|
| TSR | Classical | Shepard et al., 2003 |
| PnP-ADMM | PnP | ADMM + denoiser prior |
| DefectNet | Deep Learning | U-Net for NDT, 2021 |
| LSTM-NDT | Recurrent | Fang et al., 2022 |

### Domain Appropriateness

**Partially appropriate.** The algorithms are routed via the `industrial_inspection` category pool which covers NDT/thermography methods. TSR (Thermographic Signal Reconstruction) is a thermography-specific method, not an acoustic microscopy method. Acoustic microscopy (SAM) reconstruction is closer to ultrasound imaging (DAS, SAFT, synthetic aperture focusing). The carrier is "Acoustic" but there is no `(industrial_inspection, Acoustic)` routing rule in `_CARRIER_ROUTING`, so it falls through to the generic `industrial_inspection` pool.

**Issues:**
1. **TSR is wrong domain** -- TSR (Shepard et al., 2003) is a pulsed thermography time-series method. For acoustic microscopy, the classical baseline should be SAFT (Synthetic Aperture Focusing Technique) or time-reversal beamforming.
2. **PnP-ADMM source too vague** -- "ADMM + denoiser prior" is not a real citation. Should cite Venkatakrishnan et al., IEEE GlobalSIP 2013.
3. **DefectNet source vague** -- "U-Net for NDT, 2021" needs a real author/venue/DOI.
4. **LSTM-NDT** -- "Fang et al., 2022" is plausible but needs full venue/DOI.
5. **No carrier routing** -- Adding `("industrial_inspection", "Acoustic")` to `_CARRIER_ROUTING` to point to a dedicated ultrasonic NDT algorithm pool would be ideal but is a larger change.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists only "Adjoint" and "PnP-ADMM" as solvers, which does not match the 4 algorithms on the leaderboard page (TSR, PnP-ADMM, DefectNet, LSTM-NDT). The learning materials should reference the same algorithms shown on the benchmark page.

## Proposed Changes

1. **`_algorithm_catalog.py`**: Add carrier routing `("industrial_inspection", "Acoustic")` pointing to a new `us_ndt` pool with SAFT, PnP-ADMM (proper citation), a DL method for ultrasonic testing, and a transformer/recurrent method for acoustic time-series.
2. **`_algorithm_catalog.py`**: Fix PnP-ADMM source string to a real citation.
3. **`03_reconstruction_algorithms.md`**: Update solver table to match the 4 leaderboard algorithms.

**Priority:** MEDIUM -- algorithms are plausible for NDT but not specific to acoustic (ultrasonic) microscopy.
