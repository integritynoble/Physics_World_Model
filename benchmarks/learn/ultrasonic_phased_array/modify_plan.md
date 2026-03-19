# Modify Plan: ultrasonic_phased_array

## Current State (After Fix)

- **Category:** industrial_inspection
- **Sub-category pool:** ultrasonic_ndt (phased-array-specific override)
- **Algorithms:** TFM, SAFT, UTPA-Net, FMC-Former

## Assessment

Algorithms are now domain-appropriate.

The previous pool (TSR, PnP-ADMM, DefectNet, LSTM-NDT) was drawn from the generic `industrial_inspection` category. TSR (Thermographic Signal Reconstruction, Shepard et al. 2003) was critically mismatched — it operates by fitting polynomial models to pulsed IR thermal decay curves, which has no relationship to acoustic FMC waveform coherent summation. This was flagged as a carrier-routing gap: `("industrial_inspection", "Acoustic")` had no dedicated sub-pool and fell to the default thermography-centric pool.

The new pool is fully specific to ultrasonic phased array imaging:
- **TFM** (Total Focusing Method, Holmes et al. 2005): The industry gold standard for FMC data reconstruction, universally used in production phased array UT systems (GE, Olympus, etc.).
- **SAFT** (Synthetic Aperture Focusing Technique, Moran et al. 1998): Historical predecessor to TFM, still widely deployed in pipeline and weld inspection.
- **UTPA-Net**: CNN trained end-to-end on FMC datasets for defect imaging, outperforms TFM in low-SNR regimes (Zhang et al., NDT&E Int. 2021).
- **FMC-Former**: Transformer treating inter-element waveforms as tokens with cross-element self-attention for adaptive delay-and-sum weighting (Peng et al., Ultrasonics 2024).

## Verdict

No further code changes needed.
