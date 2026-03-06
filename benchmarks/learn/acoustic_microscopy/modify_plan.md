# Modify Plan -- acoustic_microscopy

## Algorithm Catalog Review

**Category:** industrial_inspection | **Carrier:** Acoustic | **Score key:** industrial_inspection

| Algorithm | Type | Source |
|-----------|------|--------|
| SAFT | Classical | Schickert et al., NDT&E International 36, 339 (2003) |
| PnP-ADMM | PnP | Venkatakrishnan et al., IEEE GlobalSIP 2013 |
| SAM-Net | Deep Learning | CNN for acoustic microscopy defect imaging, 2022 |
| AcousticFormer | Transformer | Transformer for acoustic NDT, 2024 |

### Domain Appropriateness

**Good fit (after correction).** The acoustic_microscopy variant now uses SAFT, PnP-ADMM, SAM-Net, and AcousticFormer — all directly appropriate for scanning acoustic microscopy (SAM) image reconstruction. The prior entry had TSR (Thermographic Signal Reconstruction), which was a thermography-specific method incorrectly included in the acoustic pool.

**Correction applied:**
- Removed TSR (Shepard et al., 2003) — this is a pulsed thermography time-series method, inappropriate for acoustic microscopy
- Added SAFT (Synthetic Aperture Focusing Technique) as the correct classical baseline for ultrasonic NDT

**Current algorithm quality:**
1. **SAFT** — Schickert et al., NDT&E Int. 36, 339 (2003): SAFT is THE standard analytical reconstruction algorithm for scanning acoustic microscopy and ultrasonic pulse-echo NDT. Correct.
2. **PnP-ADMM** — Venkatakrishnan et al., IEEE GlobalSIP 2013: Correct canonical citation for PnP framework.
3. **SAM-Net** — Acoustic microscopy CNN, 2022: Plausible DL reference; needs full citation (e.g., Zhu et al., IEEE TUFFC 2022).
4. **AcousticFormer** — Transformer for acoustic NDT, 2024: Plausible transformer reference; needs full citation.

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Adjoint" and "PnP-ADMM" as solvers, which partially matches (PnP-ADMM is present). Should be updated to reference SAFT, PnP-ADMM, SAM-Net, and AcousticFormer.

## Proposed Changes

1. **No code changes to `_algorithm_catalog.py`**: Algorithm pool now correct.
2. **`03_reconstruction_algorithms.md`**: Update solver table to list SAFT, PnP-ADMM, SAM-Net, AcousticFormer with proper citations.
3. **Score pool**: Consider adding a `acoustic_ndt` score pool to `_leaderboard_generator.py` with PSNR ranges calibrated for SAM image quality (current `industrial_inspection` pool is calibrated for thermography). LOW priority.

**Priority:** LOW — algorithms are now correct; only documentation sync and optional score calibration remain.
