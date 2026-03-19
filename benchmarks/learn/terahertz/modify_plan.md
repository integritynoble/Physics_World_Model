# Modify Plan: terahertz

## Current State (After Fix)

- **Category:** industrial_inspection
- **Sub-category pool:** thz_imaging (terahertz-specific override)
- **Algorithms:** Wiener-THz, PnP-SPIRAL, THz-Net, THz-Former

## Assessment

Algorithms are now domain-appropriate.

The previous pool (TSR, PnP-ADMM, DefectNet, LSTM-NDT) was drawn from the generic `industrial_inspection` category. TSR (Thermographic Signal Reconstruction, Shepard et al. 2003) is a technique for time-domain polynomial fitting of pulsed infrared thermography decay curves — it has zero applicability to THz spectral deconvolution or THz waveguide imaging.

The new pool is fully specific to terahertz time-domain spectroscopy and THz imaging:
- **Wiener-THz**: Frequency-domain Wiener filter deconvolution of the THz transfer function H(ω) — the canonical classical algorithm for THz-TDS (Jeon & Grischkowsky, 1997).
- **PnP-SPIRAL**: SPIRAL-TAP solver with plug-and-play denoiser prior, adapted for THz Poisson-like photon counting statistics.
- **THz-Net**: CNN operating on THz spectral features (amplitude + phase) for simultaneous denoising and material parameter extraction.
- **THz-Former**: Spatial-spectral transformer for 3D THz hyperspectral data cubes, capturing long-range frequency correlations.

## Verdict

No further code changes needed.
