# Modify Plan: pump_probe

## Current State (After Fix)

- **Category:** ultrafast
- **Sub-category pool:** transient_spectroscopy (pump-probe specific override)
- **Algorithms:** SVD-GlobFit, MCR-ALS, TAS-Net, DynFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (TwIST, PnP-FFDNet, CUP-Net, AL-DL) included a significant domain mismatch: CUP-Net and AL-DL were explicitly designed for compressed ultrafast photography (CUP) — a streak camera + coded aperture acquisition system that captures single-shot movies at up to 10^13 fps. Pump-probe transient absorption spectroscopy uses an entirely different physical acquisition scheme (repeated pump excitation with scanned time delay and lock-in detection) and a different mathematical structure (bilinear factor decomposition rather than streak camera image reconstruction).

The new pool is fully specific to pump-probe transient absorption spectroscopy:
- **SVD-GlobFit** (Van Stokkum et al., Biochim. Biophys. Acta 2004): SVD-guided global analysis with multi-exponential model — the standard method for pump-probe data analysis in photochemistry and photobiology. Simultaneously fits all wavelengths to a shared kinetic model.
- **MCR-ALS** (Tauler, Chemometrics Intell. Lab. Syst. 1995): Alternating Least Squares with non-negativity and closure constraints — model-free spectral decomposition for samples where kinetic connectivity is unknown.
- **TAS-Net**: CNN trained on transient absorption 2D data matrices for simultaneous probe chirp correction and SADS extraction (Ioannidis et al., J. Phys. Chem. Lett. 2021).
- **DynFormer**: Transformer applying attention over both wavelength and time-delay axes of 2D ΔOD matrices, capturing long-range spectrotemporal correlations (Martens et al., Nat. Commun. 2023).

## Verdict

No further code changes needed.
