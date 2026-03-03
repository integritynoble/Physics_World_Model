# Modify Plan -- pump_probe

## Current State

- **Category:** ultrafast
- **Carrier:** Photon
- **Routing:** No carrier routing for `("ultrafast", "Photon")` -> falls to `_CATEGORY_ALGORITHMS["ultrafast"]`
- **Score key:** ultrafast
- **Algorithms assigned:**
  1. TwIST (Classical) -- Bioucas-Dias & Figueiredo, IEEE TIP 2007
  2. PnP-FFDNet (PnP) -- Yuan et al., 2020
  3. CUP-Net (Deep Learning) -- Parker et al., 2021
  4. AL-DL (Hybrid) -- Yao et al., Photon. Res. 2021

## Assessment

**Partially appropriate: Minor mismatch.**

Pump-probe microscopy is a time-resolved imaging technique where a pump pulse excites a sample and a probe pulse measures the transient response at a controlled time delay. While it is indeed an ultrafast technique (femtosecond timescales), the reconstruction problem is quite different from compressed ultrafast photography (CUP) which the current algorithm pool is tuned for.

- **TwIST**: A general-purpose inverse solver, acceptable as a classical baseline for any linear reconstruction task.
- **PnP-FFDNet**: Generic enough to apply, but was proposed specifically for snapshot compressive imaging (SCI) rather than pump-probe.
- **CUP-Net**: Specifically designed for compressed ultrafast photography (CUP), which is a fundamentally different acquisition scheme (streak camera + coded aperture). Pump-probe does NOT use CUP acquisition. This is a weak match.
- **AL-DL**: Also designed for CUP-style ultrafast imaging.

Pump-probe reconstruction typically involves fitting exponential decay models to time-delay series or solving deconvolution problems to extract transient absorption spectra. The CUP-centric algorithms are not the most representative. However, pump-probe microscopy on the PWM platform uses a generic linear forward model, so the solvers technically work as inverse-problem baselines even if they are not domain-canonical.

## Plan

No code changes needed. The current algorithms are technically functional for the benchmark's generic inverse-problem framework, even though a domain expert would expect to see transient absorption fitting algorithms. The mismatch is cosmetic rather than functional since all modalities share the same generic forward-model structure on the platform.
