# Modify Plan: multispectral_sat (Multispectral Satellite Imaging)

## Current State

- **Category:** remote_sensing
- **Carrier:** Photon
- **Score key:** computational (routed via `_CARRIER_ROUTING[("remote_sensing", "Photon")]`)
- **Algorithms served (4):**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. Deep Image Prior (Deep Learning) -- Ulyanov et al., CVPR 2018
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**Appropriate.** The routing correctly identifies that optical remote sensing
(Photon carrier) should NOT use SAR-specific algorithms (Matched Filter, SAR-BM3D,
SAR-DRN, SAR-CAM), which are radar-domain methods. Instead it falls back to the
`computational` pool, which provides generic image reconstruction algorithms.

These algorithms are reasonable for multispectral satellite pan-sharpening and
super-resolution:

- Tikhonov regularization is a standard baseline for spectral unmixing / super-resolution.
- PnP-RED is well-suited for image restoration with learned denoisers.
- Deep Image Prior is used in remote sensing super-resolution (Sidorov & Hardeberg, 2019).
- SwinIR is a strong general-purpose image restoration transformer.

More domain-specific algorithms would include:
- CNMF (Yokoya et al., IEEE TGRS 2012) for coupled spectral unmixing
- HySure (Simoes et al., IEEE TGRS 2015) for hyperspectral super-resolution
- DARN (Wei et al., IEEE TGRS 2020) for deep attention-based fusion

But the current generic computational pool is a reasonable approximation.

## Current Algorithm Count (updated 2026-03-06)

Full pool (13 algorithms, computational pool): Tikhonov, LSQR, ART, PnP-RED, PnP-ADMM, Deep Image Prior, Plug-and-Play, SwinIR, Restormer, NAFNet, CompFormer, DiffusionCompute, FlowCompute.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No code changes needed.
