# Modify Plan: stm

## Current State
- **Category:** scanning_probe
- **Carrier:** Electron
- **Score key:** scanning_probe
- **Algorithms:**
  1. BTR (Classical) -- Villarrubia, JRNIST 1997
  2. Reg-Deconv (PnP) -- Dongmo et al., 2000
  3. DeepSPM (Deep Learning) -- Alldritt et al., Commun. Phys. 2020
  4. E2E-BTR (Deep Learning) -- Kossler et al., Sci. Rep. 2022

## Assessment

**Problem:** The scanning probe pool is designed primarily for AFM (Atomic Force Microscopy) tip deconvolution, not STM (Scanning Tunneling Microscopy). While both are scanning probe techniques, they measure fundamentally different things:

- **AFM** measures tip-sample force interaction, and the key artifact is tip shape convolution (BTR = Blind Tip Reconstruction). The tip physically broadens features.
- **STM** measures tunneling current, and the key artifacts are electronic state convolution (LDOS effects), drift, piezo nonlinearity, and tip electronic structure. Physical tip convolution is much less significant because STM measures electronic density of states, not topography.

Specific issues:
- **BTR** (Blind Tip Reconstruction) is specifically for AFM tip shape deconvolution -- not applicable to STM in the same way.
- **Reg-Deconv** (Regularized Deconvolution) as described for tip deconvolution is AFM-specific.
- **DeepSPM** (Alldritt et al., 2020) is actually for STM -- it identifies molecular structures from STM images. This one is appropriate.
- **E2E-BTR** is end-to-end blind tip reconstruction for AFM.

Appropriate STM algorithms include:
1. Drift correction / creep compensation (Classical) -- standard STM preprocessing
2. STS-deconv (Classical) -- scanning tunneling spectroscopy deconvolution
3. DeepSPM (Deep Learning) -- Alldritt et al., 2020 (already in pool, correct)
4. STM-DL (Deep Learning) -- DL-based STM image enhancement

## Required Changes

Add `stm` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py` with STM-specific algorithms:

```python
"stm": [
    {"name": "Drift-Correct",  "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Piezo drift/creep compensation"},
    {"name": "Wiener-STM",     "type": "Classical",     "mask_aware": True,  "params": "0",   "source": "Wiener deconvolution baseline"},
    {"name": "DeepSPM",        "type": "Deep Learning", "mask_aware": False, "params": "2M",  "source": "Alldritt et al., Commun. Phys. 2020"},
    {"name": "BM3D",           "type": "PnP",           "mask_aware": True,  "params": "0",   "source": "Dabov et al., IEEE TIP 2007"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add variant override for `stm`
