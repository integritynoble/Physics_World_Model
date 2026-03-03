# Modify Plan: talbot_lau

## Current State
- **Category:** coherent
- **Carrier:** X-ray
- **Score key:** coherent
- **Algorithms:**
  1. GS/HIO (Classical) -- Fienup, Appl. Opt. 1982
  2. prDeep (PnP) -- Metzler et al., ICML 2018
  3. PhaseNet (Deep Learning) -- Rivenson et al., LSA 2018
  4. LRGS (Deep Unrolling) -- Choi et al., 2023

## Assessment

**Problem:** The coherent pool contains general phase retrieval algorithms designed for recovering phase from intensity-only measurements (Fourier phase retrieval problem). Talbot-Lau X-ray grating interferometry is fundamentally different -- it uses a three-grating setup with phase stepping to directly measure absorption, differential phase contrast (DPC), and dark-field signals. The phase information is extracted from the stepping curve, not from solving a phase retrieval problem.

- **GS/HIO (Gerchberg-Saxton / Hybrid Input-Output)** solves the Fourier phase retrieval problem. Talbot-Lau does NOT require phase retrieval -- the phase is directly encoded in the intensity modulation of the stepping curve.
- **prDeep** is a PnP method for phase retrieval from coded diffraction patterns. Not applicable.
- **PhaseNet** is a DL method for holographic phase retrieval. Not applicable.
- **LRGS** is for learned phase retrieval. Not applicable.

Appropriate Talbot-Lau algorithms include:
1. **Phase Stepping + FFT** (Classical) -- Weitkamp et al., Opt. Express 2005; extract absorption/DPC/dark-field from stepping curve via Fourier analysis
2. **Moiré analysis** (Classical) -- Bevins et al., Med. Phys. 2012; single-shot extraction via spatial carrier frequency
3. **TV-regularized DPC** (Iterative) -- Nilchian et al., Opt. Express 2013; total variation regularization for differential phase integration
4. **DPC-Net** (Deep Learning) -- DL-based phase contrast signal extraction

## Required Changes

Add `talbot_lau` to `_VARIANT_OVERRIDES` in `_algorithm_catalog.py`:

```python
"talbot_lau": [
    {"name": "Phase-Step FFT",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Weitkamp et al., Opt. Express 2005"},
    {"name": "Moire Analysis",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Bevins et al., Med. Phys. 2012"},
    {"name": "TV-DPC",          "type": "Iterative",     "mask_aware": True,  "params": "0",    "source": "Nilchian et al., Opt. Express 2013"},
    {"name": "DPC-Net",         "type": "Deep Learning", "mask_aware": False, "params": "3M",   "source": "DL phase contrast, 2023"},
],
```

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`: Add variant override for `talbot_lau`
