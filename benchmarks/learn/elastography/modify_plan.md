# Modify Plan: elastography

## Current Assignment
- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** medical_ultrasound (routed via carrier)
- **Algorithms:** DAS (Classical), PnP-ADMM (PnP), ABLE (Deep Learning), MU-Net (Deep Learning)

## Assessment

The algorithms are **inappropriate**. Carrier-based routing sends elastography to
the `medical_ultrasound` pool, which contains B-mode ultrasound beamforming
algorithms. Elastography is fundamentally different: it reconstructs tissue
stiffness (shear modulus) from shear-wave propagation data, not ultrasound
echo images.

**Problems:**
1. **DAS** (delay-and-sum beamforming) is a B-mode US algorithm, not an
   elastography inversion method. The classical baseline should be
   **Direct Inversion** (Helmholtz inversion of the wave equation) or
   **algebraic inversion of the differential equation (AIDE)**.
2. **ABLE** and **MU-Net** are ultrasound beamforming networks, not shear-wave
   inversion networks. Domain-appropriate DL methods include
   **U-Net for elasticity** (Kibria & Rivaz, IEEE TUFFC 2023) or
   **ElastNet** (Wu et al., PMB 2022).
3. **PnP-ADMM** is generic enough to work but the source citation references
   ultrasound beamforming, not elasticity inversion.

## Recommended Changes

Add an elastography-specific entry to `_VARIANT_OVERRIDES` or create a
carrier-routing exception in `_algorithm_catalog.py`:

```python
"elastography": [
    {"name": "Direct Inversion",  "type": "Classical",     "mask_aware": True,  "params": "0",    "source": "Manduca et al., Med. Image Anal. 2001"},
    {"name": "PnP-TV",            "type": "PnP",           "mask_aware": True,  "params": "0",    "source": "TV-regularized shear modulus, 2018"},
    {"name": "U-Net Elasticity",  "type": "Deep Learning", "mask_aware": False, "params": "7M",   "source": "Kibria & Rivaz, IEEE TUFFC 2023"},
    {"name": "ElastNet",          "type": "Deep Learning", "mask_aware": True,  "params": "5M",   "source": "Wu et al., Phys. Med. Biol. 2022"},
],
```

Also add corresponding real scores to `CATEGORY_REAL_SCORES` under an
`"elastography"` key.

## Files to Modify
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`
  - Add `"elastography"` to `_VARIANT_OVERRIDES`
  - Add `"elastography"` to `CATEGORY_REAL_SCORES`
