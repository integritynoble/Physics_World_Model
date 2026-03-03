# Modify Plan -- radio_astronomy

## Current State

- **Category:** experimental_science
- **Carrier:** RF
- **Routing:** No carrier routing for `("experimental_science", "RF")` -> falls to `_CATEGORY_ALGORITHMS["experimental_science"]`
- **Score key:** experimental_science
- **Algorithms assigned:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

**INAPPROPRIATE. Needs change.**

Radio aperture synthesis is fundamentally a Fourier-sampling inverse problem: visibilities are measured in the (u,v) plane and the sky brightness distribution is recovered via image reconstruction from incomplete Fourier data. This is the same category of problem as radio interferometry and has a rich, domain-specific algorithm literature.

The current assignment gives generic experimental_science algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR). These are generic image restoration methods with no connection to radio astronomy. The correct algorithms are:

- **CLEAN** (Hogbom 1974) -- the standard deconvolution algorithm for radio aperture synthesis
- **AIRI** (Terris et al., MNRAS 2022) -- PnP approach for radio interferometric imaging
- **R2D2** (Aghabiglou et al., ApJS 2024) -- deep learning for radio imaging
- **PRIMO** (Medeiros et al., ApJL 2023) -- used for the EHT M87 image

These algorithms already exist in `_CATEGORY_ALGORITHMS["astronomy"]`. The issue is that `radio_astronomy` is categorized as `experimental_science` in the modality catalog instead of `astronomy`.

## Plan

**Option A (preferred):** Add a carrier routing entry to `_CARRIER_ROUTING` in `_algorithm_catalog.py`:
```python
("experimental_science", "RF"): "astronomy",  # radio aperture synthesis -> CLEAN/AIRI/R2D2
```

**Option B:** Add a variant override for `radio_astronomy` in `_VARIANT_OVERRIDES`.

**Option C:** Change the category in the YAML config from `experimental_science` to `astronomy` and regenerate the modality catalog.

Option A is simplest and lowest-risk. However, it would also route any other `experimental_science` + `RF` modalities to astronomy, so we should verify no other modalities have that combination. If any do, use Option B instead with a direct `_VARIANT_OVERRIDES` entry for `radio_astronomy`.

### Specific code change (Option A):

In `/home/spiritai/pwm/Physics_World_Model/platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`, add to `_CARRIER_ROUTING`:

```python
("experimental_science", "RF"): "astronomy",  # radio aperture synthesis -> CLEAN/AIRI/R2D2/PRIMO
```

Also add a matching entry to `CATEGORY_REAL_SCORES` if the astronomy score key needs mapping.
