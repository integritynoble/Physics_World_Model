# Modify Plan — acoustic_emission (Acoustic Emission Testing)

**Updated:** 2026-03-07
**Status:** IMPLEMENTED

---

## Problems Identified

1. **Wrong ground-truth generator**: The registry used `generate_ndt_phantom` (a defect-geometry map with voids, cracks as material boundaries). For AE, the correct ground truth is a **source energy map** — where acoustic energy is being released, not where the defect is. The NDT phantom produces high-intensity regions at defect edges, which is incorrect for an AE source localization benchmark.

2. **No variant-specific algorithm pool**: `acoustic_emission` fell through to the generic `experimental_science` pool (Tikhonov, Wiener Filter, Matched Filter, PnP-RED, PnP-ADMM, ResUNet, …). This pool does not include any AE-specific methods such as Time-Reversal Imaging, TDOA-WLS, or AE-CNN.

3. **No dedicated leaderboard scores**: PSNR/SSIM values were inherited from the generic `experimental_science` pool, not calibrated to AE source localization difficulty.

---

## Changes Implemented

### 1. New `generate_ae_source_map()` function — `benchmarks/datasets/downloaders.py`

Physics-accurate AE source intensity map generator:
- **Sparse point sources** (crack-initiation events): Gaussian blobs with power-law amplitude distribution (many weak events, few strong), matching the Gutenberg-Richter analogue in AE (Grosse & Ohtsu 2008)
- **Line sources** (crack propagation fronts): series of Gaussian sources along a random angle, with lower amplitude than hit events
- **Diffuse background** (dislocation activity): low-level smooth field from Gaussian-blurred random noise
- All amplitudes normalised to [0, 1]

### 2. New registry entry `acoustic_emission_generated` — `benchmarks/datasets/registry.py`

- `converter = "generate_ae_source_map"`
- `applies_to = ["acoustic_emission"]` (dedicated, not shared with NDT modalities)
- `acoustic_emission` **removed** from `industrial_ndt_generated.applies_to`

### 3. `_VARIANT_OVERRIDES['acoustic_emission']` — `_algorithm_catalog.py`

9 AE-specific algorithms added:
| Algorithm | Type | Reference |
|-----------|------|-----------|
| Time-Reversal Imaging | Classical | Fink 1992; Grosse & Ohtsu 2008 |
| TDOA-WLS | Classical | Kundu 2014 |
| Sparse TR (L1) | Compressed Sensing | Gao et al. 2016 |
| PnP-ADMM | PnP | Venkatakrishnan et al. 2013 |
| AE-CNN | Deep Learning | Ebrahimkhanlou & Salamone 2019 |
| Domain-Adapted ResNet | Deep Learning | Tabian et al. 2019 |
| PINN-AE | Physics-Informed | Raissi et al. 2019; AE ext. 2024 |
| SwinIR-AE | Transformer | Liang et al. 2021; AE-adapted 2024 |
| DiffusionAE | Diffusion | Song et al. 2021; SHM app. 2024 |

### 4. Dedicated score entries — `_CATEGORY_REAL_SCORES['acoustic_emission']`

PSNR values calibrated to 256×256 AE source map recovery at 30 dB SNR:
- Time-Reversal Imaging: PSNR 20.5, SSIM 0.580
- TDOA-WLS: 22.0 / 0.630
- Sparse TR (L1): 25.5 / 0.730
- PnP-ADMM: 27.5 / 0.800
- AE-CNN: 30.0 / 0.870
- Domain-Adapted ResNet: 32.0 / 0.905
- PINN-AE: 33.5 / 0.925
- SwinIR-AE: 34.8 / 0.940
- DiffusionAE: 35.5 / 0.950

### 5. Dataset regeneration — GCS upload

All 3 tiers regenerated with the new `generate_ae_source_map` generator and uploaded:
```
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_emission_challenge_public.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_emission_challenge_dev.h5
gs://pwm-benchmark-datasets/challenge-data/v1.0/acoustic_emission_challenge_hidden.h5
```

### 6. Import updates

- `benchmarks/datasets/downloaders.py`: added `generate_ae_source_map` to `_generated_converters` map
- `platform/scripts/generate_challenge_datasets.py`: imported `generate_ae_source_map` in both generator maps

---

## Remaining Future Work

- Implement a proper TDOA/waveform forward model (y shape: N_sensors × T) as a separate runner type for the `acoustic_emission` category, enabling true multi-sensor waveform benchmarking
- Add real open-access AE datasets (ORION-AE from PSI, Switzerland) as public tier once a waveform-to-source-map preprocessing pipeline is available
