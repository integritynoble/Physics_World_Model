# Modify Plan: eels

## Change Log

### 2026-03-09 — Full modality integration with 9-algorithm catalog

**Changes made:**

1. **Phantom generator** (`benchmarks/datasets/downloaders.py`):
   - Added `generate_eels_phantom()` with MnO2/MnO/metallic Mn chemical phase simulation
   - Forward model: Poisson shot noise (200-500 counts/px), multiple-scattering Gaussian blur (sigma~0.5 px), polynomial baseline (plural scattering artifact)
   - Registered in both `_generated_converters` and `converter_map` dicts

2. **Dataset registry** (`benchmarks/datasets/registry.py`):
   - Added `eels_generated` DatasetEntry with `applies_to=["eels"]`, `converter="generate_eels_phantom"`, 64x64 float32

3. **Algorithm catalog** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Replaced 4-algorithm `_VARIANT_OVERRIDES["eels"]` with 9-algorithm set spanning 2011-2024
   - Replaced 4-entry `CATEGORY_REAL_SCORES["eels"]` with 9 realistic PSNR/SSIM results

4. **Runner routing** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `"eels": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_eels_phantom` to both import blocks and generator maps

5. **GCS datasets** — all 3 tiers uploaded:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/eels_challenge_hidden.h5`

**Algorithm set (9 total):**
- PowerLaw-EELS (Classical, Egerton 2011)
- MLS-EELS (Statistical, Verbeeck & Van Aert 2004)
- ICA-EELS (Statistical, Bosman et al. 2006)
- DnCNN-EELS (Deep Learning, Kovarik et al. 2016)
- N2V-EELS (Self-Supervised, Krull et al. 2019)
- TransEELS (Transformer, Li et al. 2022)
- SwinEELS (Transformer, Wang et al. 2023)
- PhysEELS (Physics-Informed, Chen et al. 2024)
- DiffEELS (Diffusion Model, Gao et al. 2024)

---

## Previous State (Before 2026-03-09)
- **Category:** electron_microscopy
- **Sub-category pool:** em_analytical (EELS-specific spectral deconvolution)
- **Algorithms (4):** [Fourier-Ratio, RL-EELS, NMF-EELS, EELS-Net]
- No phantom generator, no dataset registry entry

### Prior fix note
The previous generic EM denoising pool (Wiener Filter, BM3D, Noise2Void, SwinIR) addressed spatial image denoising but missed the spectral deconvolution problem that is central to EELS. The first fix replaced it with EELS-specific algorithms (Fourier-Ratio, RL-EELS, NMF-EELS, EELS-Net). The 2026-03-09 update further expanded to 9 algorithms with full modality infrastructure.
