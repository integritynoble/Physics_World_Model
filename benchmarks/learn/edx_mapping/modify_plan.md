# Modify Plan: edx_mapping (STEM-EDX Elemental Mapping)

## Change Log

### 2026-03-09 — Full modality deployment

**Changes made:**

1. **Phantom generator** (`benchmarks/datasets/downloaders.py`):
   - Added `generate_edx_mapping_phantom()` — 64×64 float32 multi-phase material
     with Fe-rich phase (0.8-1.0), Si-rich inclusions (0.3-0.5), Al matrix (0.1-0.2)
   - EDX forward model: Poisson counting stats (~100-500 counts/pixel),
     Bremsstrahlung background (~10-30 counts), peak overlap Gaussian blur
   - Registered in both `_generated_converters` and `converter_map`

2. **Dataset registry** (`benchmarks/datasets/registry.py`):
   - Added `edx_mapping_generated` DatasetEntry (64×64, local, synthetic)

3. **Algorithm catalog** (`platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py`):
   - Added `_VARIANT_OVERRIDES["edx_mapping"]` with 9 domain-specific algorithms
     spanning Classical (MLS-EDX) → Diffusion (DiffEDX), 1995-2024 coverage
   - Added `CATEGORY_REAL_SCORES["edx_mapping"]` with realistic PSNR/SSIM values
     calibrated for low-count EDX denoising (PSNR range 22.3–39.4 dB)

4. **Runner routing** (`platform/scripts/generate_challenge_datasets.py`):
   - Added `"edx_mapping": "identity"` to `_VARIANT_TO_RUNNER`
   - Added `generate_edx_mapping_phantom` to import list and `gen_map`

5. **GCS datasets**: All 3 tiers generated and uploaded:
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/edx_mapping_challenge_public.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/edx_mapping_challenge_dev.h5`
   - `gs://pwm-benchmark-datasets/challenge-data/v1.0/edx_mapping_challenge_hidden.h5`

**Motivation:** Replace generic EM pool (4 algorithms) with 9 EDX-specific algorithms
covering the full algorithm evolution for elemental map denoising/reconstruction, from
classical maximum likelihood scaling (MLS-EDX, 1995) through modern diffusion models
(DiffEDX, 2024).

---

## Previous State (2026-03-06)

- **Category:** electron_microscopy
- **Carrier:** Electron
- **Score key:** em_generic
- **Algorithms served (EM generic pool, 4 total):**
  1. Wiener Filter (Classical) -- Analytical baseline
  2. BM3D (PnP) -- Dabov et al., IEEE TIP 2007
  3. Noise2Void (Deep Learning) -- Krull et al., CVPR 2019
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

**Status:** PASS — check.md written 2026-03-06

## Assessment

Reasonable match. STEM-EDX elemental mapping produces spectral images where
each pixel contains an X-ray energy spectrum. The primary reconstruction tasks
are: (1) spectral denoising (EDX signals are photon-starved), (2) peak
deconvolution for overlapping elemental lines, and (3) quantification
(Cliff-Lorimer or zeta-factor methods).

The generic EM denoising pool addresses task (1) well:

- Wiener Filter provides spectral denoising baseline.
- BM3D exploits spatial self-similarity in elemental maps (which often have
  repeated microstructural features).
- Noise2Void is directly applicable to low-dose STEM-EDX data.
- SwinIR provides strong spatial restoration.

The pool does not address peak deconvolution or quantification, but these are
handled by the forward model rather than the reconstruction algorithm pool.
For the image-quality-focused benchmark evaluation, denoising algorithms are
the correct category.

## Verdict (2026-03-06)

No code changes needed. The generic EM denoising pool is appropriate for
STEM-EDX elemental map restoration, which is fundamentally a low-SNR image
denoising problem.
