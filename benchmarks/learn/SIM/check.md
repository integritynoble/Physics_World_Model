# SIM (Structured Illumination Microscopy) Benchmark Dataset

**Generated:** 2026-03-11
**Generator:** `datasets/benchmark/sim/generate_dataset.py`

---

## Dataset Summary

| Tier | Samples | Seed Offset | Mean PSNR (baseline) | Mean SSIM (baseline) |
|------|---------|-------------|---------------------|---------------------|
| Public | 12 | 0 | ~24.0 dB | ~0.37 |
| Dev | 20 | 10000 | ~23.3 dB | ~0.26 |
| Hidden | 20 | 20000 | ~24.2 dB | ~0.28 |

## Forward Model

SIM uses patterned sinusoidal illumination to extend the spatial frequency support of a
fluorescence microscope beyond the diffraction limit (2x lateral resolution improvement).

```
For each orientation theta in {0, 60, 120} degrees and phase phi_k in {0, 2pi/3, 4pi/3}:
    I_k(r) = 1 + m * cos(2*pi*f*r_hat + phi_k)       -- illumination pattern
    y_k    = Poisson(PSF * (I_k * x_true) * N + bg)   -- shot noise
           + Normal(0, sigma_readout)                   -- readout noise

Measurement: y = mean(y_0, y_1, ..., y_8)              -- average of 9 raw frames
```

### Parameters
- Image size: 256 x 256, pixel size: 50 nm
- PSF sigma: 2.5 px (Gaussian approximation)
- 3 orientations x 3 phases = 9 raw SIM frames
- Nominal pattern frequency: 0.15 cycles/pixel
- Background: 5 photons/pixel
- Readout noise: 2 electrons std

## Mismatch Parameters

| Parameter | Public Range | Dev Range | Hidden Range | Unit |
|-----------|-------------|-----------|--------------|------|
| `pattern_frequency_error` | [-0.05, 0.05] | [-0.08, 0.08] | [-0.12, 0.12] | fraction |
| `modulation_depth` | [0.7, 1.0] | [0.5, 1.0] | [0.3, 0.9] | -- |
| `phase_error_deg` | [-3, 3] | [-5, 5] | [-8, 8] | degrees |
| `noise_level` | [500, 2000] | [300, 2000] | [200, 1500] | photons |

## Phantoms

Three types of biological structures characteristic of fluorescence microscopy:

1. **Actin filaments** -- thin curved lines forming a cytoskeletal mesh with occasional branching (12-25 filaments, thickness 0.8-1.8 px)
2. **Mitochondrial networks** -- elongated tubular organelles with frequent branching and merging (8-18 tubules, thickness 1.5-3.0 px)
3. **Microtubules** -- relatively straight filaments radiating from a centrosomal organizing center (15-30 filaments, thickness 0.7-1.5 px)

## CPU Baseline

Wiener-filtered SIM reconstruction (simplified Gustafsson 2000 algorithm):
1. Phase-stepping separation of 3 frequency components per orientation
2. Frequency shift of separated components to correct positions
3. Wiener deconvolution for OTF compensation
4. Butterworth apodization to suppress ringing
5. Inverse FFT and post-processing

Expected PSNR: 20-27 dB (varies with noise level and mismatch severity).

## HDF5 Structure

```
sim_challenge_{tier}.h5
  /sample_00/
    x_true                (256, 256) float32  -- ground truth fluorophore distribution [0, 1]
    y                     (256, 256) float32  -- averaged measurement (mean of 9 frames)
    raw_frames            (9, 256, 256) float32  -- all 9 raw SIM frames
    H_ideal               (256, 256) float32  -- noiseless widefield (PSF * x_true)
    reconstruction_baseline (256, 256) float32  -- Wiener SIM baseline reconstruction
    attrs:
      metadata            -- JSON: scene name, shape, n_raw_frames, psnr/ssim baseline
      true_spec           -- JSON: actual mismatch parameters
      spec_ranges         -- JSON: tier mismatch ranges
  /sample_01/
    ...
```

## GCS Paths

```
gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/sim_challenge_public.h5   (32 MiB)
gs://pwm-benchmark-datasets/datasets/Benchmark/sim/dev/sim_challenge_dev.h5          (54 MiB)
gs://pwm-benchmark-datasets/datasets/Benchmark/sim/hidden/sim_challenge_hidden.h5    (54 MiB)
```

## Gallery

4 gallery scenes at:
```
platform/pwm_platform/static/img/benchmark_gallery/sim/scene_00/  -- actin filaments
platform/pwm_platform/static/img/benchmark_gallery/sim/scene_01/  -- mitochondrial network
platform/pwm_platform/static/img/benchmark_gallery/sim/scene_02/  -- microtubules
platform/pwm_platform/static/img/benchmark_gallery/sim/scene_03/  -- actin filaments
```

Each scene contains: gt.png, measurement_I.png (averaged), measurement_II.png (single raw frame), recon_I.png (Wiener baseline), recon_II.png (|GT - recon| error map).

## Reading Example

```python
import h5py, json, numpy as np

with h5py.File("sim_challenge_public.h5", "r") as f:
    grp = f["sample_00"]
    x_true     = grp["x_true"][:]              # (256, 256) float32
    y          = grp["y"][:]                    # (256, 256) float32 -- averaged SIM measurement
    raw_frames = grp["raw_frames"][:]           # (9, 256, 256) float32 -- 9 raw SIM frames
    H_ideal    = grp["H_ideal"][:]             # (256, 256) float32 -- noiseless widefield
    spec       = json.loads(grp.attrs["spec_ranges"])
    true_spec  = json.loads(grp.attrs["true_spec"])
```
