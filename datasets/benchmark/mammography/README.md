# Mammography -- 2-D Beer-Lambert X-ray Projection

## Overview

Full-field digital mammography (FFDM) benchmark with realistic breast tissue
phantoms and clinical-grade physics: Beer-Lambert attenuation, Poisson quantum
noise, X-ray scatter, and detector point-spread function.

## Forward Model

```
y_i = I_0 * exp(-mu(x, E) * breast_thickness) + scatter + noise

where:
    x_true           : 2D attenuation map (256x256) of breast tissue
    I_0              : incident X-ray fluence (Mo/Rh target, 25-35 kVp)
    mu(x, E)         : linear attenuation coefficient (cm^-1)
    breast_thickness : compressed breast thickness (3-6 cm)
    scatter          : low-frequency scatter background
    noise            : Poisson (quantum) + readout noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| pixel_size | 0.3 mm/px |
| FOV | 76.8 mm |
| I0_per_mGy | 300 photons/pixel/mGy |
| readout_noise | 2.0 electrons sigma |

## Tissue Attenuation Coefficients (20 keV)

| Tissue | mu (cm^-1) |
|--------|-----------|
| Adipose | 0.15 |
| Fibroglandular | 0.40 |
| Mass/Tumour | 0.50 |
| Calcification | 1.20 |
| Cooper's ligament | 0.30 |
| Skin | 0.35 |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| dose_mGy | Radiation dose | 1.0-3.0 mGy | 0.5-3.0 mGy | 0.3-3.0 mGy |
| scatter_fraction | Scatter / total | 0.10-0.25 | 0.10-0.30 | 0.10-0.40 |
| detector_blur_sigma | Detector PSF | 0.5-1.5 px | 0.5-2.0 px | 0.5-3.0 px |
| breast_thickness_cm | Compressed thickness | 3-6 cm | 3-6 cm | 3-6 cm |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Fatty (BI-RADS A/B) | 4/tier | Mostly adipose with scattered glandular |
| Dense (BI-RADS C/D) | 4/tier | Prominent fibroglandular tissue |
| Lesion | 4/tier | Masses + microcalcification clusters |

## Dataset Structure

```
mammography/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal projection + true spec visible)
+-- dev/       20 samples (blind eval, augmented variants)
+-- hidden/    20 samples (adversarial: micro-calcifications, extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32          # Ground truth attenuation map [0, 1]
+-- projection_ideal (256, 256) float32 # Clean projection (no noise/scatter)
+-- projection_measured (256, 256) float32 # Measured (noisy) projection
+-- reconstruction (256, 256) float32   # Baseline Wiener+TV reconstruction
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Dance, D.R. et al. (2000) "Additional factors for the estimation of mean
   glandular dose using the UK mammography dosimetry protocol,"
   Phys. Med. Biol. 45, 3225-3240.
2. Siddon, R.L. (1985) "Fast calculation of the exact radiological path
   for a three-dimensional CT array," Med. Phys. 12, 252-255.
3. Vedantham, S. et al. (2015) "Digital Breast Tomosynthesis: State of the
   Art," Radiology 277, 663-684.
4. PWM Benchmark: https://pwm.platformai.org/benchmark/mammography
