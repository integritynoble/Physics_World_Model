# Fundus -- Retinal Fundus Photography

## Overview

Retinal fundus photography benchmark with realistic physics:
optical aberrations (defocus PSF) + illumination non-uniformity + media opacity
(cataract haze) + Gaussian sensor noise.

## Forward Model

```
y = H * (x * I(r)) + opacity * haze + noise

where:
    x           : ground-truth retinal image (256x256 grayscale, green channel)
    H           : point spread function (defocus aberration, Gaussian)
    I(r)        : illumination non-uniformity field (cosine-4th vignetting)
    opacity     : media opacity factor (cataract transmittance loss)
    haze        : low-frequency veiling glare from lens scatter
    noise       : additive Gaussian sensor noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| FOV | 30 degrees |
| pixel_size | 23.4 um/px |
| PSF_scale | ~15 px/diopter |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| defocus_diopters | Optical defocus | 0-2 D | 0-3.5 D | 0-5 D |
| illumination_nonuniformity | Vignetting strength | 0-0.15 | 0-0.25 | 0-0.40 |
| media_opacity | Cataract haze | 0-0.10 | 0-0.20 | 0-0.30 |
| noise_sigma | Sensor noise std | 0.01-0.03 | 0.01-0.05 | 0.01-0.08 |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Normal | 4/tier | Healthy retina: disc, vessels, macula, fovea |
| Pathological | 4/tier | Microaneurysms, hemorrhages, drusen, exudates |
| Varied anatomy | 4/tier | Variable disc size, CDR, vessel density |

## Dataset Structure

```
fundus/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal + measured + Wiener recon)
+-- dev/       20 samples (augmented, medium mismatch)
+-- hidden/    20 samples (adversarial: severe pathology, wide mismatch)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32            # Ground truth retinal image
+-- image_ideal (256, 256) float32       # Blurred + illumination (no noise/haze)
+-- image_measured (256, 256) float32    # Fully degraded fundus photograph
+-- psf (K, K) float32                   # Defocus PSF kernel
+-- illumination_field (256, 256) float32 # Non-uniform illumination map
+-- reconstruction_wiener (256, 256) float32 # Wiener deconvolution baseline
```

## CPU Baseline Reconstruction

Wiener deconvolution + illumination correction:
1. Estimate illumination from heavy low-pass filter of measured image
2. Divide out illumination estimate
3. Wiener deconvolution with PSF and noise regularisation
4. Post-filtering to suppress ringing

## References

1. Zhou et al. (2023) "A foundation model for generalizable disease detection
   from retinal images," Nature 622:156.
2. Li et al. (2024) "Fundus Image Enhancement via Structure-Preserving
   Diffusion Models," MICCAI 2024.
3. Frangi et al. (1998) "Multiscale vessel enhancement filtering," MICCAI 1998.
