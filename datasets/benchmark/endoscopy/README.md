# Endoscopy -- Fiber-Bundle Endoscopic Imaging

## Overview

Fiber-bundle endoscopy benchmark with realistic mucosal tissue phantoms
and clinical-grade degradation physics: barrel distortion, cos^4 vignetting,
motion/defocus blur, specular highlights, and Gaussian sensor noise.

## Forward Model

```
y = V(r) * D[H * x] + specular + noise

where:
    x_true              : 2D tissue reflectance map (256x256), range [0, 1]
    H                   : Gaussian blur PSF (motion/defocus)
    D                   : barrel distortion (Brown-Conrady, k1 coefficient)
    V(r)                : cos^4 radial vignetting
    specular            : bright specular highlight spots
    noise               : additive Gaussian sensor noise
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| Field of View | circular (endoscope aperture) |
| Distortion model | Brown-Conrady radial |
| Vignetting model | cos^4 law |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| distortion_k1 | Barrel distortion | 0-0.10 | 0-0.20 | 0-0.30 |
| vignetting_strength | Edge darkening | 0.1-0.3 | 0.1-0.45 | 0.1-0.6 |
| blur_sigma | Motion/defocus blur | 0.5-1.5 px | 0.5-2.5 px | 0.5-4.0 px |
| specular_fraction | Bright spots | 0-0.05 | 0-0.10 | 0-0.15 |
| noise_sigma | Sensor noise | 0.01-0.03 | 0.01-0.05 | 0.01-0.08 |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Normal mucosa | 4/tier | Base tissue texture + sparse vessels |
| Vessel-rich | 4/tier | Dense branching vascular network |
| Fold/rugae | 4/tier | Curved mucosal ridges (gastric/colonic) |
| Polyps (dev/hidden) | varies | Raised bumps (sessile/pedunculated) |
| Ulcers (hidden) | varies | Dark depressions with bright rim |

## Dataset Structure

```
endoscopy/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal + true spec visible)
+-- dev/       20 samples (blind eval, augmented + polyps)
+-- hidden/    20 samples (adversarial: ulcers + extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32          # Ground truth tissue reflectance [0, 1]
+-- image_ideal (256, 256) float32     # Degraded but noiseless image
+-- image_measured (256, 256) float32  # Fully degraded measurement
+-- reconstruction (256, 256) float32  # Baseline distortion-corr + Wiener + TV
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Brown, D.C. (1966) "Decentering distortion of lenses,"
   Photogrammetric Engineering 32, 444-462.
2. Vigneras, F. et al. (2020) "Endoscopic image enhancement: A review,"
   Computers in Biology and Medicine 120, 103738.
3. Ozyoruk, K.B. et al. (2021) "EndoSLAM dataset and unsupervised monocular
   visual odometry and depth estimation approach for endoscopic videos,"
   Medical Image Analysis 71, 102058.
4. PWM Benchmark: https://pwm.platformai.org/benchmark/endoscopy
