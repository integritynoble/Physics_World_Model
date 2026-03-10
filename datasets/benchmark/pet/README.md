# PET -- 2-D Parallel-Beam Emission Tomography

## Overview

Positron Emission Tomography (PET) benchmark with realistic physics:
Radon transform + attenuation + scatter + random coincidences + Poisson noise.

## Forward Model

```
y_i ~ Poisson(a_i * [A * x]_i + r_i + s_i)

where:
    x       : activity map (ground truth, 256x256)
    A       : system matrix (parallel-beam Radon transform, 256 angles)
    a_i     : attenuation correction factors (from mu-map)
    r_i     : random coincidences (uniform background)
    s_i     : scatter contribution (smooth background)
    y_i     : measured sinogram (counts)
```

## Geometry

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 256 x 256 |
| n_angles | 256 |
| n_det | 367 |
| angle_range | [0, 180) degrees |
| FOV | 220 mm |
| pixel_size | 0.86 mm/px |

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| count_rate_mcps | Total count rate | 2-5 Mcps | 1-5 Mcps | 0.5-5 Mcps |
| scatter_fraction | Scatter / total | 0.30-0.40 | 0.30-0.45 | 0.30-0.55 |
| randoms_fraction | Randoms / total | 0.10-0.25 | 0.10-0.35 | 0.10-0.50 |
| attenuation_error | Relative mu-map error | 0-3% | 0-6% | 0-10% |

## Phantoms

| Type | Samples | Description |
|------|---------|-------------|
| Brain FDG | 4/tier | Modified Shepp-Logan with gray/white matter, lesions |
| Body | 4/tier | Torso with organs, lungs, heart, lesions |
| Cardiac | 4/tier | Myocardial perfusion with defects |

## Dataset Structure

```
pet/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (GT + ideal sino + true spec visible)
+-- dev/       20 samples (blind eval, augmented variants)
+-- hidden/    20 samples (adversarial: micro-lesions, extreme params)
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         # Ground truth activity map
+-- sinogram_ideal (256, 367) float32  # Clean Radon sinogram (no noise/scatter)
+-- sinogram_measured (256, 367) float32 # Measured sinogram (Poisson + scatter + randoms)
+-- attenuation_map (256, 256) float32 # Mu-map (attenuation coefficients)
+-- angles_deg (256,) float32          # Projection angles in degrees
+-- reconstruction_fbp (256, 256) float32 # FBP baseline reconstruction
```

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

1. Shepp, L.A. & Vardi, Y. (1982) "Maximum Likelihood Reconstruction for
   Emission Tomography," IEEE TMI.
2. Hudson, H.M. & Larkin, R.S. (1994) "Accelerated Image Reconstruction
   Using Ordered Subsets of Projection Data," IEEE TMI.
3. Reader, A.J. & Verhaeghe, J. (2014) "4D image reconstruction for
   emission tomography," PMB.
