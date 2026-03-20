# MRI Benchmark Dataset

## Overview

Multi-coil brain MRI reconstruction benchmark using real M4Raw data
(Lyu et al., Scientific Data 2023). Standard Cartesian k-space
undersampling forward model with mismatch parameters.

## Forward Model

```
y = U_Omega * F * C * x + n

where:
  x: complex MR image (ground truth, 256x256)
  C: coil sensitivity maps (4 coils, estimated from calibration region)
  F: 2D Discrete Fourier Transform
  U_Omega: Cartesian undersampling mask (random k-space lines)
  n: complex Gaussian noise
```

## Mismatch Parameters

| Parameter | Description | Public | Dev | Hidden |
|-----------|-------------|--------|-----|--------|
| acceleration_factor | Undersampling ratio | 3-5x | 3-6x | 4-8x |
| noise_sigma | Complex noise std | 0.001-0.005 | 0.002-0.010 | 0.005-0.020 |
| coil_sensitivity_error | Coil map perturbation | 0-2% | 0-4% | 0-6% |
| off_resonance_hz | B0 inhomogeneity | +/-10 Hz | +/-20 Hz | +/-30 Hz |
| trajectory_deviation | k-space shift | +/-0.5% | +/-1% | +/-2% |

## Dataset Structure

```
mri/
├── README.md
├── generate_dataset.py
├── public/                    (12 samples, GT visible)
│   ├── mri_challenge_public.h5
│   ├── spec.json
│   ├── true_spec.json
│   └── images/sample_XX_*/
├── dev/                       (20 samples, blind eval)
│   ├── mri_challenge_dev.h5
│   ├── spec.json
│   ├── true_spec.json
│   └── images/sample_XX_*/
└── hidden/                    (20 samples, server-only)
    ├── mri_challenge_hidden.h5
    ├── spec.json
    ├── true_spec.json
    └── images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
├── x_true (256, 256) float32         # Ground truth (RSS image)
├── kspace_full (4, 256, 256) complex64  # Fully-sampled multi-coil k-space
├── kspace_undersampled (4, 256, 256) complex64  # Undersampled + noise
├── mask (256, 256) float32           # Undersampling mask
└── coil_maps (4, 256, 256) complex64 # Estimated coil sensitivity maps
```

## References

1. Lyu, M. et al. "M4Raw: A multi-contrast, multi-repetition, multi-channel
   MRI k-space dataset for reproducible AI research," Scientific Data, 2023.
2. Pruessmann, K. et al. "SENSE: Sensitivity encoding for fast MRI," MRM, 1999.
3. Lustig, M. et al. "Sparse MRI," MRM, 2007.
4. Zbontar, J. et al. "fastMRI: An open dataset and benchmarks for accelerated
   MRI," arXiv 2018.
