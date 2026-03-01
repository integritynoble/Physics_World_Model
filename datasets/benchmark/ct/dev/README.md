# CT Dev Tier

**Source:** LoDoPaB-CT real chest CT — **validation split, first half** (patients 0–63, LIDC/IDRI)
Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z
Zenodo record 3384092, CC BY 4.0.
20 slices at indices [20, 50, 172, 328, 441, 459, 604, 657, 799, 819, 904, 943, 977, 1093, 1126, 1153, 1419, 1585, 1760, 1787]
Completely different patients from public tier (test split).

**Access:** Blind (measured sinogram + spec ranges only)

## Mismatch Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `center_offset_px` | Δc — centre-of-rotation offset | [-3.0, 3.0] pixels |
| `angle_error_deg` | Δθ — systematic angle error | [-5.0, 5.0] degrees |
| `beam_hardening_beta` | β  — beam hardening coefficient | [0.0, 0.15]  |
| `detector_tilt_deg` | φ  — detector tilt | [-2.0, 2.0] degrees |

## Samples

| Sample | Scene | Views | Δc (px) | Δθ (°) | β | φ (°) |
|--------|-------|-------|---------|--------|---|-------|
| sample_00 | lidc_val_00 | 60 | -0.175 | 3.275 | 0.138 | -0.970 |
| sample_01 | lidc_val_01 | 60 | 2.513 | 4.620 | 0.065 | -1.840 |
| sample_02 | lidc_val_02 | 60 | -2.393 | 1.908 | 0.024 | -0.695 |
| sample_03 | lidc_val_03 | 60 | -1.003 | 0.846 | 0.107 | 1.161 |
| sample_04 | lidc_val_04 | 60 | 2.783 | -4.121 | 0.018 | 0.998 |
| sample_05 | lidc_val_05 | 60 | 1.124 | -0.485 | 0.132 | -1.047 |
| sample_06 | lidc_val_06 | 60 | -1.739 | 0.015 | 0.136 | 1.129 |
| sample_07 | lidc_val_07 | 60 | -2.682 | -3.027 | 0.006 | -1.145 |
| sample_08 | lidc_val_08 | 60 | -2.307 | 1.525 | 0.143 | 0.144 |
| sample_09 | lidc_val_09 | 60 | 2.712 | -4.499 | 0.144 | 0.621 |
| sample_10 | lidc_val_10 | 60 | -0.835 | -2.967 | 0.119 | 1.409 |
| sample_11 | lidc_val_11 | 60 | 1.500 | -2.154 | 0.137 | 1.437 |
| sample_12 | lidc_val_12 | 60 | 0.845 | 1.843 | 0.052 | 0.904 |
| sample_13 | lidc_val_13 | 60 | -0.546 | -4.155 | 0.025 | -1.831 |
| sample_14 | lidc_val_14 | 60 | 0.175 | -4.851 | 0.002 | 1.358 |
| sample_15 | lidc_val_15 | 60 | -0.779 | -1.491 | 0.119 | -1.308 |
| sample_16 | lidc_val_16 | 60 | -2.509 | -2.470 | 0.144 | -1.475 |
| sample_17 | lidc_val_17 | 60 | 2.966 | -1.080 | 0.076 | 1.211 |
| sample_18 | lidc_val_18 | 60 | -1.576 | -4.579 | 0.069 | 0.275 |
| sample_19 | lidc_val_19 | 60 | 0.558 | 2.144 | 0.132 | 0.401 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
