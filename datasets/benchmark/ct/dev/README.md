# CT Dev Tier

**Source:** LoDoPaB-CT real chest CT — **validation split, first half** (patients 0–63, LIDC/IDRI)
Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z
Zenodo record 3384092, CC BY 4.0.
20 slices at indices [0, 88, 176, 264, 352, 440, 528, 616, 704, 792, 880, 968, 1056, 1144, 1232, 1320, 1408, 1496, 1584, 1672]
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
| sample_01 | lidc_val_01 | 60 | -1.512 | -3.403 | 0.053 | -1.172 |
| sample_02 | lidc_val_02 | 60 | -1.331 | 0.387 | 0.016 | -1.921 |
| sample_03 | lidc_val_03 | 60 | 0.054 | 1.882 | 0.076 | 1.949 |
| sample_04 | lidc_val_04 | 60 | -0.375 | -1.480 | 0.103 | 1.398 |
| sample_05 | lidc_val_05 | 60 | 2.115 | 3.012 | 0.013 | -1.585 |
| sample_06 | lidc_val_06 | 60 | 2.475 | 3.406 | 0.130 | 0.454 |
| sample_07 | lidc_val_07 | 60 | -1.399 | 2.925 | 0.137 | 0.716 |
| sample_08 | lidc_val_08 | 60 | 0.856 | 0.680 | 0.067 | -0.818 |
| sample_09 | lidc_val_09 | 60 | -2.985 | 0.548 | 0.095 | -1.599 |
| sample_10 | lidc_val_10 | 60 | -1.495 | -0.443 | 0.100 | 1.396 |
| sample_11 | lidc_val_11 | 60 | 1.187 | 4.396 | 0.061 | -1.897 |
| sample_12 | lidc_val_12 | 60 | 0.347 | 4.438 | 0.017 | -0.814 |
| sample_13 | lidc_val_13 | 60 | -2.355 | -3.881 | 0.060 | -0.695 |
| sample_14 | lidc_val_14 | 60 | 1.364 | 4.607 | 0.029 | -0.733 |
| sample_15 | lidc_val_15 | 60 | -0.312 | 0.872 | 0.132 | -1.963 |
| sample_16 | lidc_val_16 | 60 | 2.052 | -1.679 | 0.115 | -0.149 |
| sample_17 | lidc_val_17 | 60 | 2.318 | 1.312 | 0.012 | 0.821 |
| sample_18 | lidc_val_18 | 60 | 0.882 | -0.349 | 0.143 | -1.506 |
| sample_19 | lidc_val_19 | 60 | 1.569 | 3.033 | 0.001 | 1.189 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
