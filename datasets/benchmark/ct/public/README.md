# CT Public Tier

**Source:** LoDoPaB-CT real chest CT (LIDC/IDRI)
Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z
Zenodo record 3384092, CC BY 4.0.

**Access:** Full (GT + true spec + ideal sinogram)

## Mismatch Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `center_offset_px` | Δc — centre-of-rotation offset | [-2.0, 2.0] pixels |
| `angle_error_deg` | Δθ — systematic angle error | [-3.0, 3.0] degrees |
| `beam_hardening_beta` | β  — beam hardening coefficient | [0.0, 0.1]  |
| `detector_tilt_deg` | φ  — detector tilt | [-1.0, 1.0] degrees |

## Samples

| Sample | Scene | Views | Δc (px) | Δθ (°) | β | φ (°) |
|--------|-------|-------|---------|--------|---|-------|
| sample_00 | lidc_chest_00 | 60 | 0.086 | 0.623 | 0.047 | -0.594 |
| sample_01 | lidc_chest_01 | 60 | 1.305 | 0.826 | 0.046 | 0.242 |
| sample_02 | lidc_chest_02 | 60 | -1.031 | 1.734 | 0.090 | -0.076 |
| sample_03 | lidc_chest_03 | 60 | -0.179 | 0.048 | 0.069 | 0.289 |
| sample_04 | lidc_chest_04 | 60 | -0.426 | 1.344 | 0.059 | 0.614 |
| sample_05 | lidc_chest_05 | 60 | 0.204 | 1.549 | 0.095 | -0.138 |
| sample_06 | lidc_chest_06 | 60 | -0.745 | -2.948 | 0.049 | -0.228 |
| sample_07 | lidc_chest_07 | 60 | -1.586 | 2.714 | 0.020 | -0.661 |
| sample_08 | lidc_chest_08 | 60 | 1.135 | 0.597 | 0.098 | -0.795 |
| sample_09 | lidc_chest_09 | 60 | 0.348 | -0.673 | 0.027 | -0.823 |
| sample_10 | lidc_chest_10 | 60 | 0.275 | 0.723 | 0.007 | -0.140 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
