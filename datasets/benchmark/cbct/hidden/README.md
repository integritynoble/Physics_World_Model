# CBCT Hidden Tier

**Source:** Adversarial dental/head CBCT phantoms (20 samples)

**Access:** Server-only

## Mismatch Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `scatter_fraction` | Scatter-to-primary ratio | [0.3, 0.6]  |
| `truncation_fov_factor` | FOV truncation factor | [0.7, 1.0]  |
| `ring_artifact_amplitude` | Ring artifact amplitude | [0.0, 0.05]  |
| `rotation_offset_deg` | Rotation centre offset | [0.0, 3.0] degrees |

## Samples

| Sample | Scene | Views | Scatter | Truncation | Ring | Rot Offset |
|--------|-------|-------|---------|------------|------|------------|
| sample_00 | cbct_hid_00_dental_panoramic_adversarial | 200 | 0.341 | 0.892 | 0.0305 | 1.454 |
| sample_01 | cbct_hid_01_head_axial_adversarial | 149 | 0.552 | 0.755 | 0.0072 | 0.046 |
| sample_02 | cbct_hid_02_dental_mixed_adversarial | 182 | 0.538 | 0.942 | 0.0288 | 1.867 |
| sample_03 | cbct_hid_03_head_lower_adversarial | 148 | 0.584 | 0.762 | 0.0225 | 1.391 |
| sample_04 | cbct_hid_04_dental_panoramic_adversarial | 135 | 0.512 | 0.774 | 0.0182 | 2.027 |
| sample_05 | cbct_hid_05_head_axial_adversarial | 134 | 0.355 | 0.955 | 0.0074 | 0.465 |
| sample_06 | cbct_hid_06_dental_mixed_adversarial | 234 | 0.594 | 0.745 | 0.0369 | 2.363 |
| sample_07 | cbct_hid_07_head_lower_adversarial | 145 | 0.491 | 0.991 | 0.0433 | 0.750 |
| sample_08 | cbct_hid_08_dental_panoramic_adversarial | 236 | 0.524 | 0.745 | 0.0198 | 2.199 |
| sample_09 | cbct_hid_09_head_axial_adversarial | 217 | 0.591 | 0.962 | 0.0393 | 1.798 |
| sample_10 | cbct_hid_10_dental_mixed_adversarial | 199 | 0.364 | 0.843 | 0.0210 | 2.029 |
| sample_11 | cbct_hid_11_head_lower_adversarial | 183 | 0.309 | 0.758 | 0.0137 | 2.708 |
| sample_12 | cbct_hid_12_dental_panoramic_adversarial | 151 | 0.535 | 0.730 | 0.0134 | 2.668 |
| sample_13 | cbct_hid_13_head_axial_adversarial | 130 | 0.413 | 0.705 | 0.0164 | 0.030 |
| sample_14 | cbct_hid_14_dental_mixed_adversarial | 199 | 0.571 | 0.719 | 0.0470 | 0.938 |
| sample_15 | cbct_hid_15_head_lower_adversarial | 140 | 0.579 | 0.748 | 0.0086 | 2.229 |
| sample_16 | cbct_hid_16_dental_panoramic_adversarial | 182 | 0.437 | 0.748 | 0.0104 | 1.236 |
| sample_17 | cbct_hid_17_head_axial_adversarial | 186 | 0.440 | 0.961 | 0.0151 | 0.907 |
| sample_18 | cbct_hid_18_dental_mixed_adversarial | 161 | 0.507 | 0.971 | 0.0190 | 2.360 |
| sample_19 | cbct_hid_19_head_lower_adversarial | 236 | 0.553 | 0.894 | 0.0282 | 0.533 |

## HDF5 Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `x_true` | (256, 256) | Ground-truth attenuation |
| `sinogram_ideal` | (n_views, n_det) | Ideal sinogram (nepers) |
| `sinogram_measured` | (n_views, n_det) | Measured sinogram (mismatch + noise) |
| `angles_nominal` | (n_views,) | Projection angles [rad] |
