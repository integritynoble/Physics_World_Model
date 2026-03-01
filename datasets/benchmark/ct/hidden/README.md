# CT Hidden Tier

**Source:** Adversarial chest/abdomen phantoms (metal, calcification, low-contrast lesions)

**Access:** Server-only

## Mismatch Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `center_offset_px` | Δc — centre-of-rotation offset | [-5.0, 5.0] pixels |
| `angle_error_deg` | Δθ — systematic angle error | [-8.0, 8.0] degrees |
| `beam_hardening_beta` | β  — beam hardening coefficient | [0.0, 0.3]  |
| `detector_tilt_deg` | φ  — detector tilt | [-3.0, 3.0] degrees |

## Samples

| Sample | Scene | Views | Δc (px) | Δθ (°) | β | φ (°) |
|--------|-------|-------|---------|--------|---|-------|
| sample_00 | chest_upper_adversarial | 73 | -3.623 | 2.245 | 0.183 | -0.091 |
| sample_01 | chest_mid_adversarial | 52 | 4.052 | 3.295 | 0.024 | -2.350 |
| sample_02 | chest_lower_adversarial | 89 | 4.681 | -3.362 | 0.015 | 2.887 |
| sample_03 | abdomen_upper_adversarial | 89 | 2.291 | 4.101 | 0.164 | -0.194 |
| sample_04 | abdomen_mid_adversarial | 72 | 4.736 | -6.356 | 0.102 | 2.700 |
| sample_05 | chest_upper_adversarial | 54 | -3.874 | -2.279 | 0.195 | 0.636 |
| sample_06 | chest_mid_adversarial | 65 | -0.254 | 6.470 | 0.111 | -0.594 |
| sample_07 | chest_lower_adversarial | 67 | -1.296 | -0.678 | 0.145 | -0.815 |
| sample_08 | abdomen_upper_adversarial | 40 | 4.568 | 0.020 | 0.132 | -1.444 |
| sample_09 | abdomen_mid_adversarial | 77 | -2.379 | 0.278 | 0.227 | 0.066 |
| sample_10 | chest_upper_adversarial | 76 | 0.756 | -5.742 | 0.284 | 2.473 |
| sample_11 | chest_mid_adversarial | 90 | 0.885 | 6.254 | 0.212 | 2.632 |
| sample_12 | chest_lower_adversarial | 40 | -4.294 | 3.426 | 0.281 | -1.915 |
| sample_13 | abdomen_upper_adversarial | 53 | 0.598 | 0.332 | 0.074 | -0.759 |
| sample_14 | abdomen_mid_adversarial | 59 | 1.541 | 6.854 | 0.141 | -2.740 |
| sample_15 | chest_upper_adversarial | 49 | 3.467 | -7.789 | 0.185 | -0.451 |
| sample_16 | chest_mid_adversarial | 41 | -4.671 | -3.195 | 0.122 | -1.848 |
| sample_17 | chest_lower_adversarial | 53 | 2.201 | -5.383 | 0.030 | -1.085 |
| sample_18 | abdomen_upper_adversarial | 83 | 2.967 | -3.995 | 0.055 | -1.379 |
| sample_19 | abdomen_mid_adversarial | 59 | -1.261 | -6.187 | 0.213 | -2.435 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
