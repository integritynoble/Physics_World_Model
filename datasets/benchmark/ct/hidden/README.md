# CT Hidden Tier

**Source:** LoDoPaB-CT real chest CT — **validation split, second half** (patients 64–127, LIDC/IDRI) + adversarial modifications
Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z
Zenodo record 3384092, CC BY 4.0.
20 slices at indices [1792, 1880, 1968, 2056, 2144, 2232, 2320, 2408, 2496, 2584, 2672, 2760, 2848, 2936, 3024, 3112, 3200, 3288, 3376, 3464]
Adversarial: metal inserts, low-contrast lesions, calcifications, high-contrast bone.

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
| sample_00 | lidc_val_h00_adversarial | 73 | -3.623 | 2.245 | 0.183 | -0.091 |
| sample_01 | lidc_val_h01_adversarial | 52 | 4.997 | -2.288 | 0.258 | 0.476 |
| sample_02 | lidc_val_h02_adversarial | 90 | 4.611 | 5.354 | 0.297 | 1.053 |
| sample_03 | lidc_val_h03_adversarial | 77 | 2.027 | 5.423 | 0.013 | 0.441 |
| sample_04 | lidc_val_h04_adversarial | 87 | -3.675 | 5.846 | 0.184 | 2.887 |
| sample_05 | lidc_val_h05_adversarial | 90 | -0.344 | 7.669 | 0.175 | -1.471 |
| sample_06 | lidc_val_h06_adversarial | 73 | -0.832 | -4.554 | 0.256 | -1.544 |
| sample_07 | lidc_val_h07_adversarial | 56 | 3.002 | 4.221 | 0.213 | -1.437 |
| sample_08 | lidc_val_h08_adversarial | 71 | -0.815 | 3.659 | 0.096 | 2.544 |
| sample_09 | lidc_val_h09_adversarial | 55 | -3.394 | 7.126 | 0.268 | 1.709 |
| sample_10 | lidc_val_h10_adversarial | 43 | -3.224 | 4.377 | 0.153 | -2.997 |
| sample_11 | lidc_val_h11_adversarial | 66 | -2.992 | 1.669 | 0.036 | -2.366 |
| sample_12 | lidc_val_h12_adversarial | 89 | 0.160 | 6.079 | 0.210 | 0.519 |
| sample_13 | lidc_val_h13_adversarial | 78 | -4.234 | -2.111 | 0.130 | -2.601 |
| sample_14 | lidc_val_h14_adversarial | 72 | -1.159 | 1.238 | 0.206 | -2.885 |
| sample_15 | lidc_val_h15_adversarial | 86 | 4.080 | -3.146 | 0.191 | -2.384 |
| sample_16 | lidc_val_h16_adversarial | 89 | -2.366 | -1.011 | 0.117 | -1.167 |
| sample_17 | lidc_val_h17_adversarial | 79 | -3.397 | -3.537 | 0.233 | 2.264 |
| sample_18 | lidc_val_h18_adversarial | 53 | -1.035 | -1.496 | 0.206 | 0.771 |
| sample_19 | lidc_val_h19_adversarial | 63 | 3.875 | -5.794 | 0.009 | -1.589 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
