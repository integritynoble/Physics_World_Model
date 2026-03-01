# CT Hidden Tier

**Source:** LoDoPaB-CT real chest CT — **validation split, second half** (patients 64–127, LIDC/IDRI) + adversarial modifications
Leuschner et al. (2021), Sci Data 8:109, doi:10.1038/s41597-021-00893-z
Zenodo record 3384092, CC BY 4.0.
20 slices at indices [1846, 2067, 2120, 2131, 2221, 2245, 2376, 2380, 2510, 2573, 2768, 2912, 3043, 3053, 3116, 3180, 3265, 3343, 3392, 3506]
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
| sample_01 | lidc_val_h01_adversarial | 52 | 0.534 | 2.282 | 0.265 | 0.982 |
| sample_02 | lidc_val_h02_adversarial | 87 | -1.650 | -4.779 | 0.257 | 0.021 |
| sample_03 | lidc_val_h03_adversarial | 46 | 2.213 | 1.439 | 0.250 | -2.214 |
| sample_04 | lidc_val_h04_adversarial | 46 | -3.019 | 5.988 | 0.190 | 2.979 |
| sample_05 | lidc_val_h05_adversarial | 85 | -0.086 | 0.228 | 0.112 | -1.088 |
| sample_06 | lidc_val_h06_adversarial | 74 | -1.278 | -7.742 | 0.144 | -1.485 |
| sample_07 | lidc_val_h07_adversarial | 90 | 2.783 | 3.192 | 0.048 | 0.165 |
| sample_08 | lidc_val_h08_adversarial | 75 | 3.091 | 7.860 | 0.268 | -1.710 |
| sample_09 | lidc_val_h09_adversarial | 77 | 0.188 | 6.453 | 0.248 | -0.273 |
| sample_10 | lidc_val_h10_adversarial | 84 | -3.343 | -3.352 | 0.250 | -1.277 |
| sample_11 | lidc_val_h11_adversarial | 86 | 1.749 | -5.623 | 0.238 | -0.927 |
| sample_12 | lidc_val_h12_adversarial | 88 | 4.885 | -5.573 | 0.194 | 1.317 |
| sample_13 | lidc_val_h13_adversarial | 67 | 4.050 | -1.890 | 0.144 | -2.109 |
| sample_14 | lidc_val_h14_adversarial | 68 | -1.899 | -1.542 | 0.109 | -0.375 |
| sample_15 | lidc_val_h15_adversarial | 82 | 2.195 | 0.690 | 0.224 | 0.342 |
| sample_16 | lidc_val_h16_adversarial | 58 | -1.099 | -7.265 | 0.030 | 0.028 |
| sample_17 | lidc_val_h17_adversarial | 76 | 4.820 | 6.679 | 0.235 | -2.974 |
| sample_18 | lidc_val_h18_adversarial | 46 | 1.937 | -4.187 | 0.087 | 0.197 |
| sample_19 | lidc_val_h19_adversarial | 53 | 2.146 | -5.133 | 0.221 | 0.364 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
