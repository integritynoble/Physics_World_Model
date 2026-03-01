# CT Dev Tier

**Source:** Procedural chest/abdomen phantoms — anatomy and HU scale match LoDoPaB-CT

**Access:** Blind (measured sinogram + spec ranges)

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
| sample_00 | chest_upper | 60 | -0.175 | 3.275 | 0.138 | -0.970 |
| sample_01 | chest_mid | 60 | -2.986 | -3.604 | 0.075 | 1.263 |
| sample_02 | chest_lower | 60 | -1.824 | -2.626 | 0.150 | 0.314 |
| sample_03 | abdomen_upper | 60 | 1.961 | -3.093 | 0.090 | 1.323 |
| sample_04 | abdomen_mid | 60 | -2.053 | 2.810 | 0.092 | 0.319 |
| sample_05 | chest_upper | 60 | 1.453 | 1.145 | 0.122 | 0.745 |
| sample_06 | chest_mid | 60 | -2.413 | 2.880 | 0.057 | 0.475 |
| sample_07 | chest_lower | 60 | -1.705 | -4.198 | 0.133 | 1.519 |
| sample_08 | abdomen_upper | 60 | 0.910 | 4.226 | 0.075 | -1.179 |
| sample_09 | abdomen_mid | 60 | 0.870 | -1.095 | 0.134 | -1.238 |
| sample_10 | chest_upper | 60 | -2.782 | -1.513 | 0.099 | 0.417 |
| sample_11 | chest_mid | 60 | -1.176 | -4.455 | 0.068 | 0.408 |
| sample_12 | chest_lower | 60 | -2.062 | 3.364 | 0.074 | 0.930 |
| sample_13 | abdomen_upper | 60 | -0.442 | -0.020 | 0.049 | 1.206 |
| sample_14 | abdomen_mid | 60 | 0.139 | 0.201 | 0.099 | -0.545 |
| sample_15 | chest_upper | 60 | 1.182 | 1.666 | 0.123 | -1.474 |
| sample_16 | chest_mid | 60 | 2.904 | -3.804 | 0.117 | 0.217 |
| sample_17 | chest_lower | 60 | -0.521 | -2.232 | 0.081 | -1.488 |
| sample_18 | abdomen_upper | 60 | -2.069 | -4.945 | 0.146 | -0.828 |
| sample_19 | abdomen_mid | 60 | -1.844 | 4.801 | 0.011 | -1.435 |

## HDF5 Datasets per Sample

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `x_true` | (362, 362) | float32 | Ground-truth attenuation, x=(HU+1000)/4071 |
| `sinogram_ideal` | (n_views, 736) | float32 | Ideal fan-beam sinogram (nepers) |
| `sinogram_measured` | (n_views, 736) | float32 | Measured sinogram (mismatch + noise, nepers) |
| `angles_nominal` | (n_views,) | float32 | Nominal projection angles [rad] |
