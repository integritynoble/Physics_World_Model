# MRI Dev Tier

## Source
Procedural brain-like phantoms (20 samples, natural tissue statistics)

## Per-Sample Mismatch Values

| Sample     | Scene                  | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | proc_dev_00            |   13.6 | 0.0036 | 0.011 | 0.0030 | 0.0229 | gray_white_matter |
| sample_01  | proc_dev_01            |   15.6 | 0.0036 | 0.044 | 0.0015 | 0.0168 | gray_white_matter |
| sample_02  | proc_dev_02            |   17.3 | 0.0025 | 0.019 | 0.0034 | 0.0286 | gray_white_matter |
| sample_03  | proc_dev_03            |   10.8 | 0.0050 | 0.028 | 0.0049 | 0.0139 | gray_white_matter |
| sample_04  | proc_dev_04            |   10.6 | 0.0030 | 0.030 | 0.0023 | 0.0265 | gray_white_matter |
| sample_05  | proc_dev_05            |   13.1 | 0.0048 | 0.021 | 0.0015 | 0.0240 | gray_white_matter |
| sample_06  | proc_dev_06            |   12.5 | 0.0011 | 0.018 | 0.0046 | 0.0219 | fat_saturated |
| sample_07  | proc_dev_07            |   13.3 | 0.0047 | 0.019 | 0.0034 | 0.0234 | fat_saturated |
| sample_08  | proc_dev_08            |   17.6 | 0.0044 | 0.027 | 0.0027 | 0.0149 | gray_white_matter |
| sample_09  | proc_dev_09            |    7.4 | 0.0048 | 0.034 | 0.0020 | 0.0220 | gray_white_matter |
| sample_10  | proc_dev_10            |    6.9 | 0.0036 | 0.033 | 0.0026 | 0.0162 | gray_white_matter |
| sample_11  | proc_dev_11            |   19.4 | 0.0044 | 0.020 | 0.0019 | 0.0221 | gray_white_matter |
| sample_12  | proc_dev_12            |   11.5 | 0.0032 | 0.043 | 0.0017 | 0.0157 | fat_saturated |
| sample_13  | proc_dev_13            |   18.4 | 0.0032 | 0.048 | 0.0020 | 0.0116 | with_vessels |
| sample_14  | proc_dev_14            |   18.2 | 0.0029 | 0.028 | 0.0011 | 0.0227 | with_vessels |
| sample_15  | proc_dev_15            |   19.8 | 0.0040 | 0.026 | 0.0012 | 0.0229 | gray_white_matter |
| sample_16  | proc_dev_16            |    8.5 | 0.0048 | 0.018 | 0.0048 | 0.0249 | with_vessels |
| sample_17  | proc_dev_17            |   13.9 | 0.0015 | 0.013 | 0.0020 | 0.0113 | gray_white_matter |
| sample_18  | proc_dev_18            |   16.8 | 0.0017 | 0.026 | 0.0015 | 0.0294 | gray_white_matter |
| sample_19  | proc_dev_19            |   16.4 | 0.0032 | 0.030 | 0.0049 | 0.0144 | with_vessels |

## HDF5 Datasets (per sample)

| Key           | Shape                        | Dtype     | Description                          |
|---------------|------------------------------|-----------|--------------------------------------|
| `x_true`      | (256, 256)                   | float32   | GT magnitude image [0, 1]            |
| `y_kspace`    | (8, 256, 256)                | complex64 | Undersampled k-space per coil        |
| `mask`        | (256,)                       | uint8     | 1D ky undersampling mask             |
| `coil_maps`   | (8, 256, 256)                | complex64 | **Nominal** coil sensitivity maps    |
| `B0_map`      | (256, 256)                   | float32   | True B0 field map (oracle)           |
| `warp_field`  | (2, 256, 256)                | float32   | True gradient warp (dy, dx) px       |

## Image Files (per sample)

- `ground_truth.png`       — True MR magnitude image
- `rss_reconstruction.png` — Zero-filled RSS (shows aliasing artefacts)
- `kspace_magnitude.png`   — Log|y| averaged over coils
- `undersampling_mask.png` — Cartesian ky undersampling pattern
- `coil_sensitivity.png`   — Mosaic of |S_c| for all 8 coils
- `b0_map.png`             — B0 field inhomogeneity map
- `overview.png`           — 2×3 summary grid
- `spec.json`              — Per-sample mismatch specification
