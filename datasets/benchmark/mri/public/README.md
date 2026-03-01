# MRI Public Tier

## Source
Real multi-coil brain MRI (axial T2w, 320×320, 4→15 coils)
Source: local real_mri/multicoil_val — set REAL_MRI_ROOT for custom path

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | real_2022061203_T201_sl03  |   10.2 | 0.0022 | 0.019 | 0.0014 | 0.0153 | real_axt2 |
| sample_01  | real_2022061203_T201_sl04  |   10.5 | 0.0014 | 0.022 | 0.0019 | 0.0165 | real_axt2 |
| sample_02  | real_2022061203_T201_sl05  |    8.6 | 0.0022 | 0.013 | 0.0012 | 0.0146 | real_axt2 |
| sample_03  | real_2022061203_T201_sl06  |    7.1 | 0.0010 | 0.010 | 0.0017 | 0.0136 | real_axt2 |
| sample_04  | real_2022061203_T201_sl07  |    5.1 | 0.0018 | 0.013 | 0.0016 | 0.0130 | real_axt2 |
| sample_05  | real_2022061203_T201_sl08  |   13.0 | 0.0027 | 0.022 | 0.0018 | 0.0168 | real_axt2 |
| sample_06  | real_2022061203_T201_sl09  |   10.6 | 0.0017 | 0.026 | 0.0019 | 0.0127 | real_axt2 |
| sample_07  | real_2022061203_T201_sl10  |    8.8 | 0.0011 | 0.019 | 0.0026 | 0.0188 | real_axt2 |
| sample_08  | real_2022061203_T201_sl11  |   12.7 | 0.0018 | 0.025 | 0.0029 | 0.0184 | real_axt2 |
| sample_09  | real_2022061203_T201_sl12  |   12.9 | 0.0020 | 0.015 | 0.0015 | 0.0106 | real_axt2 |
| sample_10  | real_2022061203_T201_sl13  |   10.5 | 0.0011 | 0.012 | 0.0027 | 0.0170 | real_axt2 |

## HDF5 Datasets (per sample)

| Key           | Shape              | Dtype     | Description                             |
|---------------|--------------------|-----------|---------------------------------------------|
| `x_true`      | (320, 320)         | float32   | GT magnitude image [0, 1]               |
| `y_kspace`    | (15, 320, 320)     | complex64 | Undersampled k-space per coil           |
| `mask`        | (320,)             | uint8     | 1D ky undersampling mask                |
| `coil_maps`   | (15, 320, 320)     | complex64 | Nominal coil sensitivity maps           |
| `B0_map`      | (320, 320)         | float32   | True B0 field map (oracle)              |
| `warp_field`  | (2, 320, 320)      | float32   | True gradient warp (dy, dx) in pixels   |

## Image Files (per sample)

- `ground_truth.png`       — True MR magnitude image
- `rss_reconstruction.png` — Zero-filled RSS (shows aliasing artefacts)
- `kspace_magnitude.png`   — Log|y| averaged over coils
- `undersampling_mask.png` — Cartesian ky undersampling pattern
- `coil_sensitivity.png`   — Mosaic of |S_c| for all 15 coils (3×5 grid)
- `b0_map.png`             — B0 field inhomogeneity map
- `overview.png`           — 2×3 summary grid
- `spec.json`              — Per-sample mismatch specification
