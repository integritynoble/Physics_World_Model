# MRI Public Tier

## Source
Shepp-Logan phantom variants (11 analytic samples)

## Per-Sample Mismatch Values

| Sample     | Scene                  | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | shepp_logan_00         |   10.2 | 0.0022 | 0.019 | 0.0014 | 0.0153 | shepp_logan |
| sample_01  | shepp_logan_01         |   13.3 | 0.0027 | 0.030 | 0.0017 | 0.0151 | shepp_logan |
| sample_02  | shepp_logan_02         |   14.6 | 0.0028 | 0.011 | 0.0011 | 0.0159 | shepp_logan |
| sample_03  | shepp_logan_03         |    9.0 | 0.0029 | 0.028 | 0.0011 | 0.0197 | shepp_logan |
| sample_04  | shepp_logan_04         |    6.7 | 0.0014 | 0.019 | 0.0014 | 0.0161 | shepp_logan |
| sample_05  | shepp_logan_05         |    8.4 | 0.0025 | 0.028 | 0.0026 | 0.0179 | shepp_logan |
| sample_06  | shepp_logan_06         |   12.6 | 0.0028 | 0.021 | 0.0029 | 0.0129 | shepp_logan |
| sample_07  | shepp_logan_07         |   12.4 | 0.0010 | 0.017 | 0.0024 | 0.0119 | shepp_logan |
| sample_08  | shepp_logan_08         |   14.6 | 0.0010 | 0.012 | 0.0029 | 0.0165 | shepp_logan |
| sample_09  | shepp_logan_09         |    9.6 | 0.0022 | 0.013 | 0.0012 | 0.0170 | shepp_logan |
| sample_10  | shepp_logan_10         |   13.8 | 0.0022 | 0.023 | 0.0019 | 0.0189 | shepp_logan |

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
