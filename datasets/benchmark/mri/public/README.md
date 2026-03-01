# MRI Public Tier

## Source
Synthetic Shepp-Logan variants (PLACEHOLDER — set FASTMRI_ROOT for real data)

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | shepp_logan_00             |   10.2 | 0.0022 | 0.019 | 0.0014 | 0.0153 | shepp_logan |
| sample_01  | shepp_logan_01             |   10.5 | 0.0014 | 0.022 | 0.0019 | 0.0165 | shepp_logan |
| sample_02  | shepp_logan_02             |    8.6 | 0.0022 | 0.013 | 0.0012 | 0.0146 | shepp_logan |
| sample_03  | shepp_logan_03             |    7.1 | 0.0010 | 0.010 | 0.0017 | 0.0136 | shepp_logan |
| sample_04  | shepp_logan_04             |    5.1 | 0.0018 | 0.013 | 0.0016 | 0.0130 | shepp_logan |
| sample_05  | shepp_logan_05             |   13.0 | 0.0027 | 0.022 | 0.0018 | 0.0168 | shepp_logan |
| sample_06  | shepp_logan_06             |   10.6 | 0.0017 | 0.026 | 0.0019 | 0.0127 | shepp_logan |
| sample_07  | shepp_logan_07             |    8.8 | 0.0011 | 0.019 | 0.0026 | 0.0188 | shepp_logan |
| sample_08  | shepp_logan_08             |   12.7 | 0.0018 | 0.025 | 0.0029 | 0.0184 | shepp_logan |
| sample_09  | shepp_logan_09             |   12.9 | 0.0020 | 0.015 | 0.0015 | 0.0106 | shepp_logan |
| sample_10  | shepp_logan_10             |   10.5 | 0.0011 | 0.012 | 0.0027 | 0.0170 | shepp_logan |

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
