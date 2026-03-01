# MRI Dev Tier

## Source
Procedural brain T2w axial phantoms (20 samples, mild mismatch)

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | proc_dev_00                |   13.6 | 0.0036 | 0.011 | 0.0030 | 0.0229 | brain_t2_supratentorial |
| sample_01  | proc_dev_01                |    9.9 | 0.0027 | 0.011 | 0.0016 | 0.0147 | brain_t2_supratentorial |
| sample_02  | proc_dev_02                |   10.2 | 0.0013 | 0.028 | 0.0014 | 0.0120 | brain_t2_frontal_slice |
| sample_03  | proc_dev_03                |   16.9 | 0.0044 | 0.047 | 0.0028 | 0.0275 | brain_t2_temporal_slice |
| sample_04  | proc_dev_04                |    8.7 | 0.0019 | 0.046 | 0.0012 | 0.0171 | brain_t2_supratentorial |
| sample_05  | proc_dev_05                |   14.3 | 0.0049 | 0.032 | 0.0027 | 0.0290 | brain_t2_supratentorial |
| sample_06  | proc_dev_06                |    5.8 | 0.0044 | 0.044 | 0.0018 | 0.0155 | brain_t2_elderly |
| sample_07  | proc_dev_07                |   13.2 | 0.0043 | 0.038 | 0.0045 | 0.0106 | brain_t2_elderly |
| sample_08  | proc_dev_08                |    8.6 | 0.0011 | 0.043 | 0.0031 | 0.0225 | brain_t2_posterior_fossa |
| sample_09  | proc_dev_09                |   17.9 | 0.0043 | 0.037 | 0.0043 | 0.0272 | brain_t2_supratentorial |
| sample_10  | proc_dev_10                |   19.0 | 0.0026 | 0.030 | 0.0042 | 0.0221 | brain_t2_temporal_slice |
| sample_11  | proc_dev_11                |   15.6 | 0.0024 | 0.022 | 0.0037 | 0.0155 | brain_t2_frontal_slice |
| sample_12  | proc_dev_12                |   10.3 | 0.0041 | 0.042 | 0.0031 | 0.0256 | brain_t2_elderly |
| sample_13  | proc_dev_13                |    7.4 | 0.0045 | 0.046 | 0.0037 | 0.0188 | brain_t2_csf_rich |
| sample_14  | proc_dev_14                |   14.2 | 0.0048 | 0.037 | 0.0038 | 0.0196 | brain_t2_posterior_fossa |
| sample_15  | proc_dev_15                |   15.6 | 0.0026 | 0.022 | 0.0033 | 0.0170 | brain_t2_temporal_slice |
| sample_16  | proc_dev_16                |    8.1 | 0.0024 | 0.026 | 0.0015 | 0.0223 | brain_t2_csf_rich |
| sample_17  | proc_dev_17                |    8.9 | 0.0048 | 0.019 | 0.0022 | 0.0247 | brain_t2_supratentorial |
| sample_18  | proc_dev_18                |   10.5 | 0.0032 | 0.011 | 0.0011 | 0.0115 | brain_t2_temporal_slice |
| sample_19  | proc_dev_19                |    6.4 | 0.0016 | 0.032 | 0.0043 | 0.0257 | brain_t2_csf_rich |

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
