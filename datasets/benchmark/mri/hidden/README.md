# MRI Hidden Tier

## Source
Adversarial brain T2w phantoms (20 samples, severe mismatch)

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | proc_hidden_00             |   37.5 | 0.0099 | 0.092 | 0.0079 | 0.0384 | brain_t2_wm_lesions |
| sample_01  | proc_hidden_01             |   57.7 | 0.0097 | 0.131 | 0.0168 | 0.0483 | brain_t2_atrophy |
| sample_02  | proc_hidden_02             |   35.4 | 0.0066 | 0.150 | 0.0088 | 0.0345 | brain_t2_wm_lesions |
| sample_03  | proc_hidden_03             |   30.8 | 0.0054 | 0.071 | 0.0079 | 0.0332 | brain_t2_high_contrast |
| sample_04  | proc_hidden_04             |   56.6 | 0.0157 | 0.082 | 0.0059 | 0.0415 | brain_t2_wm_lesions |
| sample_05  | proc_hidden_05             |   51.5 | 0.0062 | 0.115 | 0.0077 | 0.0330 | brain_t2_fine_gyri |
| sample_06  | proc_hidden_06             |   34.2 | 0.0152 | 0.062 | 0.0148 | 0.0489 | brain_t2_wm_lesions |
| sample_07  | proc_hidden_07             |   21.5 | 0.0144 | 0.097 | 0.0075 | 0.0354 | brain_t2_wm_lesions |
| sample_08  | proc_hidden_08             |   32.7 | 0.0190 | 0.067 | 0.0054 | 0.0422 | brain_t2_wm_lesions |
| sample_09  | proc_hidden_09             |   50.0 | 0.0173 | 0.056 | 0.0152 | 0.0475 | brain_t2_high_contrast |
| sample_10  | proc_hidden_10             |   34.4 | 0.0194 | 0.077 | 0.0115 | 0.0340 | brain_t2_fine_gyri |
| sample_11  | proc_hidden_11             |   49.7 | 0.0101 | 0.145 | 0.0066 | 0.0526 | brain_t2_fine_gyri |
| sample_12  | proc_hidden_12             |   35.0 | 0.0195 | 0.088 | 0.0087 | 0.0389 | brain_t2_wm_lesions |
| sample_13  | proc_hidden_13             |   51.8 | 0.0159 | 0.062 | 0.0096 | 0.0448 | brain_t2_atrophy |
| sample_14  | proc_hidden_14             |   24.7 | 0.0096 | 0.139 | 0.0197 | 0.0530 | brain_t2_fine_gyri |
| sample_15  | proc_hidden_15             |   49.8 | 0.0052 | 0.065 | 0.0130 | 0.0574 | brain_t2_wm_lesions |
| sample_16  | proc_hidden_16             |   43.2 | 0.0083 | 0.139 | 0.0082 | 0.0551 | brain_t2_fine_gyri |
| sample_17  | proc_hidden_17             |   49.3 | 0.0064 | 0.054 | 0.0148 | 0.0385 | brain_t2_high_contrast |
| sample_18  | proc_hidden_18             |   23.8 | 0.0105 | 0.107 | 0.0198 | 0.0481 | brain_t2_wm_lesions |
| sample_19  | proc_hidden_19             |   30.0 | 0.0076 | 0.121 | 0.0050 | 0.0593 | brain_t2_wm_lesions |

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
