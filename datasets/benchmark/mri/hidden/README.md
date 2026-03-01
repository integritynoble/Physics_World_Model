# MRI Hidden Tier

## Source
Adversarial stress-test phantoms (20 samples, severe mismatch)

## Per-Sample Mismatch Values

| Sample     | Scene                  | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | proc_hidden_00         |   37.5 | 0.0099 | 0.092 | 0.0079 | 0.0384 | lesion_pathological |
| sample_01  | proc_hidden_01         |   27.6 | 0.0054 | 0.087 | 0.0111 | 0.0487 | fine_structure |
| sample_02  | proc_hidden_02         |   30.6 | 0.0170 | 0.065 | 0.0197 | 0.0374 | lesion_pathological |
| sample_03  | proc_hidden_03         |   32.2 | 0.0141 | 0.145 | 0.0155 | 0.0578 | high_contrast |
| sample_04  | proc_hidden_04         |   58.2 | 0.0189 | 0.107 | 0.0160 | 0.0525 | lesion_pathological |
| sample_05  | proc_hidden_05         |   57.8 | 0.0190 | 0.130 | 0.0191 | 0.0354 | edge_heavy |
| sample_06  | proc_hidden_06         |   48.0 | 0.0116 | 0.069 | 0.0150 | 0.0588 | lesion_pathological |
| sample_07  | proc_hidden_07         |   51.0 | 0.0173 | 0.082 | 0.0179 | 0.0523 | lesion_pathological |
| sample_08  | proc_hidden_08         |   25.3 | 0.0193 | 0.143 | 0.0066 | 0.0596 | lesion_pathological |
| sample_09  | proc_hidden_09         |   54.4 | 0.0124 | 0.077 | 0.0112 | 0.0443 | high_contrast |
| sample_10  | proc_hidden_10         |   50.0 | 0.0170 | 0.079 | 0.0126 | 0.0317 | edge_heavy |
| sample_11  | proc_hidden_11         |   53.9 | 0.0064 | 0.114 | 0.0071 | 0.0521 | high_contrast |
| sample_12  | proc_hidden_12         |   42.1 | 0.0101 | 0.135 | 0.0107 | 0.0562 | lesion_pathological |
| sample_13  | proc_hidden_13         |   34.2 | 0.0053 | 0.057 | 0.0146 | 0.0550 | fine_structure |
| sample_14  | proc_hidden_14         |   34.6 | 0.0188 | 0.135 | 0.0062 | 0.0314 | edge_heavy |
| sample_15  | proc_hidden_15         |   25.2 | 0.0135 | 0.127 | 0.0150 | 0.0501 | lesion_pathological |
| sample_16  | proc_hidden_16         |   46.5 | 0.0086 | 0.139 | 0.0158 | 0.0402 | edge_heavy |
| sample_17  | proc_hidden_17         |   40.4 | 0.0084 | 0.133 | 0.0067 | 0.0355 | fine_structure |
| sample_18  | proc_hidden_18         |   36.1 | 0.0162 | 0.130 | 0.0072 | 0.0364 | lesion_pathological |
| sample_19  | proc_hidden_19         |   49.6 | 0.0163 | 0.084 | 0.0190 | 0.0348 | lesion_pathological |

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
