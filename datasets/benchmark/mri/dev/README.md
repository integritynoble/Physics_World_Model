# MRI Dev Tier

## Source
IXI T2w brain MRI — 578 healthy subjects, 3 sites, 1.5 T / 3 T
Sites: Hammersmith (3 T Philips), Guy's (1.5 T Philips), IOP (1.5 T GE)
Licence: CC BY-SA 3.0
Reference: brain-development.org/ixi-dataset/
Download:  python download_datasets.py --ixi-dir ~/pwm_data/ixi_t2

## Per-Sample Mismatch Values

| Sample     | Scene                      | B0 (Hz) | grad_nonlin | coil_sens | k_traj | noise_σ | recipe |
|------------|----------------------------|---------|-------------|-----------|--------|---------|--------|
| sample_00  | ixi_IXI012-HH-1211-T2_sl055 |   13.6 | 0.0036 | 0.011 | 0.0030 | 0.0229 | ixi_t2_hammersmith_3T |
| sample_01  | ixi_IXI002-Guys-0828-T2_sl039 |    9.9 | 0.0027 | 0.011 | 0.0016 | 0.0147 | ixi_t2_guys_1.5T |
| sample_02  | ixi_IXI035-IOP-0873-T2_sl067 |   10.2 | 0.0013 | 0.028 | 0.0014 | 0.0120 | ixi_t2_iop_1.5T |
| sample_03  | ixi_IXI013-HH-1212-T2_sl057 |   16.9 | 0.0044 | 0.047 | 0.0028 | 0.0275 | ixi_t2_hammersmith_3T |
| sample_04  | ixi_IXI016-Guys-0697-T2_sl083 |    8.7 | 0.0019 | 0.046 | 0.0012 | 0.0171 | ixi_t2_guys_1.5T |
| sample_05  | ixi_IXI230-IOP-0869-T2_sl067 |   14.3 | 0.0049 | 0.032 | 0.0027 | 0.0290 | ixi_t2_iop_1.5T |
| sample_06  | ixi_IXI014-HH-1236-T2_sl018 |    5.8 | 0.0044 | 0.044 | 0.0018 | 0.0155 | ixi_t2_hammersmith_3T |
| sample_07  | ixi_IXI017-Guys-0698-T2_sl086 |   13.2 | 0.0043 | 0.038 | 0.0045 | 0.0106 | ixi_t2_guys_1.5T |
| sample_08  | ixi_IXI231-IOP-0866-T2_sl040 |    8.6 | 0.0011 | 0.043 | 0.0031 | 0.0225 | ixi_t2_iop_1.5T |
| sample_09  | ixi_IXI015-HH-1258-T2_sl036 |   17.9 | 0.0043 | 0.037 | 0.0043 | 0.0272 | ixi_t2_hammersmith_3T |
| sample_10  | ixi_IXI019-Guys-0702-T2_sl090 |   19.0 | 0.0026 | 0.030 | 0.0042 | 0.0221 | ixi_t2_guys_1.5T |
| sample_11  | ixi_IXI232-IOP-0898-T2_sl040 |   15.6 | 0.0024 | 0.022 | 0.0037 | 0.0155 | ixi_t2_iop_1.5T |
| sample_12  | ixi_IXI033-HH-1259-T2_sl059 |   10.3 | 0.0041 | 0.042 | 0.0031 | 0.0256 | ixi_t2_hammersmith_3T |
| sample_13  | ixi_IXI020-Guys-0700-T2_sl083 |    7.4 | 0.0045 | 0.046 | 0.0037 | 0.0188 | ixi_t2_guys_1.5T |
| sample_14  | ixi_IXI233-IOP-0875-T2_sl040 |   14.2 | 0.0048 | 0.037 | 0.0038 | 0.0196 | ixi_t2_iop_1.5T |
| sample_15  | ixi_IXI034-HH-1260-T2_sl059 |   15.6 | 0.0026 | 0.022 | 0.0033 | 0.0170 | ixi_t2_hammersmith_3T |
| sample_16  | ixi_IXI021-Guys-0703-T2_sl090 |    8.1 | 0.0024 | 0.026 | 0.0015 | 0.0223 | ixi_t2_guys_1.5T |
| sample_17  | ixi_IXI234-IOP-0870-T2_sl094 |    8.9 | 0.0048 | 0.019 | 0.0022 | 0.0247 | ixi_t2_iop_1.5T |
| sample_18  | ixi_IXI039-HH-1261-T2_sl036 |   10.5 | 0.0032 | 0.011 | 0.0011 | 0.0115 | ixi_t2_hammersmith_3T |
| sample_19  | ixi_IXI022-Guys-0701-T2_sl039 |    6.4 | 0.0016 | 0.032 | 0.0043 | 0.0257 | ixi_t2_guys_1.5T |

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
