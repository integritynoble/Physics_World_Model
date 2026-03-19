# PWMI-CASSI Statistical Summary

## Paired t-tests: UPWMI vs baselines (PSNR)

| Family | Severity | Test | UPWMI PSNR | Other PSNR | Diff | t-stat | p-value | Sig? | Cohen's d | Effect |
|--------|----------|------|------------|------------|------|--------|---------|------|-----------|--------|
| disp_step | mild | upwmi_vs_no_calibration | 25.55 | 20.10 | +5.45 | 19.27 | 0.0000 | Yes | 8.62 | large |
| disp_step | mild | upwmi_vs_grid_search | 25.55 | 23.30 | +2.25 | 8.45 | 0.0000 | Yes | 3.78 | large |
| disp_step | mild | upwmi_vs_gradient_descent | 25.55 | 23.63 | +1.92 | 7.20 | 0.0000 | Yes | 3.22 | large |
| disp_step | mild | upwmi_gradient_vs_upwmi | 25.69 | 25.55 | +0.14 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| disp_step | moderate | upwmi_vs_no_calibration | 24.71 | 19.72 | +4.99 | 15.08 | 0.0000 | Yes | 6.75 | large |
| disp_step | moderate | upwmi_vs_grid_search | 24.71 | 22.91 | +1.80 | 5.63 | 0.0000 | Yes | 2.52 | large |
| disp_step | moderate | upwmi_vs_gradient_descent | 24.71 | 23.51 | +1.20 | 3.75 | 0.0002 | Yes | 1.67 | large |
| disp_step | moderate | upwmi_gradient_vs_upwmi | 25.34 | 24.71 | +0.63 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| disp_step | severe | upwmi_vs_no_calibration | 23.74 | 19.18 | +4.56 | 21.04 | 0.0000 | Yes | 9.41 | large |
| disp_step | severe | upwmi_vs_grid_search | 23.74 | 22.17 | +1.57 | 7.92 | 0.0000 | Yes | 3.54 | large |
| disp_step | severe | upwmi_vs_gradient_descent | 23.74 | 22.58 | +1.16 | 5.87 | 0.0000 | Yes | 2.63 | large |
| disp_step | severe | upwmi_gradient_vs_upwmi | 24.64 | 23.74 | +0.90 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| mask_shift | mild | upwmi_vs_no_calibration | 25.76 | 20.49 | +5.27 | 23.49 | 0.0000 | Yes | 10.51 | large |
| mask_shift | mild | upwmi_vs_grid_search | 25.76 | 23.89 | +1.88 | 9.00 | 0.0000 | Yes | 4.03 | large |
| mask_shift | mild | upwmi_vs_gradient_descent | 25.76 | 24.61 | +1.15 | 5.70 | 0.0000 | Yes | 2.55 | large |
| mask_shift | mild | upwmi_gradient_vs_upwmi | 26.31 | 25.76 | +0.55 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| mask_shift | moderate | upwmi_vs_no_calibration | 24.37 | 19.10 | +5.27 | 17.48 | 0.0000 | Yes | 7.82 | large |
| mask_shift | moderate | upwmi_vs_grid_search | 24.37 | 21.85 | +2.52 | 8.53 | 0.0000 | Yes | 3.81 | large |
| mask_shift | moderate | upwmi_vs_gradient_descent | 24.37 | 22.45 | +1.92 | 6.58 | 0.0000 | Yes | 2.94 | large |
| mask_shift | moderate | upwmi_gradient_vs_upwmi | 24.24 | 24.37 | -0.13 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| mask_shift | severe | upwmi_vs_no_calibration | 26.32 | 21.35 | +4.97 | 15.78 | 0.0000 | Yes | 7.06 | large |
| mask_shift | severe | upwmi_vs_grid_search | 26.32 | 24.03 | +2.29 | 7.34 | 0.0000 | Yes | 3.28 | large |
| mask_shift | severe | upwmi_vs_gradient_descent | 26.32 | 24.63 | +1.70 | 5.61 | 0.0000 | Yes | 2.51 | large |
| mask_shift | severe | upwmi_gradient_vs_upwmi | 26.54 | 26.32 | +0.22 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| PSF_blur | mild | upwmi_vs_no_calibration | 25.33 | 20.81 | +4.52 | 17.60 | 0.0000 | Yes | 7.87 | large |
| PSF_blur | mild | upwmi_vs_grid_search | 25.33 | 23.56 | +1.77 | 7.18 | 0.0000 | Yes | 3.21 | large |
| PSF_blur | mild | upwmi_vs_gradient_descent | 25.33 | 23.92 | +1.41 | 5.82 | 0.0000 | Yes | 2.60 | large |
| PSF_blur | mild | upwmi_gradient_vs_upwmi | 25.58 | 25.33 | +0.25 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| PSF_blur | moderate | upwmi_vs_no_calibration | 27.04 | 21.53 | +5.51 | 18.17 | 0.0000 | Yes | 8.13 | large |
| PSF_blur | moderate | upwmi_vs_grid_search | 27.04 | 24.68 | +2.36 | 8.01 | 0.0000 | Yes | 3.58 | large |
| PSF_blur | moderate | upwmi_vs_gradient_descent | 27.04 | 25.30 | +1.74 | 6.01 | 0.0000 | Yes | 2.69 | large |
| PSF_blur | moderate | upwmi_gradient_vs_upwmi | 26.68 | 27.04 | -0.36 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |
| PSF_blur | severe | upwmi_vs_no_calibration | 24.94 | 19.69 | +5.26 | 17.20 | 0.0000 | Yes | 7.69 | large |
| PSF_blur | severe | upwmi_vs_grid_search | 24.94 | 22.77 | +2.18 | 7.34 | 0.0000 | Yes | 3.28 | large |
| PSF_blur | severe | upwmi_vs_gradient_descent | 24.94 | 23.14 | +1.80 | 6.14 | 0.0000 | Yes | 2.74 | large |
| PSF_blur | severe | upwmi_gradient_vs_upwmi | 24.87 | 24.94 | -0.07 | 0.00 | 1.0000 | No | 0.00 | n/a (single comparison) |

## CI Coverage Analysis

| Family | Severity | CI [low, high] | Width | PSNR Gain | Covers? | theta-err before | theta-err after | Reduction |
|--------|----------|----------------|-------|-----------|---------|-----------------|-----------------|-----------|
| disp_step | mild | [1.62, 3.40] | 1.779 | 2.51 | Yes | 1.8864 | 0.4194 | 1.4670 |
| disp_step | moderate | [2.48, 4.10] | 1.618 | 3.29 | Yes | 1.6626 | 0.3683 | 1.2942 |
| disp_step | severe | [2.51, 3.59] | 1.084 | 3.05 | Yes | 1.8576 | 0.3329 | 1.5248 |
| mask_shift | mild | [1.61, 4.53] | 2.919 | 3.07 | Yes | 1.3020 | 0.3688 | 0.9332 |
| mask_shift | moderate | [0.65, 3.14] | 2.497 | 1.90 | Yes | 1.9625 | 0.5127 | 1.4498 |
| mask_shift | severe | [1.54, 4.31] | 2.769 | 2.93 | Yes | 1.8222 | 0.4393 | 1.3829 |
| PSF_blur | mild | [2.19, 3.45] | 1.262 | 2.82 | Yes | 1.8679 | 0.1828 | 1.6851 |
| PSF_blur | moderate | [1.82, 4.34] | 2.516 | 3.08 | Yes | 2.0285 | 0.4993 | 1.5292 |
| PSF_blur | severe | [2.74, 3.99] | 1.243 | 3.36 | Yes | 1.9290 | 0.3844 | 1.5446 |

**Overall CI coverage rate:** 100.0% (PASS: target >= 90%)
