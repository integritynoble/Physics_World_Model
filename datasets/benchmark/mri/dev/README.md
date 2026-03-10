# MRI Dev Tier

**Samples:** 20
**Source:** M4Raw multi-coil brain MRI (Lyu et al., Scientific Data 2023)
**Forward model:** y = U_Omega * F * C * x + n (Cartesian k-space undersampling)

## Mismatch Parameter Ranges

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| acceleration_factor | 3.0 | 6.0 | x |
| noise_sigma | 0.002 | 0.01 |  |
| coil_sensitivity_error | 0.0 | 0.04 | relative |
| off_resonance_hz | -20.0 | 20.0 | Hz |
| trajectory_deviation | -0.01 | 0.01 | relative |

## Samples

- **sample_00**: accel=5.7x, noise_sigma=0.0022, coil_err=0.0237
- **sample_01**: accel=3.7x, noise_sigma=0.0033, coil_err=0.0061
- **sample_02**: accel=5.9x, noise_sigma=0.0032, coil_err=0.0298
- **sample_03**: accel=4.5x, noise_sigma=0.0054, coil_err=0.0212
- **sample_04**: accel=5.2x, noise_sigma=0.0082, coil_err=0.0258
- **sample_05**: accel=6.0x, noise_sigma=0.0057, coil_err=0.0314
- **sample_06**: accel=4.4x, noise_sigma=0.0039, coil_err=0.0247
- **sample_07**: accel=3.9x, noise_sigma=0.0036, coil_err=0.0375
- **sample_08**: accel=4.5x, noise_sigma=0.0032, coil_err=0.0079
- **sample_09**: accel=5.5x, noise_sigma=0.0067, coil_err=0.0009
- **sample_10**: accel=3.3x, noise_sigma=0.0059, coil_err=0.0259
- **sample_11**: accel=3.4x, noise_sigma=0.0050, coil_err=0.0105
- **sample_12**: accel=4.2x, noise_sigma=0.0048, coil_err=0.0001
- **sample_13**: accel=4.6x, noise_sigma=0.0066, coil_err=0.0051
- **sample_14**: accel=5.8x, noise_sigma=0.0087, coil_err=0.0016
- **sample_15**: accel=4.7x, noise_sigma=0.0086, coil_err=0.0300
- **sample_16**: accel=5.7x, noise_sigma=0.0050, coil_err=0.0362
- **sample_17**: accel=3.5x, noise_sigma=0.0061, coil_err=0.0350
- **sample_18**: accel=4.2x, noise_sigma=0.0080, coil_err=0.0032
- **sample_19**: accel=3.9x, noise_sigma=0.0097, coil_err=0.0206
