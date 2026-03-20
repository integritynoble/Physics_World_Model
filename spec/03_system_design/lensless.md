# Lensless (Diffuser Camera) Imaging — System Design

```
[LED Source] → [2D Object] → [Phase Diffuser (PSF)] → [Bare CMOS] → [12-bit ADC] → y
                                    ↓                      ↓
                              [Convolution            [Poisson noise
                               y = H * x]              + readout σ=3 e⁻]
```

**Mismatch**: PSF shift `[-5, +5] px`
**Input**: diffuser measurement (H × W, float32)  **Algorithms**: 17 — see `spec/lensless.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`
**Paper**: `papers/system_design/outputs/lensless_forward_v1_iter1.md`

```python
from algorithm_base.lensless.solvers import run_solver
from pwm_core.mismatch.operators import lensless_calibrate_shift
shift = lensless_calibrate_shift(y)
calib_cfg = {"psf_shift": shift}
x = run_solver('wiener', y, cfg=calib_cfg)
```
