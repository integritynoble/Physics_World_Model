# Fourier Ptychographic Microscopy (FPM) — System Design

```
[LED array] → [Sample] → [Low-NA images] → y
                                ↓
         [Ptychographic phase retrieval] → x
                    ↓ LED position calibration
```

**Mismatch**: LED position error `[-2, +2] mm`
**Input**: LED images (N_leds × H × W, float32)  **Algorithms**: 16 — see `spec/fpm.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/public/`

```python
from algorithm_base.fpm.solvers import run_solver
from pwm_core.mismatch.operators import fpm_calibrate_led_pos
led_err = fpm_calibrate_led_pos(y)
calib_cfg = {"led_pos_error": led_err}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
