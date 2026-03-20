# Coded Aperture Compressive Temporal Imaging (CACTI) — System Design

```
[Source] → [Forward (Coded Aperture Compressive Temporal Imaging (CACTI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: frame timing offset `[-1, +1] frames`
**Input**: coded frames (B × H × W, float32)  **Algorithms**: 7 — see `spec/cacti.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/public/`

```python
from algorithm_base.cacti.solvers import run_solver
from pwm_core.mismatch.operators import cacti_calibrate_timing
timing = cacti_calibrate_timing(y)
calib_cfg = {"timing_offset": timing}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
