# Industrial X-ray CT — System Design

```
[Source] → [Forward (Industrial X-ray CT)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: center-of-rotation offset `[-10, +10] px`
**Input**: sinogram (angles × detectors, float32)  **Algorithms**: 15 — see `spec/industrial_ct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/public/`

```python
from algorithm_base.industrial_ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor
cor_offset = ct_calibrate_cor(y, shift_range=10)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
