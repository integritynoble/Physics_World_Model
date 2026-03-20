# Machine Vision / AOI — System Design

```
[Source] → [Forward (Machine Vision / AOI)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: image (H × W × 3, uint8)  **Algorithms**: 15 — see `spec/machine_vision.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/machine_vision/public/`

```python
from algorithm_base.machine_vision.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
