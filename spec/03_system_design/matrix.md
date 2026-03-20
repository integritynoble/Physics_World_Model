# Generic Matrix Sensing — System Design

```
[Source] → [Forward (Generic Matrix Sensing)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: partial matrix (M × N, float32, NaN=missing)  **Algorithms**: 16 — see `spec/matrix.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/public/`

```python
from algorithm_base.matrix.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
