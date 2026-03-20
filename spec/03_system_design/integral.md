# Integral Photography — System Design

```
[Source] → [Forward (Integral Photography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: integral image (H × W, float32)  **Algorithms**: 16 — see `spec/integral.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/public/`

```python
from algorithm_base.integral.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
