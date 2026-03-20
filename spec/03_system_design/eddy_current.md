# Eddy Current Imaging — System Design

```
[Source] → [Forward (Eddy Current Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: induced voltage (coils × time, float32)  **Algorithms**: 15 — see `spec/eddy_current.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eddy_current/public/`

```python
from algorithm_base.eddy_current.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
