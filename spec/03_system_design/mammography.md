# Mammography — System Design

```
[Source] → [Forward (Mammography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: projection pair (2 × H × W, float32)  **Algorithms**: 15 — see `spec/mammography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mammography/public/`

```python
from algorithm_base.mammography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
