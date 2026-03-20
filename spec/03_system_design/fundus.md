# Fundus Camera — System Design

```
[Source] → [Forward (Fundus Camera)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: photograph (H × W × 3, uint8)  **Algorithms**: 15 — see `spec/fundus.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/public/`

```python
from algorithm_base.fundus.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
