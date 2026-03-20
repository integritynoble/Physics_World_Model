# Dual-Energy X-ray Absorptiometry (DEXA) — System Design

```
[Source] → [Forward (Dual-Energy X-ray Absorptiometry (DEXA))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: dual-energy projections (2 × H × W, float32)  **Algorithms**: 15 — see `spec/dexa.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/public/`

```python
from algorithm_base.dexa.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
