# Compressed Ultrafast Photography (CUP) — System Design

```
[Source] → [Forward (Compressed Ultrafast Photography (CUP))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: streak image (H × W_streak, float32)  **Algorithms**: 15 — see `spec/cup.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/public/`

```python
from algorithm_base.cup.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
