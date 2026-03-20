# TIRF Microscopy — System Design

```
[Source] → [Forward (TIRF Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: TIRF frames (T × H × W, float32)  **Algorithms**: 15 — see `spec/tirf.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tirf/public/`

```python
from algorithm_base.tirf.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
