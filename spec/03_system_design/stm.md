# Scanning Tunneling Microscopy (STM) — System Design

```
[Source] → [Forward (Scanning Tunneling Microscopy (STM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: tunneling map (H × W, float32)  **Algorithms**: 15 — see `spec/stm.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
