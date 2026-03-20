# Focused Ion Beam SEM (FIB-SEM) — System Design

```
[Source] → [Forward (Focused Ion Beam SEM (FIB-SEM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: cross-sections (Z × H × W, uint8)  **Algorithms**: 15 — see `spec/fib_sem.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/public/`

```python
from algorithm_base.fib_sem.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
