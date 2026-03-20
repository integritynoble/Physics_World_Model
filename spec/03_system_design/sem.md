# Scanning Electron Microscopy (SEM) — System Design

```
[Source] → [Forward (Scanning Electron Microscopy (SEM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: SEM image (H × W, uint16)  **Algorithms**: 15 — see `spec/sem.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
