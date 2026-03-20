# Atomic Force Microscopy (AFM) — System Design

```
[Source] → [Forward (Atomic Force Microscopy (AFM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: force-distance map (H × W, float32)  **Algorithms**: 15 — see `spec/afm.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/public/`

```python
from algorithm_base.afm.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
