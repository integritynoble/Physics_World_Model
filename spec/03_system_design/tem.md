# Transmission Electron Microscopy (TEM) — System Design

```
[Source] → [Forward (Transmission Electron Microscopy (TEM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: TEM image (H × W, float32)  **Algorithms**: 15 — see `spec/tem.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

```python
from algorithm_base.tem.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
