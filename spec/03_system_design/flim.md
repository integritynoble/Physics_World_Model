# Fluorescence Lifetime Imaging (FLIM) — System Design

```
[Source] → [Forward (Fluorescence Lifetime Imaging (FLIM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: photon arrivals (H × W × T, float32)  **Algorithms**: 16 — see `spec/flim.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/flim/public/`

```python
from algorithm_base.flim.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
