# Small-Angle X-ray Scattering (SAXS) — System Design

```
[Source] → [Forward (Small-Angle X-ray Scattering (SAXS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: scattering pattern (H × W, float32)  **Algorithms**: 15 — see `spec/saxs.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/saxs/public/`

```python
from algorithm_base.saxs.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
