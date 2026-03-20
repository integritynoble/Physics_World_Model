# Wide-Angle X-ray Scattering (WAXS) — System Design

```
[Source] → [Forward (Wide-Angle X-ray Scattering (WAXS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: wide-angle pattern (H × W, float32)  **Algorithms**: 15 — see `spec/waxs.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/public/`

```python
from algorithm_base.waxs.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
