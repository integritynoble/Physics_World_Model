# Portal Imaging (EPID) — System Design

```
[Source] → [Forward (Portal Imaging (EPID))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: EPID projection (H × W, float32)  **Algorithms**: 15 — see `spec/portal_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/public/`

```python
from algorithm_base.portal_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
