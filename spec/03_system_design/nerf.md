# Neural Radiance Fields (NeRF) — System Design

```
[Source] → [Forward (Neural Radiance Fields (NeRF))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: posed images (N × H × W × 3, float32)  **Algorithms**: 6 — see `spec/nerf.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/public/`

```python
from algorithm_base.nerf.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
