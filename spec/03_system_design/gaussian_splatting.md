# 3D Gaussian Splatting (3DGS) — System Design

```
[Source] → [Forward (3D Gaussian Splatting (3DGS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: posed images (N × H × W × 3, float32)  **Algorithms**: 16 — see `spec/gaussian_splatting.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/public/`

```python
from algorithm_base.gaussian_splatting.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
