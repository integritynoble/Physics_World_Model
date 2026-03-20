# X-ray NDT (Radiography) — System Design

```
[Source] → [Forward (X-ray NDT (Radiography))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: projection (H × W, float32)  **Algorithms**: 15 — see `spec/xray_ndt.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/public/`

```python
from algorithm_base.xray_ndt.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
