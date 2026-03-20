# Time-of-Flight Depth Camera — System Design

```
[Source] → [Forward (Time-of-Flight Depth Camera)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: depth + amplitude (H × W × 2, float32)  **Algorithms**: 15 — see `spec/tof_camera.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tof_camera/public/`

```python
from algorithm_base.tof_camera.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
