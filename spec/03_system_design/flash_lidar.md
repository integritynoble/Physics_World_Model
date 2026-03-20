# Flash LiDAR — System Design

```
[Source] → [Forward (Flash LiDAR)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: range + intensity (H × W × 2, float32)  **Algorithms**: 15 — see `spec/flash_lidar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/public/`

```python
from algorithm_base.flash_lidar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
