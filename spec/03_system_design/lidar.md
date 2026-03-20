# LiDAR Scanner — System Design

```
[Source] → [Forward (LiDAR Scanner)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: point cloud (N × 3, float32)  **Algorithms**: 15 — see `spec/lidar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lidar/public/`

```python
from algorithm_base.lidar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
