# Ocean Color Remote Sensing — System Design

```
[Source] → [Forward (Ocean Color Remote Sensing)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: radiance (H × W × bands, float32)  **Algorithms**: 15 — see `spec/ocean_color.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_color/public/`

```python
from algorithm_base.ocean_color.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
