# Structured-Light Depth Camera — System Design

```
[Source] → [Forward (Structured-Light Depth Camera)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: pattern images (N × H × W, float32)  **Algorithms**: 15 — see `spec/structured_light.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/public/`

```python
from algorithm_base.structured_light.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
