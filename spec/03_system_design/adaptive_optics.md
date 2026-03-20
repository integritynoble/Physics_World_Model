# Adaptive Optics (AO) Imaging — System Design

```
[Source] → [Forward (Adaptive Optics (AO) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: wavefront sensor (H × W, float32)  **Algorithms**: 15 — see `spec/adaptive_optics.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/public/`

```python
from algorithm_base.adaptive_optics.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
