# Hyperspectral Remote Sensing — System Design

```
[Source] → [Forward (Hyperspectral Remote Sensing)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: radiance cube (H × W × bands, float32)  **Algorithms**: 15 — see `spec/hyperspectral_remote.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/public/`

```python
from algorithm_base.hyperspectral_remote.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
