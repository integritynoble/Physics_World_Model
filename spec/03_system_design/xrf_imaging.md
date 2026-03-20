# X-ray Fluorescence (XRF) Imaging — System Design

```
[Source] → [Forward (X-ray Fluorescence (XRF) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: fluorescence map (H × W × elements, float32)  **Algorithms**: 15 — see `spec/xrf_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/public/`

```python
from algorithm_base.xrf_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
