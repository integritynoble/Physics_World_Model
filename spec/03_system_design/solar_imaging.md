# Solar EUV/X-ray Imaging — System Design

```
[Source] → [Forward (Solar EUV/X-ray Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: EUV image (H × W, float32)  **Algorithms**: 15 — see `spec/solar_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/public/`

```python
from algorithm_base.solar_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
