# Fluoroscopy — System Design

```
[Source] → [Forward (Fluoroscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: X-ray frames (T × H × W, float32)  **Algorithms**: 15 — see `spec/fluoroscopy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/public/`

```python
from algorithm_base.fluoroscopy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
