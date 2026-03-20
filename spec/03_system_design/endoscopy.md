# Fiber Bundle Endoscopy — System Design

```
[Source] → [Forward (Fiber Bundle Endoscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: image (H × W × 3, uint8)  **Algorithms**: 15 — see `spec/endoscopy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/endoscopy/public/`

```python
from algorithm_base.endoscopy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
