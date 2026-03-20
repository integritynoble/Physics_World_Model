# Dark-Field Microscopy — System Design

```
[Source] → [Forward (Dark-Field Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: grating images (2 × H × W, float32)  **Algorithms**: 15 — see `spec/dark_field.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/public/`

```python
from algorithm_base.dark_field.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
