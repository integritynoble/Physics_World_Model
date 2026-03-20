# Diffuse Optical Tomography (DOT) — System Design

```
[Source] → [Forward (Diffuse Optical Tomography (DOT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: boundary flux (sources × detectors, float32)  **Algorithms**: 15 — see `spec/dot.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dot/public/`

```python
from algorithm_base.dot.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
