# Polarization Microscopy — System Design

```
[Source] → [Forward (Polarization Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Stokes images (4 × H × W, float32)  **Algorithms**: 15 — see `spec/polarization.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polarization/public/`

```python
from algorithm_base.polarization.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
