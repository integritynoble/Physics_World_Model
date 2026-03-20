# Stimulated Raman Scattering (SRS) Microscopy — System Design

```
[Source] → [Forward (Stimulated Raman Scattering (SRS) Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: SRS image (H × W, float32)  **Algorithms**: 15 — see `spec/srs.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/srs/public/`

```python
from algorithm_base.srs.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
