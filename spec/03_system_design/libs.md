# Laser-Induced Breakdown Spectroscopy (LIBS) Imaging — System Design

```
[Source] → [Forward (Laser-Induced Breakdown Spectroscopy (LIBS) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: emission spectrum (wavelengths, float32)  **Algorithms**: 15 — see `spec/libs.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/public/`

```python
from algorithm_base.libs.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
