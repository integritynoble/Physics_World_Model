# Raman Imaging / Microscopy — System Design

```
[Source] → [Forward (Raman Imaging / Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Raman spectra (H × W × wn, float32)  **Algorithms**: 15 — see `spec/raman_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/`

```python
from algorithm_base.raman_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
