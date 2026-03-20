# Two-Photon / Multiphoton Microscopy — System Design

```
[Source] → [Forward (Two-Photon / Multiphoton Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 15 — see `spec/two_photon.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/two_photon/public/`

```python
from algorithm_base.two_photon.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
