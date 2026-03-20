# Three-Photon Microscopy — System Design

```
[Source] → [Forward (Three-Photon Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 15 — see `spec/three_photon.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/three_photon/public/`

```python
from algorithm_base.three_photon.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
