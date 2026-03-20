# Entangled Photon Microscopy — System Design

```
[Source] → [Forward (Entangled Photon Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: coincidence counts (H × W, float32)  **Algorithms**: 15 — see `spec/entangled_photon.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/public/`

```python
from algorithm_base.entangled_photon.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
