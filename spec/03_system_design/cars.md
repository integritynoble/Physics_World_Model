# Coherent Anti-Stokes Raman (CARS) Microscopy — System Design

```
[Source] → [Forward (Coherent Anti-Stokes Raman (CARS) Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: CARS spectrum cube (H × W × λ, float32)  **Algorithms**: 15 — see `spec/cars.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

```python
from algorithm_base.cars.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
