# MINFLUX Nanoscopy — System Design

```
[Source] → [Forward (MINFLUX Nanoscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: photon records (N × 5: t,x,y,z,id)  **Algorithms**: 15 — see `spec/minflux.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/minflux/public/`

```python
from algorithm_base.minflux.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
