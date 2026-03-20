# Magnetic Particle Imaging (MPI) — System Design

```
[Source] → [Forward (Magnetic Particle Imaging (MPI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: system function (freq × ch, complex64)  **Algorithms**: 15 — see `spec/magnetic_particle.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/public/`

```python
from algorithm_base.magnetic_particle.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
