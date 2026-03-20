# Gravitational Wave Detection — System Design

```
[Source] → [Forward (Gravitational Wave Detection)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: strain (samples, float32)  **Algorithms**: 15 — see `spec/gravitational_wave.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/public/`

```python
from algorithm_base.gravitational_wave.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
