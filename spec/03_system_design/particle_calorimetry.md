# Particle Calorimetry — System Design

```
[Source] → [Forward (Particle Calorimetry)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: deposits (N × 5, float32)  **Algorithms**: 15 — see `spec/particle_calorimetry.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/public/`

```python
from algorithm_base.particle_calorimetry.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
