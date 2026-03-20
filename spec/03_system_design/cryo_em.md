# Cryo-EM Single Particle Analysis — System Design

```
[Source] → [Forward (Cryo-EM Single Particle Analysis)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: defocus value `[0.5, 3.0] μm`
**Input**: particle images (N × H × W, float32)  **Algorithms**: 17 — see `spec/cryo_em.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/public/`

```python
from algorithm_base.cryo_em.solvers import run_solver
from pwm_core.mismatch.operators import cryoem_calibrate_defocus
defocus = cryoem_calibrate_defocus(y)
calib_cfg = {"defocus": float(defocus)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
