# Cryo-Electron Tomography (Cryo-ET) — System Design

```
[Source] → [Forward (Cryo-Electron Tomography (Cryo-ET))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: tilt series (angles × H × W, float32)  **Algorithms**: 15 — see `spec/cryo_et.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_et/public/`

```python
from algorithm_base.cryo_et.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
