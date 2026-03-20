# Electron Tomography — System Design

```
[Source] → [Forward (Electron Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: tilt series (angles × H × W, float32)  **Algorithms**: 15 — see `spec/electron_tomography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/public/`

```python
from algorithm_base.electron_tomography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
