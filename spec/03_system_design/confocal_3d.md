# Confocal 3D Z-Stack — System Design

```
[Source] → [Forward (Confocal 3D Z-Stack)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 16 — see `spec/confocal_3d.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_3d/public/`

```python
from algorithm_base.confocal_3d.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
