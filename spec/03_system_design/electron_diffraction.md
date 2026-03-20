# 4D-STEM Electron Diffraction — System Design

```
[Source] → [Forward (4D-STEM Electron Diffraction)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: diffraction pattern (H × W, float32)  **Algorithms**: 15 — see `spec/electron_diffraction.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/public/`

```python
from algorithm_base.electron_diffraction.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
