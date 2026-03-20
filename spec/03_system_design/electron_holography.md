# Electron Holography — System Design

```
[Source] → [Forward (Electron Holography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: hologram (H × W, float32)  **Algorithms**: 15 — see `spec/electron_holography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_holography/public/`

```python
from algorithm_base.electron_holography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
