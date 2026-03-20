# Confocal Live-Cell Microscopy — System Design

```
[Source] → [Forward (Confocal Live-Cell Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: time-lapse (T × H × W, float32)  **Algorithms**: 16 — see `spec/confocal_livecell.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/public/`

```python
from algorithm_base.confocal_livecell.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
