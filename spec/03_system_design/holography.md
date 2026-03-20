# Digital Holographic Microscopy — System Design

```
[Source] → [Forward (Digital Holographic Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: hologram (H × W, float32)  **Algorithms**: 17 — see `spec/holography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
