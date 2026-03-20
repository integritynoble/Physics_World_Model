# Lucky Imaging — System Design

```
[Source] → [Forward (Lucky Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: speckle frames (N × H × W, float32)  **Algorithms**: 15 — see `spec/lucky_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lucky_imaging/public/`

```python
from algorithm_base.lucky_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
