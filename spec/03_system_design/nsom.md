# Near-field Scanning Optical Microscopy (NSOM) — System Design

```
[Source] → [Forward (Near-field Scanning Optical Microscopy (NSOM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: near-field signal (H × W, float32)  **Algorithms**: 15 — see `spec/nsom.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

```python
from algorithm_base.nsom.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
