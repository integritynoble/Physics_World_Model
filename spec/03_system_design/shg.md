# Second Harmonic Generation (SHG) Microscopy — System Design

```
[Source] → [Forward (Second Harmonic Generation (SHG) Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 15 — see `spec/shg.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/public/`

```python
from algorithm_base.shg.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
