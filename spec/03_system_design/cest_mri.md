# CEST MRI — System Design

```
[Source] → [Forward (CEST MRI)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-spectrum (offsets × H × W, float32)  **Algorithms**: 15 — see `spec/cest_mri.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/public/`

```python
from algorithm_base.cest_mri.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
