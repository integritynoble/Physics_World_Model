# Arterial Spin Labeling (ASL) MRI — System Design

```
[Source] → [Forward (Arterial Spin Labeling (ASL) MRI)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: label-control pairs (2 × H × W, float32)  **Algorithms**: 15 — see `spec/asl_mri.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/asl_mri/public/`

```python
from algorithm_base.asl_mri.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
