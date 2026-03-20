# MR Elastography (MRE) — System Design

```
[Source] → [Forward (MR Elastography (MRE))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: wave images (slices × H × W, complex64)  **Algorithms**: 15 — see `spec/mr_elastography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/public/`

```python
from algorithm_base.mr_elastography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
