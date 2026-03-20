# MR Spectroscopy (MRS) — System Design

```
[Source] → [Forward (MR Spectroscopy (MRS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: FID (T, complex64)  **Algorithms**: 15 — see `spec/mrs.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/public/`

```python
from algorithm_base.mrs.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
