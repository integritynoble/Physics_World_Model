# MR Angiography (MRA) — System Design

```
[Source] → [Forward (MR Angiography (MRA))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: k-space (kx × ky × kz, complex64)  **Algorithms**: 15 — see `spec/mra.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mra/public/`

```python
from algorithm_base.mra.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
