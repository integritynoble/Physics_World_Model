# MR Fingerprinting (MRF) — System Design

```
[Source] → [Forward (MR Fingerprinting (MRF))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: signal evolution (T × H × W, complex64)  **Algorithms**: 15 — see `spec/mr_fingerprinting.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/public/`

```python
from algorithm_base.mr_fingerprinting.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
