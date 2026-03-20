# Susceptibility-Weighted Imaging (SWI) — System Design

```
[Source] → [Forward (Susceptibility-Weighted Imaging (SWI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: phase image (H × W × slices, float32)  **Algorithms**: 15 — see `spec/swi.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/swi/public/`

```python
from algorithm_base.swi.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
