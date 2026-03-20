# Functional MRI (BOLD fMRI) — System Design

```
[Source] → [Forward (Functional MRI (BOLD fMRI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: BOLD volumes (T × H × W × D, float32)  **Algorithms**: 15 — see `spec/fmri.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fmri/public/`

```python
from algorithm_base.fmri.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
