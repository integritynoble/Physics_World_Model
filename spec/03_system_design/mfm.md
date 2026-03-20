# Magnetic Force Microscopy (MFM) — System Design

```
[Source] → [Forward (Magnetic Force Microscopy (MFM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: magnetic force map (H × W, float32)  **Algorithms**: 15 — see `spec/mfm.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mfm/public/`

```python
from algorithm_base.mfm.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
