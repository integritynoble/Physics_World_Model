# Functional Near-Infrared Spectroscopy (fNIRS) — System Design

```
[Source] → [Forward (Functional Near-Infrared Spectroscopy (fNIRS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: optical signal (channels × T, float32)  **Algorithms**: 15 — see `spec/nirs_brain.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/public/`

```python
from algorithm_base.nirs_brain.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
