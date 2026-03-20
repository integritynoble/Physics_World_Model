# X-ray Fluorescence Tomography — System Design

```
[Source] → [Forward (X-ray Fluorescence Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: XRF sinograms (elem × angles × det, float32)  **Algorithms**: 15 — see `spec/xrf_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/public/`

```python
from algorithm_base.xrf_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
