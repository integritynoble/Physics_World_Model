# X-ray Crystallography — System Design

```
[Source] → [Forward (X-ray Crystallography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: structure factors (hkl × F, float32)  **Algorithms**: 15 — see `spec/xray_crystallography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/public/`

```python
from algorithm_base.xray_crystallography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
