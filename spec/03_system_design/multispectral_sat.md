# Multispectral Satellite Imaging — System Design

```
[Source] → [Forward (Multispectral Satellite Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: radiance (H × W × bands, float32)  **Algorithms**: 15 — see `spec/multispectral_sat.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/public/`

```python
from algorithm_base.multispectral_sat.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
