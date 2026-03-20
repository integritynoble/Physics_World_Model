# DESI Mass Spectrometry Imaging — System Design

```
[Source] → [Forward (DESI Mass Spectrometry Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: mass image (H × W × m/z, float32)  **Algorithms**: 15 — see `spec/desi.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/desi/public/`

```python
from algorithm_base.desi.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
