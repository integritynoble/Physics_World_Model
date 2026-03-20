# High Dynamic Range (HDR) Imaging — System Design

```
[Source] → [Forward (High Dynamic Range (HDR) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: multi-exposure (K × H × W × 3, uint8)  **Algorithms**: 15 — see `spec/hdr_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/public/`

```python
from algorithm_base.hdr_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
