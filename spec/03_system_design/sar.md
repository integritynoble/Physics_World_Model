# Synthetic Aperture Radar (SAR) — System Design

```
[Source] → [Forward (Synthetic Aperture Radar (SAR))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: raw data (range × azimuth, complex64)  **Algorithms**: 15 — see `spec/sar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sar/public/`

```python
from algorithm_base.sar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
