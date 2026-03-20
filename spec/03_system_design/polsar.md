# Polarimetric SAR (PolSAR) — System Design

```
[Source] → [Forward (Polarimetric SAR (PolSAR))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: scattering matrix (H × W × 4, complex64)  **Algorithms**: 15 — see `spec/polsar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

```python
from algorithm_base.polsar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
