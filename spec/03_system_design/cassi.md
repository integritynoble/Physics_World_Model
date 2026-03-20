# Coded Aperture Snapshot Spectral Imaging (CASSI) — System Design

```
[Scene] → [Coded aperture + prism] → [Detector] → y
                                          ↓
                   [ADMM / deep unrolling] → x
                          ↓ dispersion calibration
```

**Mismatch**: dispersion step `[1, 5] px`
**Input**: coded snapshot (H × W, float32)  **Algorithms**: 22 — see `spec/cassi.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

```python
from algorithm_base.cassi.solvers import run_solver
from pwm_core.mismatch.operators import cassi_calibrate_step
disp = cassi_calibrate_step(y)
calib_cfg = {"disp_step": float(disp)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
