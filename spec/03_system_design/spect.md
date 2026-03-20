# Single Photon Emission CT (SPECT) — System Design

```
[Radiotracer] → [Gamma camera] → [Projections] → y
                                       ↓
                       [OS-EM / MLEM] → x
                            ↓ scatter calibration
```

**Mismatch**: scatter fraction `[0.1, 0.4]`
**Input**: projections (angles × detectors, float32)  **Algorithms**: 15 — see `spec/spect.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
from pwm_core.mismatch.operators import spect_calibrate_scatter
scatter_frac = spect_calibrate_scatter(y)
calib_cfg = {"scatter_frac": float(scatter_frac)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
