# Cone-Beam Computed Tomography (CBCT) — System Design

```
[X-ray source] → [Patient] → [Flat-panel] → projections y
                                   ↓
                   [FDK / TV-CBCT] → x
                       ↓ geometry calibration
```

**Mismatch**: source-detector geometry `SAD/SDD ±5 mm`
**Input**: projections (angles × H × W, float32)  **Algorithms**: 22 — see `spec/cbct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
from pwm_core.mismatch.operators import cbct_calibrate_offset
geo = cbct_calibrate_offset(y)
calib_cfg = {"geometry_error": geo}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
