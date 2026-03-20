# Coherent Diffractive Imaging / Phase Retrieval — System Design

```
[Coherent beam] → [Sample] → [Detector] → y
                                  ↓
           [HIO / RAAR / ERα] → x
                  ↓ distance calibration
```

**Mismatch**: detector-sample distance error `[-5, +5] mm`
**Input**: diffraction intensities (H × W, float32)  **Algorithms**: 16 — see `spec/phase_retrieval.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/public/`

```python
from algorithm_base.phase_retrieval.solvers import run_solver
from pwm_core.mismatch.operators import phase_calibrate_distance
dist_err = phase_calibrate_distance(y)
calib_cfg = {"dist_error": float(dist_err)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
