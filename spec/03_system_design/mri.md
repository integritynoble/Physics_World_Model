# Magnetic Resonance Imaging (MRI) — System Design

```
[RF coils] → [Spin ensemble] → [k-space y]
                                    ↓
                [ESPIRiT / CS-MRI] → x
                       ↓ coil sensitivity calibration
```

**Mismatch**: coil sensitivity maps `[0.9, 1.1] gain per coil`
**Input**: k-space (H × W × 2: real+imag, float32)  **Algorithms**: 41 — see `spec/mri.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
from pwm_core.mismatch.operators import mri_estimate_sensitivities_acs
sens = mri_estimate_sensitivities_acs(y)
calib_cfg = {"sensitivities": sens}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
