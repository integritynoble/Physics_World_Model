# X-ray Computed Tomography (CT) — System Design

```
[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]
       ↓                      ↓                        ↓
  [Polychromatic         [Beer-Lambert           [CoR offset
   beam hardening]        attenuation]            mismatch]
                                                       ↓
                              → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y
                                        ↓
                                  [Poisson I0=1e4]
                                  [Gaussian σ=3 e⁻]
                                  [Dark current 0.05 e⁻/s]
```

**Mismatch**: center-of-rotation offset `[-5, +5] px`
**Input**: sinogram (angles × detectors, float32)  **Algorithms**: 41 — see `spec/ct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`
**Paper**: `papers/system_design/outputs/ct_forward_v1_iter1.md`

```python
from algorithm_base.ct.solvers import run_solver
from pwm_core.mismatch.operators import ct_calibrate_cor
cor_offset = ct_calibrate_cor(y, shift_range=5)
calib_cfg = {"cor_offset": float(cor_offset)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
