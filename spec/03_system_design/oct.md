# Optical Coherence Tomography (OCT) — System Design

```
[Broadband source] → [Interferometer] → [Spectrometer] → y
                                               ↓
                         [iFFT / BM3D-deconv] → x
                                ↓ dispersion compensation
```

**Mismatch**: dispersion coefficients `β₂ ∈ [-1e-27, 1e-27] s²/m`
**Input**: spectrum (wavenumbers × A-scans, float32)  **Algorithms**: 16 — see `spec/oct.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

```python
from algorithm_base.oct.solvers import run_solver
from pwm_core.mismatch.operators import oct_calibrate_dispersion
disp = oct_calibrate_dispersion(y)
calib_cfg = {"disp_coeff": disp}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
