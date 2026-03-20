# Magnetic Resonance Imaging (MRI) — E2E-VarNet + Gradient

**GPU**  **PSNR**: ~40.5 dB  **Mismatch**: coil sensitivity maps `[0.9, 1.1] gain per coil`
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
from pwm_core.mismatch.operators import mri_estimate_sensitivities_acs

x_wrong = run_solver('varnet', y)           # no correction
sens = mri_estimate_sensitivities_acs(y)
calib_cfg = {"sensitivities": sens}
x = run_solver('varnet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
