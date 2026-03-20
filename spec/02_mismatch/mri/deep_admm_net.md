# Magnetic Resonance Imaging (MRI) — Deep ADMM-Net + Gradient

**GPU**  **Mismatch**: coil sensitivity maps `[0.9, 1.1] gain per coil`
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
from pwm_core.mismatch.operators import mri_estimate_sensitivities_acs

x_wrong = run_solver('deep_admm_net', y)           # no correction
sens = mri_estimate_sensitivities_acs(y)
calib_cfg = {"sensitivities": sens}
x = run_solver('deep_admm_net', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
