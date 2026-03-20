# Diffusion MRI (DTI) — MedMamba + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: DWI (N_dirs × H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/public/`

```python
from algorithm_base.diffusion_mri.solvers import run_solver


x_wrong = run_solver('dl_mamba', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_mamba', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
