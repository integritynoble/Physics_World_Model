# PET/CT Fusion — PET-CT-Fusion-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: PET sino + CT proj (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_ct/public/`

```python
from algorithm_base.pet_ct.solvers import run_solver


x_wrong = run_solver('petct_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('petct_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
