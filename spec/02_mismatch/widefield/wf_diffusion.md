# Widefield Fluorescence Microscopy — WF-Diffusion (PnP-PGD DRUNet) + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver


x_wrong = run_solver('wf_diffusion', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('wf_diffusion', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
