# Brachytherapy Imaging — DiffusionRecon + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: dose map (H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/public/`

```python
from algorithm_base.brachytherapy_img.solvers import run_solver


x_wrong = run_solver('dl_diffusion', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_diffusion', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
