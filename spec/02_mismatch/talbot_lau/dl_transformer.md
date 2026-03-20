# Talbot-Lau X-ray Grating Interferometry — Phase-Transformer + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: stepping images (N_steps × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/public/`

```python
from algorithm_base.talbot_lau.solvers import run_solver


x_wrong = run_solver('dl_transformer', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_transformer', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
