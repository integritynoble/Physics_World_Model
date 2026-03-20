# Particle Calorimetry — DL-Diffusion + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: deposits (N × 5, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/public/`

```python
from algorithm_base.particle_calorimetry.solvers import run_solver


x_wrong = run_solver('dl_diffusion', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_diffusion', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
