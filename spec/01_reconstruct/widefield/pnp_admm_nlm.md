# Widefield Fluorescence Microscopy — PnP-ADMM (NLM denoiser)

**CPU**  *Venkatakrishnan et al. 2013, GlobalSIP*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('pnp_admm_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
