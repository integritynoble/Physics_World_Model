# Ultrasound B-mode Imaging — PnP-ADMM (NLM denoiser)

**CPU**  *Venkatakrishnan et al. 2013, GlobalSIP*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('pnp_admm_nlm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
