# MINFLUX Nanoscopy — MINFLUX-Net

**GPU**  *Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217*
**Input**: photon records (N × 5: t,x,y,z,id)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/minflux/public/`

```python
from algorithm_base.minflux.solvers import run_solver
x = run_solver('minflux_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
