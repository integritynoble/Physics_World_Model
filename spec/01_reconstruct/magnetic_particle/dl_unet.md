# Magnetic Particle Imaging (MPI) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: system function (freq × ch, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/public/`

```python
from algorithm_base.magnetic_particle.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
