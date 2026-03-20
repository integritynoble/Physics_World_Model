# Magnetic Particle Imaging (MPI) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: system function (freq × ch, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/public/`

```python
from algorithm_base.magnetic_particle.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
