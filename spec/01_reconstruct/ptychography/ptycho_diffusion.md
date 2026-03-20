# Ptychographic Imaging — Ptychography Diffusion (DL-PGD)

**GPU**  *Cherukara, M.J. et al. (2023) Diffusion model for ptychographic phase retrieval, Nature Computational Science*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('ptycho_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
