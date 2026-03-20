# Ptychographic Imaging — PtychoNN (DL-PGD)

**GPU**  *Cherukara, M.J. et al. (2020) AI-enabled high-resolution scanning coherent imaging, Applied Physics Letters*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
