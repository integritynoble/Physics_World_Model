# Ptychographic Imaging — AutoPhase (DL-PGD)

**GPU**  *Nguyen, T. et al. (2018) Deep learning approach for Fourier ptychography microscopy, Optics Express*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
