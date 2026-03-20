# Ptychographic Imaging — PtychoFormer (DL-DRS)

**GPU**  *Shi, J. et al. (2024) PtychoFormer: transformer-based ptychographic reconstruction, Optica*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('ptycho_former', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
