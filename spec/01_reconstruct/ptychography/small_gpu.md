# Ptychographic Imaging — PtychoNN 2.0 (DnCNN)

**GPU**  *Wu, L. et al. (2022) PtychoNN 2.0: on-the-fly neural network-based reconstruction, Journal of Applied Crystallography*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
